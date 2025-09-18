#!/usr/bin/env python3
"""Resilient Groq API client with rate limiting, caching, and CLI utilities.

This module provides an asynchronous Groq client that satisfies the requirements
outlined in the repository README:

* Per-key token buckets derived from RPM.
* Respect for `Retry-After` headers with support for HTTP-date values.
* Exponential backoff with decorrelated jitter for transient errors.
* Persistent JSON cache to deduplicate identical chat requests.
* Bounded concurrency via an asyncio semaphore.
* Optional soft daily token budget accounting.
* Inline documentation and offline self-tests accessible with
  ``python groq_client_compliant.py --self-test``.

The client exposes :class:`GroqClient` for programmatic use and a CLI that can
read prompts from positional arguments, files, stdin, or defaults. When no
prompt is supplied the script runs its self-tests automatically, ensuring CI
friendliness out of the box.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import email.utils
import hashlib
import json
import logging
import os
import random
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

import httpx


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


class ColorFormatter(logging.Formatter):
    """Minimal ANSI color formatter without third-party dependencies."""

    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }

    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if sys.stderr.isatty():
            color = self.COLORS.get(record.levelno)
            if color:
                return f"{color}{message}{self.RESET}"
        return message


def build_logger(name: str = "groq-client", level: int = logging.INFO) -> logging.Logger:
    """Create a logger configured for CLI usage."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


LOGGER = build_logger()


# ---------------------------------------------------------------------------
# Retry-After parsing and backoff helpers
# ---------------------------------------------------------------------------


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse a ``Retry-After`` header value into seconds.

    The header can be either an integer/float number of seconds or an HTTP-date
    string. Returns ``None`` if the value cannot be parsed.
    """

    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        seconds = float(value)
        if seconds >= 0:
            return seconds
    except ValueError:
        pass

    # Try HTTP-date
    parsed = email.utils.parsedate_to_datetime(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    delta = parsed.timestamp() - datetime.now(timezone.utc).timestamp()
    return max(delta, 0.0)


def decorrelated_jitter(previous: float, base: float, cap: float) -> float:
    """Decorrelated jitter backoff as described by AWS Architecture blog.

    Args:
        previous: Previous sleep interval in seconds.
        base: Minimum bound for new sleep.
        cap: Maximum bound for new sleep.
    """

    previous = max(previous, base)
    return min(cap, random.uniform(base, previous * 3))


def estimate_tokens(messages: Iterable[Dict[str, Any]]) -> int:
    """Rudimentary token estimator based on message text length."""

    total_chars = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            total_chars += sum(len(str(chunk)) for chunk in content)
        else:
            total_chars += len(str(content))
    return max(1, total_chars // 4)


# ---------------------------------------------------------------------------
# Token bucket and key management
# ---------------------------------------------------------------------------


class TokenBucket:
    """Async token bucket enforcing requests-per-minute limits."""

    def __init__(
        self,
        rpm: int,
        *,
        time_func: Optional[callable] = None,
        sleep_func: Optional[callable] = None,
    ) -> None:
        if rpm <= 0:
            raise ValueError("RPM must be positive")
        self.capacity = float(rpm)
        self.tokens = float(rpm)
        self.refill_rate = float(rpm) / 60.0
        self._time = time_func or time.monotonic
        self._sleep = sleep_func or asyncio.sleep
        self._last_refill = self._time()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._time()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        if tokens <= 0:
            return
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                deficit = tokens - self.tokens
                wait_for = deficit / self.refill_rate
            await self._sleep(wait_for)


@dataclass
class KeyState:
    key: str
    bucket: TokenBucket
    cooldown_until: float = 0.0


class KeyManager:
    """Round-robin key allocator respecting per-key cooldowns."""

    def __init__(self, keys: Iterable[str], rpm_per_key: int) -> None:
        keys = [k.strip() for k in keys if k.strip()]
        if not keys:
            raise ValueError("At least one API key is required")
        self._states: List[KeyState] = [
            KeyState(key=k, bucket=TokenBucket(rpm_per_key)) for k in keys
        ]
        self._index = 0
        self._lock = asyncio.Lock()
        self._time = time.monotonic

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[str]:
        state = await self._next_available_state()
        await state.bucket.acquire()
        try:
            yield state.key
        finally:
            pass

    async def _next_available_state(self) -> KeyState:
        while True:
            async with self._lock:
                now = self._time()
                candidate: Optional[KeyState] = None
                min_wait: Optional[float] = None
                for _ in range(len(self._states)):
                    state = self._states[self._index]
                    self._index = (self._index + 1) % len(self._states)
                    remaining = state.cooldown_until - now
                    if remaining > 0:
                        if min_wait is None or remaining < min_wait:
                            min_wait = remaining
                        continue
                    candidate = state
                    break
            if candidate:
                return candidate
            await asyncio.sleep(min_wait or 0.1)

    def apply_cooldown(self, key: str, seconds: float) -> None:
        if seconds <= 0:
            return
        until = self._time() + seconds
        for state in self._states:
            if state.key == key:
                state.cooldown_until = max(state.cooldown_until, until)
                break


# ---------------------------------------------------------------------------
# Persistent cache
# ---------------------------------------------------------------------------


class ResponseCache:
    """Tiny JSON-backed cache keyed by request payload hash."""

    def __init__(
        self,
        path: Path,
        *,
        ttl: int = 6 * 60 * 60,
        time_func: Optional[callable] = None,
    ) -> None:
        self.path = path
        self.ttl = ttl
        self._time = time_func or time.time
        self._lock = asyncio.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    @staticmethod
    def make_key(payload: Dict[str, Any]) -> str:
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return digest

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return
        now = self._time()
        for key, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            ts = entry.get("ts")
            if ts is None or now - ts > self.ttl:
                continue
            self._data[key] = entry

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self._data, fh, ensure_ascii=False, indent=2)
            tmp.replace(self.path)
        except OSError:
            LOGGER.warning("Failed to persist cache to %s", self.path)

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._data.get(key)
            if not entry:
                return None
            now = self._time()
            if now - entry.get("ts", 0) > self.ttl:
                self._data.pop(key, None)
                self._write()
                return None
            return entry.get("value")

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._data[key] = {"ts": self._time(), "value": value}
            self._write()


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------


class GroqClient:
    """Async Groq client implementing rate limiting and resilience features."""

    API_BASE = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        *,
        keys: Iterable[str],
        rpm_per_key: int = 30,
        max_concurrency: int = 4,
        cache_path: Optional[Path] = None,
        cache_ttl: int = 6 * 60 * 60,
        budget: Optional[int] = None,
        timeout: float = 60.0,
        logger: Optional[logging.Logger] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
    ) -> None:
        self._logger = logger or LOGGER
        self._client = httpx.AsyncClient(base_url=base_url or self.API_BASE, timeout=timeout)
        self._key_manager = KeyManager(keys, rpm_per_key)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._cache = ResponseCache(cache_path, ttl=cache_ttl) if cache_path else None
        self._budget_limit = budget
        self._budget_lock = asyncio.Lock()
        self._budget_consumed = 0
        self._max_retries = max(1, max_retries)
        self._backoff_base = 1.0
        self._backoff_cap = 30.0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "GroqClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:  # type: ignore[override]
        await self.aclose()

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **extra: Any,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(extra)

        cache_key = self._cache.make_key(payload) if self._cache else None
        if cache_key and (cached := await self._cache.get(cache_key)):
            self._logger.info("cache hit for request")
            return cached["text"]

        reservation = await self._reserve_budget(estimate_tokens(messages))

        async with self._semaphore:
            async with self._key_manager.acquire() as key:
                response_data = await self._send_request(key, payload)

        text = self._extract_text(response_data)
        if cache_key and self._cache:
            await self._cache.set(cache_key, {"text": text, "response": response_data})

        usage = response_data.get("usage", {}).get("total_tokens") if isinstance(response_data, dict) else None
        await self._adjust_budget(reservation, usage)

        return text

    async def _reserve_budget(self, estimate: int) -> int:
        if not self._budget_limit:
            return 0
        async with self._budget_lock:
            if self._budget_consumed + estimate > self._budget_limit:
                raise RuntimeError("Token budget exhausted for the day")
            self._budget_consumed += estimate
        return estimate

    async def _adjust_budget(self, reserved: int, actual: Optional[int]) -> None:
        if not self._budget_limit or actual is None:
            return
        diff = actual - reserved
        if diff == 0:
            return
        async with self._budget_lock:
            self._budget_consumed = max(0, self._budget_consumed + diff)

    async def _send_request(self, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {key}"}
        sleep = self._backoff_base
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                last_exc = exc
                self._logger.warning(
                    "network error (attempt %s/%s): %s", attempt, self._max_retries, exc
                )
                await asyncio.sleep(sleep)
                sleep = decorrelated_jitter(sleep, self._backoff_base, self._backoff_cap)
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise RuntimeError("Groq API returned malformed JSON") from exc

            if response.status_code == 429:
                retry_after = parse_retry_after(response.headers.get("Retry-After")) or sleep
                self._logger.warning(
                    "received 429; cooling down key for %.2fs", retry_after
                )
                self._key_manager.apply_cooldown(key, retry_after)
                await asyncio.sleep(retry_after)
                sleep = decorrelated_jitter(sleep, self._backoff_base, self._backoff_cap)
                continue

            if 500 <= response.status_code < 600:
                self._logger.warning(
                    "server error %s (attempt %s/%s)", response.status_code, attempt, self._max_retries
                )
                await asyncio.sleep(sleep)
                sleep = decorrelated_jitter(sleep, self._backoff_base, self._backoff_cap)
                continue

            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(f"Groq API error {response.status_code}: {detail}")

        if last_exc:
            raise RuntimeError("Request failed after retries") from last_exc
        raise RuntimeError("Request failed after retries")

    @staticmethod
    def _extract_text(response: Dict[str, Any]) -> str:
        try:
            choices = response["choices"]
            if not choices:
                raise KeyError("choices")
            message = choices[0]["message"]
            return message.get("content", "")
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response payload: {response}") from exc


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------


def load_api_keys() -> List[str]:
    keys_env = os.getenv("GROQ_API_KEYS")
    if keys_env:
        return [k.strip() for k in keys_env.split(",") if k.strip()]
    single = os.getenv("GROQ_API_KEY")
    return [single.strip()] if single else []


def resolve_prompt(args: argparse.Namespace, stdin_text: Optional[str]) -> Optional[str]:
    if getattr(args, "prompt", None):
        return args.prompt
    if getattr(args, "prompt_file", None):
        path = Path(args.prompt_file)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()
    if getattr(args, "stdin", False):
        if stdin_text is None:
            stdin_text = sys.stdin.read()
        return stdin_text.strip()
    if getattr(args, "default_prompt", None):
        return args.default_prompt
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Groq API client with rate-limit compliance")
    parser.add_argument("prompt", nargs="?", help="Prompt text (overrides other inputs)")
    parser.add_argument("--prompt-file", help="Read prompt from file")
    parser.add_argument("--stdin", action="store_true", help="Read prompt from STDIN")
    parser.add_argument("--default-prompt", help="Fallback prompt when no other inputs")
    parser.add_argument("--model", default="llama3-8b-8192", help="Model name")
    parser.add_argument("--rpm", type=int, default=30, help="Per-key RPM")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent requests")
    parser.add_argument("--budget", type=int, help="Optional daily token budget")
    parser.add_argument("--cache", default=".groq_cache.json", help="Cache file path")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, help="Max tokens in response")
    parser.add_argument("--base-url", default=GroqClient.API_BASE, help="Override API base URL")
    parser.add_argument("--self-test", action="store_true", help="Run built-in tests and exit")
    return parser


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


class FakeClock:
    """Utility for deterministic timing during tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.now += delay

    def advance(self, delay: float) -> None:
        self.now += delay


class SelfTests(unittest.TestCase):
    def test_retry_after_seconds(self) -> None:
        self.assertEqual(parse_retry_after("120"), 120.0)

    def test_retry_after_http_date(self) -> None:
        target = time.time() + 42
        header = email.utils.format_datetime(datetime.fromtimestamp(target, tz=timezone.utc))
        parsed = parse_retry_after(header)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertAlmostEqual(parsed, 42, delta=1)

    def test_decorrelated_jitter_bounds(self) -> None:
        random.seed(1)
        value = decorrelated_jitter(1, 1, 5)
        self.assertGreaterEqual(value, 1)
        self.assertLessEqual(value, 5)

    def test_cache_roundtrip_and_expiry(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            cache = ResponseCache(path, ttl=10, time_func=clock.time)
            key = cache.make_key({"foo": "bar"})

            asyncio.run(cache.set(key, {"text": "hello"}))
            result = asyncio.run(cache.get(key))
            self.assertEqual(result, {"text": "hello"})

            clock.advance(11)
            result_after = asyncio.run(cache.get(key))
            self.assertIsNone(result_after)

    def test_token_bucket_waits(self) -> None:
        clock = FakeClock()
        bucket = TokenBucket(60, time_func=clock.monotonic, sleep_func=clock.sleep)

        async def scenario() -> float:
            for _ in range(int(bucket.capacity)):
                await bucket.acquire()
            start = clock.monotonic()
            await bucket.acquire()
            return clock.monotonic() - start

        elapsed = asyncio.run(scenario())
        self.assertAlmostEqual(elapsed, 1.0, delta=0.01)

    def test_prompt_resolution_precedence(self) -> None:
        args = argparse.Namespace(
            prompt="positional",
            prompt_file=None,
            stdin=False,
            default_prompt="default",
        )
        resolved = resolve_prompt(args, stdin_text="stdin")
        self.assertEqual(resolved, "positional")

        with tempfile.NamedTemporaryFile("w+", delete=False) as fh:
            fh.write("from-file")
            fh.flush()
            file_path = Path(fh.name)
        args2 = argparse.Namespace(
            prompt=None,
            prompt_file=str(file_path),
            stdin=False,
            default_prompt="default",
        )
        resolved_file = resolve_prompt(args2, stdin_text="stdin")
        self.assertEqual(resolved_file, "from-file")
        file_path.unlink()

        args3 = argparse.Namespace(
            prompt=None,
            prompt_file=None,
            stdin=True,
            default_prompt="default",
        )
        resolved_stdin = resolve_prompt(args3, stdin_text=" stdin value ")
        self.assertEqual(resolved_stdin, "stdin value")

        args4 = argparse.Namespace(
            prompt=None,
            prompt_file=None,
            stdin=False,
            default_prompt="fallback",
        )
        resolved_default = resolve_prompt(args4, stdin_text=None)
        self.assertEqual(resolved_default, "fallback")


def run_self_tests() -> bool:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTests)
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


async def run_prompt(prompt: str, args: argparse.Namespace) -> None:
    keys = load_api_keys()
    if not keys:
        raise RuntimeError(
            "No API keys supplied. Set GROQ_API_KEY or GROQ_API_KEYS environment variables."
        )
    cache_path = Path(args.cache) if args.cache else None
    async with GroqClient(
        keys=keys,
        rpm_per_key=args.rpm,
        max_concurrency=args.concurrency,
        cache_path=cache_path,
        budget=args.budget,
        logger=LOGGER,
        base_url=args.base_url,
    ) as client:
        messages = [{"role": "user", "content": prompt}]
        text = await client.chat(
            messages,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(text)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        success = run_self_tests()
        return 0 if success else 1

    stdin_text = None
    if args.stdin:
        stdin_text = sys.stdin.read()

    try:
        prompt = resolve_prompt(args, stdin_text)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return 1

    if not prompt:
        LOGGER.info("No prompt provided; running self-tests by default.")
        success = run_self_tests()
        return 0 if success else 1

    try:
        asyncio.run(run_prompt(prompt, args))
        return 0
    except Exception as exc:  # pragma: no cover - CLI guardrail
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
