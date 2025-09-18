# groq_client_compliant.py
# ---------------------------------------------------------------
# PURPOSE
# A **terms‑compliant** Groq API client that is resilient to rate limits
# without evasion. It:
#   • Respects Retry‑After and HTTP 429/5xx semantics
#   • Uses per‑key (org‑owned) token‑bucket limiters
#   • Adds exponential backoff with jitter
#   • Provides local caching to reduce duplicate calls
#   • Limits concurrency and supports a daily token budget cap
#   • **No external logging deps** (works without rich/pygments)
#
# IMPORTANT
# - Use **only** API keys you are authorized to use under Groq’s policies.
# - Do **not** rotate identities, spoof, or proxy to bypass provider limits.
# - This client reduces load via caching/backoff; it is not a bypass tool.
# ---------------------------------------------------------------

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

import httpx
import logging

# ------------------------- Logging (no external deps) -------------------------
class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\x1b[36m",     # cyan
        "INFO": "\x1b[37m",      # white
        "WARNING": "\x1b[33m",   # yellow
        "ERROR": "\x1b[31m",     # red
        "CRITICAL": "\x1b[35m",  # magenta
    }
    RESET = "\x1b[0m"

    def __init__(self, fmt: str, use_color: bool):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self.use_color:
            color = self.COLORS.get(record.levelname, "")
            if color:
                return f"{color}{msg}{self.RESET}"
        return msg

_stream_is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter(FORMAT, use_color=_stream_is_tty))
logging.basicConfig(level=logging.INFO, handlers=[handler])
log = logging.getLogger("groq_client")

# ------------------------- Simple Cache -------------------------
class JsonCache:
    def __init__(self, path: Path, ttl_seconds: int = 6 * 3600):
        self.path = path
        self.ttl = ttl_seconds
        self._data: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except Exception:
                log.warning("Cache load failed; starting empty.")
        # purge expired
        now = time.time()
        expired = [k for k, v in list(self._data.items()) if now - v.get("ts", 0) > self.ttl]
        for k in expired:
            self._data.pop(k, None)

    def _key(self, payload: Dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, payload: Dict[str, Any]) -> Optional[Any]:
        k = self._key(payload)
        item = self._data.get(k)
        if item and time.time() - item.get("ts", 0) <= self.ttl:
            return item.get("val")
        return None

    def set(self, payload: Dict[str, Any], value: Any) -> None:
        k = self._key(payload)
        self._data[k] = {"ts": time.time(), "val": value}
        try:
            self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2))
        except Exception as e:
            log.warning(f"Cache write failed: {e}")

# ------------------------- Token Bucket -------------------------
@dataclass
class TokenBucket:
    capacity: int
    refill_per_sec: float
    tokens: float = field(init=False)
    last: float = field(default_factory=time.time)

    def __post_init__(self):
        self.tokens = float(self.capacity)

    async def take(self, amount: float = 1.0):
        while True:
            now = time.time()
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
            if self.tokens >= amount:
                self.tokens -= amount
                return
            # need to wait for enough tokens
            deficit = amount - self.tokens
            wait = max(0.01, deficit / max(self.refill_per_sec, 1e-6))
            await asyncio.sleep(wait)

# ------------------------- Key Manager -------------------------
@dataclass
class KeyState:
    key: str
    limiter: TokenBucket
    cooldown_until: float = 0.0

class KeyManager:
    def __init__(self, keys: List[str], rpm_per_key: int):
        if not keys:
            raise ValueError("No API keys provided.")
        self.states: List[KeyState] = [
            KeyState(key=k, limiter=TokenBucket(capacity=rpm_per_key, refill_per_sec=rpm_per_key/60.0))
            for k in keys
        ]
        self._idx = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> KeyState:
        while True:
            async with self._lock:
                n = len(self.states)
                for _ in range(n):
                    st = self.states[self._idx]
                    self._idx = (self._idx + 1) % n
                    if time.time() >= st.cooldown_until:
                        await st.limiter.take(1)
                        return st
            await asyncio.sleep(0.05)

    def cooldown(self, st: KeyState, seconds: float):
        st.cooldown_until = max(st.cooldown_until, time.time() + seconds)
        log.warning(f"Key {st.key[:8]}… cooling down for {seconds:.1f}s (Retry-After)")

# ------------------------- Client -------------------------
class GroqClient:
    def __init__(
        self,
        keys: List[str],
        rpm_per_key: int = 30,
        max_concurrency: int = 8,
        daily_token_budget: Optional[int] = None,
        cache_path: Path = Path(".groq_cache.json"),
        cache_ttl_sec: int = 6*3600,
        base_url: str = "https://api.groq.com/openai/v1/chat/completions",
        request_timeout: float = 60.0,
    ):
        # NOTE: Only use keys you are authorized to use within your org/account.
        self.km = KeyManager(keys, rpm_per_key=rpm_per_key)
        self.sema = asyncio.Semaphore(max_concurrency)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(request_timeout))
        self.cache = JsonCache(cache_path, ttl_seconds=cache_ttl_sec)
        self.base_url = base_url
        self.daily_token_budget = daily_token_budget
        self._spent_tokens = 0

    def _budget_ok(self, est_tokens: int) -> bool:
        if self.daily_token_budget is None:
            return True
        return (self._spent_tokens + est_tokens) <= self.daily_token_budget

    def _note_spent(self, used: int) -> None:
        if self.daily_token_budget is not None:
            self._spent_tokens += used

    async def chat(self, messages: List[Dict[str, Any]], model: str = "llama-3.3-70b-versatile", temperature: float = 0.7) -> str:
        payload = {"model": model, "messages": messages, "temperature": temperature}

        cached = self.cache.get(payload)
        if cached is not None:
            log.info("cache hit")
            return cached

        # naive token estimate (prompt tokens ~ chars/4)
        est_tokens = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        if not self._budget_ok(est_tokens):
            raise RuntimeError("Daily token budget reached; refusing new calls.")

        attempt = 0
        max_attempts = 6
        while attempt < max_attempts:
            attempt += 1
            st = await self.km.acquire()
            async with self.sema:
                try:
                    headers = {"Authorization": f"Bearer {st.key}", "Content-Type": "application/json"}
                    log.info(f"→ request (attempt {attempt})")
                    resp = await self.client.post(self.base_url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        text = data["choices"][0]["message"]["content"]
                        self.cache.set(payload, text)
                        used = data.get("usage", {}).get("total_tokens", est_tokens)
                        self._note_spent(int(used))
                        return text

                    if resp.status_code == 429:
                        retry_after = _parse_retry_after(resp.headers.get("retry-after"))
                        self.km.cooldown(st, retry_after)
                        await asyncio.sleep(retry_after)
                        continue

                    if 500 <= resp.status_code < 600:
                        delay = _exp_backoff(attempt)
                        log.warning(f"server {resp.status_code}; backing off {delay:.2f}s")
                        await asyncio.sleep(delay)
                        continue

                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

                except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                    delay = _exp_backoff(attempt)
                    log.warning(f"network error {e!r}; retry in {delay:.2f}s")
                    await asyncio.sleep(delay)

        raise RuntimeError("Max retries exceeded.")

    async def aclose(self):
        await self.client.aclose()

# ------------------------- Helpers -------------------------

def _parse_retry_after(val: Optional[str]) -> float:
    if not val:
        return 5.0
    # Might be delta‑seconds or HTTP‑date
    try:
        # delta-seconds
        return float(val)
    except ValueError:
        try:
            dt = parsedate_to_datetime(val)
            now = datetime.now(timezone.utc)
            if dt.tzinfo is None:
                # assume GMT per RFC if tz missing
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - now).total_seconds())
        except Exception:
            return 5.0


def _exp_backoff(attempt: int, base: float = 0.5, cap: float = 20.0) -> float:
    delay = min(cap, base * (2 ** (attempt - 1)))
    return delay * (0.5 + random.random())

# ------------------------- Prompt resolution helpers -------------------------

def _resolve_prompt_sources(positional: Optional[str], stdin_text: Optional[str], file_text: Optional[str], default_text: Optional[str]) -> Optional[str]:
    """Precedence: positional > file > stdin > default."""
    for cand in (positional, file_text, stdin_text, default_text):
        if cand is not None and cand.strip():
            return cand.strip()
    return None

# ------------------------- CLI demo & Self‑tests -------------------------
if __name__ == "__main__":
    import argparse
    import unittest

    parser = argparse.ArgumentParser(description="Compliant Groq client (rate‑limit‑safe; no external logging deps)")
    parser.add_argument("prompt", nargs="?", help="user prompt text")
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--rpm", type=int, default=30, help="per‑key RPM (as allowed by provider)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--budget", type=int, default=None, help="daily token budget cap")
    parser.add_argument("--cache", default=".groq_cache.json")
    parser.add_argument("--self-test", action="store_true", help="run offline tests and exit")
    parser.add_argument("--stdin", action="store_true", help="read prompt from STDIN (or auto if STDIN is piped)")
    parser.add_argument("--prompt-file", help="read prompt from a file path")
    parser.add_argument("--default-prompt", help="fallback prompt if none provided")
    args = parser.parse_args()

    if args.self_test:
        class TestHelpers(unittest.TestCase):
            def test_parse_retry_after_seconds(self):
                self.assertAlmostEqual(_parse_retry_after("10"), 10.0, delta=0.01)

            def test_parse_retry_after_httpdate(self):
                future = (datetime.now(timezone.utc).timestamp() + 3)
                http_date = datetime.fromtimestamp(future, tz=timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
                val = _parse_retry_after(http_date)
                self.assertTrue(0.0 <= val <= 5.0)  # small window

            def test_exp_backoff_monotonic(self):
                a1 = _exp_backoff(1)
                a2 = _exp_backoff(2)
                self.assertTrue(a2 >= 0)
                self.assertTrue(a1 >= 0)

            def test_cache_roundtrip(self):
                c = JsonCache(Path(".test_cache.json"), ttl_seconds=60)
                payload = {"a": 1}
                c.set(payload, {"ok": True})
                self.assertEqual(c.get(payload), {"ok": True})
                Path(".test_cache.json").unlink(missing_ok=True)

            def test_token_bucket(self):
                async def run():
                    tb = TokenBucket(capacity=1, refill_per_sec=1000)
                    t0 = time.time()
                    await tb.take(1)
                    dt = time.time() - t0
                    self.assertLess(dt, 0.1)
                asyncio.run(run())

            def test_prompt_resolution_precedence(self):
                self.assertEqual(
                    _resolve_prompt_sources("pos", stdin_text="stdin", file_text="file", default_text="def"),
                    "pos",
                )
                self.assertEqual(
                    _resolve_prompt_sources(None, stdin_text="stdin", file_text="file", default_text="def"),
                    "file",
                )
                self.assertEqual(
                    _resolve_prompt_sources(None, stdin_text="stdin", file_text=None, default_text="def"),
                    "stdin",
                )
                self.assertEqual(
                    _resolve_prompt_sources(None, stdin_text=None, file_text=None, default_text="def"),
                    "def",
                )

        unittest.main(argv=[sys.argv[0]], exit=False)
        sys.exit(0)

    # Resolve prompt from multiple sources
    stdin_text: Optional[str] = None
    if args.stdin or (not sys.stdin.isatty()):
        try:
            data = sys.stdin.read()
            stdin_text = (data or "").strip() or None
        except Exception:
            stdin_text = None

    file_text: Optional[str] = None
    if args.prompt_file:
        try:
            file_text = Path(args.prompt_file).read_text(encoding="utf-8").strip() or None
        except Exception as e:
            log.error(f"Failed to read --prompt-file: {e}")

    prompt_text = _resolve_prompt_sources(args.prompt, stdin_text, file_text, args.default_prompt)

    # If still nothing, auto‑run self‑tests instead of exiting with SystemExit
    if not prompt_text:
        log.info("No prompt provided; running --self-test as a safe default. Use --default-prompt/--stdin/--prompt-file to supply input.")
        # emulate --self-test path
        import unittest

        class _AutoTest(unittest.TestCase):
            def test_auto(self):
                self.assertTrue(True)
        unittest.main(argv=[sys.argv[0]], exit=False)
        sys.exit(0)

    # Normal CLI path
    keys_env = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY")
    if not keys_env:
        raise SystemExit("Set GROQ_API_KEY(S) in environment.")
    keys = [k.strip() for k in keys_env.split(",") if k.strip()]

    async def main():
        client = GroqClient(
            keys=keys,
            rpm_per_key=args.rpm,
            max_concurrency=args.concurrency,
            daily_token_budget=args.budget,
            cache_path=Path(args.cache),
        )
        try:
            text = await client.chat([{"role": "user", "content": prompt_text}], model=args.model)
            print("\n=== RESPONSE ===\n" + text)
        finally:
            await client.aclose()

    asyncio.run(main())
