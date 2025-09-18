#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Groq Tool — key‑aware, rate‑limit‑safe CLI (single file, no external deps)

What this script does (high level):
  • Validates one or many Groq API keys (GET /models) and prints a colored report.
  • Offers to save valid keys into a .env file (GROQ_API_KEYS='gsk_x,...').
  • Provides a resilient chat client with:
      - per‑key token‑bucket (RPM) throttling
      - Retry‑After (seconds or HTTP‑date) compliance
      - exponential backoff with jitter for 5xx/network errors
      - per‑key usage tracking (TPM/TPD estimates) in a local JSON state
      - light JSON cache to avoid duplicate requests
      - fair/shuffled key selection to pick the most “free” key
  • Ships with built‑in --help and subcommands: validate, chat, status.

Design choices:
  - Pure standard library HTTP via urllib, wrapped in asyncio with to_thread.
  - No rich/pygments or third‑party logging; we use ANSI color codes if TTY.
  - Inline comments explain non‑obvious parts.

NOTE (compliance):
  Use only keys you are authorized to use within provider policies. This tool
  respects server rate‑limit signals; it is NOT a limit‑evasion utility.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as _urlreq, error as _urlerr

# ============================== Tiny ANSI logger ==============================
# We implement minimal colored printing without external deps. Colors appear
# only when stderr is a TTY; otherwise plain text is shown.
class _Log:
    COLORS = {
        "INFO": "\x1b[37m",     # white/neutral
        "OK": "\x1b[32m",       # green
        "WARN": "\x1b[33m",     # yellow
        "ERR": "\x1b[31m",      # red
        "DBG": "\x1b[36m",      # cyan
    }
    RESET = "\x1b[0m"

    def __init__(self):
        self.use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    def _fmt(self, level: str, msg: str) -> str:
        ts = datetime.now().strftime("%H:%M:%S")
        if self.use_color:
            return f"[{ts}] {self.COLORS.get(level,'')}{level:<4}{self.RESET} {msg}"
        return f"[{ts}] {level:<4} {msg}"

    def info(self, msg: str):
        print(self._fmt("INFO", msg), file=sys.stderr)

    def ok(self, msg: str):
        print(self._fmt("OK", msg), file=sys.stderr)

    def warn(self, msg: str):
        print(self._fmt("WARN", msg), file=sys.stderr)

    def err(self, msg: str):
        print(self._fmt("ERR", msg), file=sys.stderr)

    def dbg(self, msg: str):
        # Debug prints are verbose; keep them behind an env flag if needed.
        if os.environ.get("GROQ_TOOL_DEBUG"):
            print(self._fmt("DBG", msg), file=sys.stderr)

log = _Log()

# ============================= HTTP (std library) =============================
# We provide small helper wrappers around urllib to keep dependencies at zero.
# They return (status_code:int, headers:dict[str,str], text:str).

USER_AGENT = "groq-tool/1.0 (+stdlib)"

def _http_request(method: str, url: str, headers: Dict[str, str], body: Optional[bytes] = None, timeout: float = 60.0) -> Tuple[int, Dict[str, str], str]:
    req = _urlreq.Request(url, data=body, method=method)
    # Merge provided headers with sensible defaults.
    req.add_header("User-Agent", USER_AGENT)
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            # Convert email.message.Message to plain dict[str,str]
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            text = resp.read().decode(hdrs.get("content-type", "utf-8"), errors="replace")
            return status, hdrs, text
    except _urlerr.HTTPError as e:
        # For HTTP errors, we still try to read the body for diagnostics
        hdrs = {k.lower(): v for k, v in e.headers.items()} if e.headers else {}
        text = e.read().decode(hdrs.get("content-type", "utf-8"), errors="replace") if hasattr(e, 'read') else str(e)
        return e.code or 599, hdrs, text
    except _urlerr.URLError as e:
        # Networking problem (DNS, connection, timeout)
        return 599, {}, str(e)

async def _http_get(url: str, headers: Dict[str, str], timeout: float = 60.0) -> Tuple[int, Dict[str, str], str]:
    # Run blocking IO in a thread to avoid freezing the event loop.
    return await asyncio.to_thread(_http_request, "GET", url, headers, None, timeout)

async def _http_post(url: str, headers: Dict[str, str], json_body: Dict[str, Any], timeout: float = 60.0) -> Tuple[int, Dict[str, str], str]:
    body = json.dumps(json_body).encode("utf-8")
    hdrs = {"content-type": "application/json", **{k.lower(): v for k, v in headers.items()}}
    return await asyncio.to_thread(_http_request, "POST", url, hdrs, body, timeout)

# =============================== JSON cache ==================================
class JsonCache:
    """Tiny file‑backed cache keyed by SHA‑256 of the request payload."""
    def __init__(self, path: Path, ttl_sec: int = 6 * 3600):
        self.path = path
        self.ttl = ttl_sec
        self._data: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except Exception:
                log.warn("cache load failed; starting empty")
        # Eagerly drop expired entries to keep file small.
        now = time.time()
        for k, v in list(self._data.items()):
            if now - v.get("ts", 0) > self.ttl:
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
            log.warn(f"cache write failed: {e}")

# =============================== Token bucket ================================
@dataclass
class TokenBucket:
    """Classic token bucket for RPM throttling.

    capacity: max tokens stored (burst allowance)
    refill_per_sec: tokens regenerated per second
    take(amount): waits until enough tokens then consumes them
    """
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
            # Refill proportionally to elapsed time, but cap at capacity.
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
            if self.tokens >= amount:
                self.tokens -= amount
                return
            # Not enough tokens: compute the minimum wait time.
            deficit = amount - self.tokens
            wait = max(0.01, deficit / max(self.refill_per_sec, 1e-6))
            await asyncio.sleep(wait)

# ============================== Per‑key usage ================================
@dataclass
class UsageWindow:
    minute_started: int = 0
    minute_tokens: int = 0
    day_started: int = 0
    day_tokens: int = 0

@dataclass
class KeyState:
    key: str
    limiter: TokenBucket
    cooldown_until: float = 0.0
    usage: UsageWindow = field(default_factory=UsageWindow)

class UsageStore:
    """Persists per‑key usage in a JSON file.

    We store token usage by key for the current minute and current UTC day.
    This let us (roughly) avoid overshooting TPM/TPD estimates.
    """
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except Exception:
                log.warn("usage store load failed; starting empty")

    def load(self, key: str) -> UsageWindow:
        u = self.data.get(key, {})
        return UsageWindow(
            minute_started=u.get("minute_started", 0),
            minute_tokens=u.get("minute_tokens", 0),
            day_started=u.get("day_started", 0),
            day_tokens=u.get("day_tokens", 0),
        )

    def save(self, key: str, uw: UsageWindow) -> None:
        self.data[key] = dataclasses.asdict(uw)
        try:
            self.path.write_text(json.dumps(self.data, indent=2))
        except Exception as e:
            log.warn(f"usage store write failed: {e}")

# ============================== Model profiles ===============================
# Default conservative limits per model (can be overridden via CLI JSON file).
MODEL_PROFILES: Dict[str, Dict[str, Optional[int]]] = {
    # name                         rpm   tpm     tpd
    "llama-3.3-70b-versatile":            {"rpm": 30,  "tpm": 6000,  "tpd": None},
    "llama3-70b-versatile":      {"rpm": 15,  "tpm": 18000, "tpd": None},
}

# ================================ Key manager ================================
class KeyManager:
    """Round‑robin key allocator that also accounts for TPM/TPD headroom.

    Selection strategy:
      1) Filter out keys in cooldown.
      2) For remaining keys, compute remaining headroom for the current minute
         and day from UsageStore and model profile.
      3) Rank by highest headroom, then shuffle top ties to avoid starvation.
      4) Respect per‑key RPM via token buckets before returning the key.
    """
    def __init__(self, keys: List[str], rpm_per_key: int, usage_store: UsageStore, model_profile: Dict[str, Optional[int]]):
        if not keys:
            raise ValueError("no API keys provided")
        self.states: List[KeyState] = []
        for k in keys:
            st = KeyState(key=k, limiter=TokenBucket(capacity=rpm_per_key, refill_per_sec=rpm_per_key/60.0))
            st.usage = usage_store.load(k)
            self.states.append(st)
        self.usage_store = usage_store
        self.profile = model_profile
        self._lock = asyncio.Lock()

    def _refresh_windows(self, st: KeyState):
        now = time.time()
        # Reset minute window if we crossed a minute boundary.
        minute = int(now // 60)
        if st.usage.minute_started != minute:
            st.usage.minute_started = minute
            st.usage.minute_tokens = 0
        # Reset day window if we crossed UTC day boundary.
        day = int(now // 86400)
        if st.usage.day_started != day:
            st.usage.day_started = day
            st.usage.day_tokens = 0

    def _headroom(self, st: KeyState) -> Tuple[int, int]:
        """Return (minute_headroom, day_headroom or large number if None)."""
        self._refresh_windows(st)
        tpm = self.profile.get("tpm")
        tpd = self.profile.get("tpd")
        # If tpm/tpd unknown, treat as very large to avoid being a tie‑breaker.
        mh = (tpm - st.usage.minute_tokens) if isinstance(tpm, int) else 10**9
        dh = (tpd - st.usage.day_tokens) if isinstance(tpd, int) else 10**12
        return max(0, mh), max(0, dh)

    async def acquire(self) -> KeyState:
        while True:
            async with self._lock:
                # Consider only keys not in cooldown.
                ready = [st for st in self.states if time.time() >= st.cooldown_until]
                if not ready:
                    # Everyone cooling: sleep a bit and retry.
                    await asyncio.sleep(0.05)
                    continue
                # Rank by headroom; shuffle ties for fairness.
                scored = []
                for st in ready:
                    mh, dh = self._headroom(st)
                    # negative of usage → more remaining headroom sorts first
                    scored.append((-(mh + dh/1000.0), random.random(), st))
                scored.sort(key=lambda x: (x[0], x[1]))
                st = scored[0][2]
                # Enforce per‑key RPM token bucket before handing out.
                await st.limiter.take(1)
                return st
            # Should rarely occur; small sleep to avoid hot loop.
            await asyncio.sleep(0.01)

    def cooldown(self, st: KeyState, seconds: float):
        st.cooldown_until = max(st.cooldown_until, time.time() + seconds)
        log.warn(f"🧊 Key {st.key[:8]}… cooling down for {seconds:.1f}s (Retry-After)")

    def note_tokens(self, st: KeyState, used: int):
        # Update per‑key minute/day counters and persist them.
        self._refresh_windows(st)
        st.usage.minute_tokens += int(max(0, used))
        st.usage.day_tokens += int(max(0, used))
        self.usage_store.save(st.key, st.usage)

# ================================ Client core =================================
class GroqClient:
    def __init__(
        self,
        keys: List[str],
        model: str,
        rpm_per_key: int,
        usage_store: UsageStore,
        cache_path: Path,
        base_url: str = "https://api.groq.com/openai/v1/chat/completions",
        timeout: float = 60.0,
        max_concurrency: int = 8,
    ):
        profile = MODEL_PROFILES.get(model, {"rpm": rpm_per_key, "tpm": None, "tpd": None})
        # Use model’s own RPM if known; otherwise the provided rpm_per_key.
        eff_rpm = int(profile.get("rpm") or rpm_per_key)
        self.km = KeyManager(keys, eff_rpm, usage_store, profile)
        self.cache = JsonCache(cache_path)
        self.base_url = base_url
        self.timeout = timeout
        self.sema = asyncio.Semaphore(max_concurrency)

    async def chat(self, messages: List[Dict[str, Any]], model: str, temperature: float = 0.7) -> str:
        # Cache by exact payload to deduplicate identical requests.
        payload = {"model": model, "messages": messages, "temperature": temperature}
        cached = self.cache.get(payload)
        if cached is not None:
            log.info("✨ cache hit")
            return cached

        # Estimate tokens very roughly to decide early budget usage when API
        # doesn’t return usage (some implementations). We still prefer API usage.
        est_tokens = max(1, sum(len(m.get("content", "")) for m in messages) // 4)

        attempt = 0
        max_attempts = 6
        while attempt < max_attempts:
            attempt += 1
            st = await self.km.acquire()
            async with self.sema:
                try:
                    headers = {"authorization": f"Bearer {st.key}"}
                    log.info(f"→ request (attempt {attempt})")
                    status, hdrs, text = await _http_post(self.base_url, headers, payload, timeout=self.timeout)
                    if status == 200:
                        data = json.loads(text)
                        out = data["choices"][0]["message"]["content"]
                        used = int(data.get("usage", {}).get("total_tokens", est_tokens))
                        self.km.note_tokens(st, used)
                        self.cache.set(payload, out)
                        return out
                    if status == 429:
                        # Honor Retry‑After (seconds or HTTP‑date).
                        ra = _parse_retry_after(hdrs.get("retry-after"))
                        self.km.cooldown(st, ra)
                        await asyncio.sleep(ra)
                        continue
                    if 500 <= status < 600 or status in (408, 425, 599):
                        # Transient or networkish
                        delay = _exp_backoff(attempt)
                        log.warn(f"server {status}; backing off {delay:.2f}s")
                        await asyncio.sleep(delay)
                        continue
                    # Non‑retryable
                    raise RuntimeError(f"HTTP {status}: {text[:300]}")
                except Exception as e:
                    delay = _exp_backoff(attempt)
                    log.warn(f"network/parse error {e!r}; retry in {delay:.2f}s")
                    await asyncio.sleep(delay)
        raise RuntimeError("Max retries exceeded")

# ================================ Utilities ==================================
def _parse_retry_after(val: Optional[str]) -> float:
    """Parse Retry‑After header (delta‑seconds or HTTP‑date)."""
    if not val:
        return 5.0
    try:
        return float(val)
    except ValueError:
        try:
            dt = parsedate_to_datetime(val)
            now = datetime.now(timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - now).total_seconds())
        except Exception:
            return 5.0

def _exp_backoff(attempt: int, base: float = 0.5, cap: float = 20.0) -> float:
    """Exponential backoff with decorrelated jitter."""
    delay = min(cap, base * (2 ** (attempt - 1)))
    return delay * (0.5 + random.random())

def _resolve_prompt_sources(positional: Optional[str], stdin_text: Optional[str], file_text: Optional[str], default_text: Optional[str]) -> Optional[str]:
    """Precedence: positional > file > stdin > default."""
    for cand in (positional, file_text, stdin_text, default_text):
        if cand is not None and cand.strip():
            return cand.strip()
    return None

# ================================ Validation =================================
async def validate_keys(keys: List[str], models_url: str = "https://api.groq.com/openai/v1/models") -> Tuple[List[str], List[str]]:
    """Check keys with GET /models.

    Returns (valid_keys, invalid_keys). We never print keys fully; only show
    short prefixes for safety. Emojis are used for quick visual feedback.
    """
    valids: List[str] = []
    invalids: List[str] = []
    # We do requests sequentially to avoid tripping provider defenses during validation.
    for k in keys:
        headers = {"authorization": f"Bearer {k}"}
        status, _, _ = await _http_get(models_url, headers, timeout=20.0)
        if status == 200:
            log.ok(f"✅ valid  — {k[:12]}…")
            valids.append(k)
        else:
            log.err(f"❌ invalid — {k[:12]}… (HTTP {status})")
            invalids.append(k)
    # Deduplicate while preserving order
    def _dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                out.append(s); seen.add(s)
        return out
    return _dedupe(valids), _dedupe(invalids)

# ================================== CLI ======================================
def _parse_keys_from_sources(args: argparse.Namespace) -> List[str]:
    # priority: --keys > --keys-file > env vars
    if args.keys:
        buf = args.keys
    elif args.keys_file:
        buf = Path(args.keys_file).read_text(encoding="utf-8")
    else:
        buf = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY") or ""
    # accept comma and/or newline separated values
    raw = [x.strip() for x in buf.replace("\n", ",").split(",")]
    return [x for x in raw if x]

def _print_usage_summary(valids: List[str]):
    if not valids:
        log.err("No valid keys found.")
        return
    masked = ", ".join([v[:8] + "…" for v in valids])
    log.ok(f"✅ unique & valid keys ({len(valids)}): {masked}")

def _maybe_write_env(valids: List[str], path: Optional[str]):
    if not valids:
        return
    env_line = f"GROQ_API_KEYS='{" ,".join(valids)}'\n"
    if path:
        Path(path).write_text(env_line, encoding="utf-8")
        log.ok(f"📦 wrote {path} with GROQ_API_KEYS ({len(valids)} keys)")
        return
    # Interactive prompt only if TTY
    if sys.stdin.isatty():
        try:
            resp = input("Save valid keys to .env? [y/N] ")
        except EOFError:
            resp = "n"
        if resp.lower().startswith("y"):
            Path(".env").write_text(env_line, encoding="utf-8")
            log.ok("📦 wrote .env with GROQ_API_KEYS; run: export $(cat .env | xargs)")

async def cmd_validate(args: argparse.Namespace) -> int:
    keys = _parse_keys_from_sources(args)
    if not keys:
        log.err("Provide keys via --keys/--keys-file or GROQ_API_KEYS/GROQ_API_KEY env.")
        return 2
    log.info(f"🔍 checking {len(keys)} key(s)…")
    valids, invalids = await validate_keys(keys)
    _print_usage_summary(valids)
    _maybe_write_env(valids, args.write_env)
    return 0 if valids else 1

async def cmd_chat(args: argparse.Namespace) -> int:
    # Resolve prompt text from multiple sources (positional, file, stdin, default)
    stdin_text = None
    if args.stdin or (not sys.stdin.isatty()):
        try:
            data = sys.stdin.read()
            stdin_text = (data or "").strip() or None
        except Exception:
            stdin_text = None
    file_text = None
    if args.prompt_file:
        try:
            file_text = Path(args.prompt_file).read_text(encoding="utf-8").strip() or None
        except Exception as e:
            log.err(f"failed to read --prompt-file: {e}")
    prompt_text = _resolve_prompt_sources(args.prompt, stdin_text, file_text, args.default_prompt)
    if not prompt_text:
        log.info("No input provided; try: groq_tool.py chat --default-prompt 'Hello'")
        return 2

    # Gather keys
    keys = _parse_keys_from_sources(args)
    if not keys:
        log.err("Set GROQ_API_KEYS/GROQ_API_KEY or use --keys/--keys-file.")
        return 2

    # Initialize usage store + client
    usage_store = UsageStore(Path(args.state_file))
    client = GroqClient(
        keys=keys,
        model=args.model,
        rpm_per_key=args.rpm,
        usage_store=usage_store,
        cache_path=Path(args.cache),
        timeout=args.timeout,
        max_concurrency=args.concurrency,
    )
    try:
        text = await client.chat([{"role": "user", "content": prompt_text}], model=args.model, temperature=args.temperature)
        print("\n=== RESPONSE ===\n" + text)
        return 0
    finally:
        # nothing to close (urllib), but we could flush files if needed
        pass

async def cmd_status(args: argparse.Namespace) -> int:
    # Print a quick view of per‑key usage windows and cooldowns.
    keys = _parse_keys_from_sources(args)
    if not keys:
        log.err("No keys available (use --keys/--keys-file or env vars)")
        return 2
    usage_store = UsageStore(Path(args.state_file))
    profile = MODEL_PROFILES.get(args.model, {"rpm": args.rpm, "tpm": None, "tpd": None})
    print("Model:", args.model)
    print("RPM:", profile.get("rpm"), "TPM:", profile.get("tpm"), "TPD:", profile.get("tpd"))
    print("Keys:")
    for k in keys:
        u = usage_store.load(k)
        minute = int(time.time() // 60)
        day = int(time.time() // 86400)
        # Remaining headroom based on profile (approximate)
        tpm = profile.get("tpm")
        tpd = profile.get("tpd")
        mh = (tpm - u.minute_tokens) if isinstance(tpm, int) else None
        dh = (tpd - u.day_tokens) if isinstance(tpd, int) else None
        print(f"  • {k[:12]}…  min[{u.minute_tokens} used @{u.minute_started}/{minute}]  day[{u.day_tokens} used @{u.day_started}/{day}]  headroom[min={mh}, day={dh}]")
    return 0

# ================================== Main =====================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="groq_tool.py",
        description="Key‑aware, rate‑limit‑safe Groq CLI (validate keys, chat, status)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # validate
    v = sub.add_parser("validate", help="validate keys via GET /models and optionally save to .env")
    v.add_argument("--keys", help="keys separated by comma or newlines")
    v.add_argument("--keys-file", help="file containing keys (comma/newline separated)")
    v.add_argument("--write-env", metavar="PATH", help="write GROQ_API_KEYS to a file (e.g., .env)")
    v.set_defaults(func=cmd_validate)

    # chat
    c = sub.add_parser("chat", help="send one prompt using resilient rolling keys")
    c.add_argument("prompt", nargs="?", help="prompt text")
    c.add_argument("--stdin", action="store_true", help="read prompt from STDIN (auto when piped)")
    c.add_argument("--prompt-file", help="read prompt from a file")
    c.add_argument("--default-prompt", help="fallback prompt if none provided")
    c.add_argument("--keys", help="keys separated by comma or newlines")
    c.add_argument("--keys-file", help="file containing keys (comma/newline separated)")
    c.add_argument("--model", default="llama-3.3-70b-versatile", help="model name")
    c.add_argument("--rpm", type=int, default=30, help="per‑key RPM budget")
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--concurrency", type=int, default=8)
    c.add_argument("--timeout", type=float, default=60.0)
    c.add_argument("--cache", default=".groq_cache.json")
    c.add_argument("--state-file", default=".groq_usage.json", help="per‑key usage persistence")
    c.set_defaults(func=cmd_chat)

    # status
    s = sub.add_parser("status", help="show per‑key usage windows and headroom")
    s.add_argument("--keys", help="keys separated by comma or newlines")
    s.add_argument("--keys-file", help="file containing keys (comma/newline separated)")
    s.add_argument("--model", default="llama-3.3-70b-versatile")
    s.add_argument("--rpm", type=int, default=30)
    s.add_argument("--state-file", default=".groq_usage.json")
    s.set_defaults(func=cmd_status)

    # Epilogue examples (acts like a built‑in help/cheat‑sheet)
    p.epilog = (
        "Examples:\n"
        "  # 1) Validate keys from a file and save to .env\n"
        "  groq_tool.py validate --keys-file keys.txt --write-env .env\n\n"
        "  # 2) One‑shot prompt\n"
        "  GROQ_API_KEYS='gsk_x,gsk_y' groq_tool.py chat 'What is red teaming?'\n\n"
        "  # 3) From STDIN\n"
        "  echo 'Summarize incident triage' | groq_tool.py chat --stdin\n\n"
        "  # 4) Inspect current usage windows\n"
        "  groq_tool.py status --model llama-3.3-70b-versatile\n"
    )
    return p

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(args.func(args))
    except KeyboardInterrupt:
        log.warn("Interrupted by user")
        return 130
    except Exception as e:
        log.err(f"fatal: {e}")
        return 1

# ----------------------------- Helper functions ------------------------------
# Retry‑After parser/backoff are intentionally duplicated near top for clarity
# but kept simple; unit tests could be added similarly to earlier modules.

if __name__ == "__main__":
    sys.exit(main())
