# Repository Blurb (Short Description) ✨🔧📘

**EN:** Standards‑conformant Groq client implementing per‑key token‑bucket rate limiting, first‑class `Retry‑After` semantics, exponential backoff with decorrelated jitter, persistent JSON‑backed caching, and bounded concurrency — with no ancillary logging dependencies. ✨🔧📘

**AZ:** Groq üçün siyasətə uyğun müştəri: açar‑başına token bucket, `Retry‑After` hörməti, eksponensial backoff, JSON keş və paralel icra nəzarəti — əlavə logging asılılığı yoxdur. ✨🔧📘

---

# README.md — Compliant Groq Client (Rate‑limit‑safe Kit) ✨🔧📘

A resilient, **terms‑compliant** Groq API client that gracefully handles rate limits and network hiccups. It honors `HTTP 429` and `Retry‑After`, applies **per‑key token‑bucket** throttling, uses **exponential backoff with jitter**, and avoids duplicate calls via a lightweight **JSON cache**. Designed to be drop‑in and CI‑friendly with **no external logging dependencies**. ✨🔧📘

> Code lives in `groq_client_compliant.py` and includes inline comments plus self‑tests. ✨🔧📘

## Features ✨🔧📘

* **Policy‑compliant throttling:** Honors `Retry‑After` (seconds or HTTP‑date).
* **Per‑key token buckets:** Convert RPM → per‑second refill; fair round‑robin across keys.
* **Exponential backoff + jitter:** Gentle retries for 5xx/network errors.
* **Local JSON cache:** Deduplicates identical requests (default TTL: 6h).
* **Concurrency control:** Global semaphore to prevent overload.
* **Daily token budget (optional):** Soft cap on estimated token usage.
* **Zero extra logging deps:** ANSI‑colored logs without `rich/pygments`.

## Requirements ✨🔧📘

* Python **3.9+**
* Package: `httpx` (install below)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install httpx>=0.27
```

## Quick Start ✨🔧📘

Export a key and run one of the input modes. ✨🔧📘

```bash
# Single key
export GROQ_API_KEY='gsk_...'
# Multiple keys (same org/account, policy‑compliant)
export GROQ_API_KEYS='gsk_key1,gsk_key2'
```

### Input options (choose one) ✨🔧📘

* **Positional text:**

  ```bash
  python groq_client_compliant.py "What is red teaming?"
  ```
* **From file:**

  ```bash
  echo "Explain DDoS indicators." > prompt.txt
  python groq_client_compliant.py --prompt-file prompt.txt
  ```
* **From STDIN (pipe):**

  ```bash
  echo "Nginx 4xx spike causes?" | python groq_client_compliant.py --stdin
  ```
* **Fallback text (no other input):**

  ```bash
  python groq_client_compliant.py --default-prompt "Summarize incident triage steps."
  ```
* **Self‑tests only:**

  ```bash
  python groq_client_compliant.py --self-test
  ```

### Common flags ✨🔧📘

```bash
--model llama3-8b-8192         # API model name
--rpm 30                        # Per‑key RPM for token bucket
--concurrency 8                 # Max parallel in‑flight calls
--budget 150000                 # Optional daily token budget (soft)
--cache .groq_cache.json        # Cache file path
```

## Programmatic Usage ✨🔧📘

```python
import asyncio
from pathlib import Path
from groq_client_compliant import GroqClient

async def main():
    keys = ["gsk_...1", "gsk_...2"]  # authorized keys
    client = GroqClient(keys=keys, rpm_per_key=30, max_concurrency=8, cache_path=Path('.groq_cache.json'))
    try:
        messages = [{"role": "user", "content": "Give a one‑liner on DDoS."}]
        text = await client.chat(messages, model="llama3-8b-8192")
        print(text)
    finally:
        await client.aclose()

asyncio.run(main())
```

## How It Works ✨🔧📘

* **KeyManager:** Round‑robin allocator across keys; applies per‑key token bucket and temporary **cooldowns** when servers return `429` with `Retry‑After`.
* **TokenBucket:** Standard leaky/token bucket where tokens refill by time and are required per request.
* **Backoff:** For 5xx/network errors, wait using exponential backoff with jitter to avoid synchronized retries.
* **Cache:** SHA‑256 of request payload → value, written to a small JSON file; TTL prunes stale entries on startup.
* **Concurrency:** Global semaphore ensures at most *N* concurrent requests.

## Logging ✨🔧📘

* Uses standard `logging` with a minimal ANSI color formatter.
* Colors appear when attached to a TTY; otherwise log lines are plain text.

## CLI Behavior ✨🔧📘

* If **no input** is provided, the tool **runs self‑tests** by default and exits successfully.
* Provide input via positional arg, `--prompt-file`, `--stdin`, or `--default-prompt`.

## Environment Variables ✨🔧📘

* `GROQ_API_KEY` — single key (alternative to `GROQ_API_KEYS`).
* `GROQ_API_KEYS` — comma‑separated keys (policy‑compliant use only).

> Keys must be authorized and used within provider policies. This client is built to **respect** limits, not circumvent them. ✨🔧📘

## Testing ✨🔧📘

Run built‑in offline tests: ✨🔧📘

```bash
python groq_client_compliant.py --self-test
```

Tests cover: ✨🔧📘

* `Retry‑After` parsing (seconds & HTTP‑date)
* Exponential backoff jitter
* Cache read/write
* Token bucket behavior
* Prompt resolution precedence

## Security & Compliance ✨🔧📘

* **Do not** use unapproved keys or identities.
* This client **does not** proxy/rotate to evade limits; it obeys provider signals.
* Rotate credentials if you ever suspect leakage; **never commit `.env`** files.

## Limitations ✨🔧📘

* Token counting is an estimate unless the API returns exact usage.
* Cache TTL is static (6h) and configured in code.
* Only the chat completions endpoint is implemented by default.

## Roadmap (optional) ✨🔧📘

* Configurable cache TTL via CLI
* Pluggable storage (SQLite/Redis) for cache
* Streaming responses and partial retry
* Metrics hooks (Prometheus/OpenTelemetry)

## License ✨🔧📘

Choose a license that fits your project (e.g., MIT). Add a `LICENSE` file at repo root. ✨🔧📘

## Contributing ✨🔧📘

1. Fork → feature branch → PR.
2. Keep changes small and covered by tests (`--self-test` can be extended).
3. Follow conventional commit messages (e.g., `feat(client): add streaming`).
