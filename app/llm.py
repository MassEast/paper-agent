import contextvars
import json
import logging
import threading
import os
import re
import time
import random
import httpx

_LOG_LLM = os.environ.get("LOG_LLM", "").strip() in ("1", "true", "yes")
_log = logging.getLogger(__name__)

_LLM_MODELS_RAW = os.environ.get("LLM_MODELS")
if not _LLM_MODELS_RAW:
    raise EnvironmentError(
        "LLM_MODELS environment variable is required (comma-separated model names, "
        "tried in order as a fallback cascade — e.g. LLM_MODELS=gpt-4o,gpt-4o-mini)"
    )
MODELS = [m.strip() for m in _LLM_MODELS_RAW.split(",") if m.strip()]

API_BASE = os.environ.get("LLM_API_BASE")
if not API_BASE:
    raise EnvironmentError(
        "LLM_API_BASE environment variable is required (base URL of an "
        "OpenAI-compatible chat-completions endpoint, e.g. https://api.openai.com/v1)"
    )
API_KEY = os.environ.get("LLM_API_KEY")
if not API_KEY:
    raise EnvironmentError("LLM_API_KEY environment variable is required")


MAX_TOKENS_CEILING = 16384

# Requests in flight at once from this process (LLM_MAX_CONCURRENCY, unset = no limit). An endpoint with a
# concurrency quota turns away the requests above it; the web app and the nightly job share one key, so
# their two limits together must stay within the quota.
_MAX_CONCURRENCY = int(os.environ.get("LLM_MAX_CONCURRENCY") or 0)
_slots = threading.BoundedSemaphore(_MAX_CONCURRENCY) if _MAX_CONCURRENCY > 0 else None


class _slot:
    """Holds one of the LLM_MAX_CONCURRENCY request slots (nothing to hold without a limit)."""

    def __enter__(self):
        if _slots:
            _slots.acquire()

    def __exit__(self, *_):
        if _slots:
            _slots.release()


class LLMUnavailableError(Exception):
    pass


def _estimate_token_count(text: str) -> int:
    return max(1, int(len(text) / 4))


def _estimate_usage(messages: list[dict], response_text: str) -> dict:
    prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
    prompt_tokens = _estimate_token_count(prompt_text)
    completion_tokens = _estimate_token_count(response_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


class UsageMeter:
    """Adds up the tokens of every LLM call made while it is the current meter (see metered)."""

    def __init__(self):
        self.tokens = 0
        self.calls = 0
        self._lock = threading.Lock()

    def add(self, usage: dict) -> None:
        with self._lock:
            self.tokens += int(usage.get("total_tokens")
                               or (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0))
            self.calls += 1


_meter: contextvars.ContextVar = contextvars.ContextVar("llm_usage_meter", default=None)


class metered:
    """`with metered() as m:` counts the LLM calls made inside the block into m. A thread pool started
    inside does not inherit it on its own: submit with contextvars.copy_context().run."""

    def __enter__(self) -> UsageMeter:
        self.meter = UsageMeter()
        self._token = _meter.set(self.meter)
        return self.meter

    def __exit__(self, *exc):
        _meter.reset(self._token)


def _count(usage: dict) -> None:
    m = _meter.get()
    if m is None:
        _log.info("[usage] LLM call outside any usage scope (%s tokens), not counted for a project",
                  usage.get("total_tokens"))
        return
    m.add(usage)


def _llm(
    messages: list[dict], temperature: float = 0.9, max_tokens: int = 4096
) -> tuple[str, str, int, dict]:
    """Single entry point for all LLM calls — an OpenAI-compatible chat-completions request.

    Tries each model in MODELS in order; for each, retries up to 5 times with exponential
    backoff on connection errors, timeouts, HTTP errors, or an empty/think-only completion.
    Falls through to the next model only after exhausting retries on the current one, or at once on a 400
    (prompt too long for it) or 404 (no such model). Every failed attempt is logged.

    A completion cut off by max_tokens is retried at once with double the budget (up to
    MAX_TOKENS_CEILING): a reasoning model sometimes reasons past the cap, and a same-budget retry at low
    temperature reproduces the identical cut-off (2026-10-08 log: 4x the same 18,357 chars).

    Returns (response_text, model_used, elapsed_ms, usage_dict). Raises LLMUnavailableError
    if every model/attempt combination fails.
    """
    t0 = time.monotonic()
    last_err = None

    for model in MODELS:
        budget = max_tokens
        for attempt in range(5):
            try:
                with _slot(), httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as http:
                    r = http.post(
                        f"{API_BASE}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {API_KEY}",
                            "Content-Type": "application/json",
                            # No x-api-key: BHT's Sophia Lumis gateway (2026-10-07) answers 401
                            # when it is sent alongside a valid Bearer token.
                        },
                        json={
                            "model": model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": budget,
                        },
                    )
                    r.raise_for_status()
                    payload = r.json()
                    raw = payload["choices"][0]["message"]["content"]
                    _count(payload.get("usage") or _estimate_usage(messages, raw or ""))
                    finish_reason = payload["choices"][0].get("finish_reason")
                    if not raw or not raw.strip():
                        last_err = ValueError("empty completion from server")
                        if attempt < 4:
                            time.sleep((2**attempt) + random.uniform(0, 2))
                        continue
                    stripped = _strip(raw)
                    if finish_reason == "length" and budget < MAX_TOKENS_CEILING:
                        # Cut off mid-thought or mid-answer: the output is unusable either way
                        _log.warning("[LLM] hit max_tokens %d (raw %d chars, model %s), retrying with %d",
                                     budget, len(raw), model, min(budget * 2, MAX_TOKENS_CEILING))
                        last_err = ValueError(f"completion cut off at max_tokens {budget}")
                        budget = min(budget * 2, MAX_TOKENS_CEILING)
                        continue
                    if not stripped:
                        # Model returned only a <think> block with no actual output
                        _log.warning("[LLM] empty after strip (raw %d chars, model %s): %s", len(raw), model, raw[:200])
                        last_err = ValueError("empty completion after stripping think blocks")
                        if attempt < 4:
                            time.sleep((2**attempt) + random.uniform(0, 2))
                        continue
                    usage = payload.get("usage") or _estimate_usage(messages, raw)
                elapsed = round((time.monotonic() - t0) * 1000)
                return stripped, model, elapsed, usage
            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.NetworkError,
                httpx.PoolTimeout,
                httpx.HTTPStatusError,
            ) as e:
                last_err = e
                _log.warning("[LLM] %s attempt %d failed: %s", model, attempt + 1, str(e)[:300])
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (400, 404):
                    break  # too long for this model, or no such model: retrying it can't help
                if attempt < 4:
                    wait = (2**attempt) + random.uniform(0, 2)
                    time.sleep(wait)
        time.sleep(1)

    raise LLMUnavailableError(str(last_err))


def _llm_json(
    messages: list[dict], temperature: float = 0.3, max_tokens: int = 4096
) -> tuple[dict, str, int, dict]:
    """Like _llm, but parses the response as JSON — strips ```json fences first, then falls
    back to regex-extracting an embedded [...] or {...} if the model wrapped the JSON in
    prose. Returns {} (not an exception) if no valid JSON could be extracted at all —
    callers must check for that rather than assume a populated dict."""
    content, model, elapsed, usage = _llm(messages, temperature=temperature, max_tokens=max_tokens)
    content = content.strip()

    if "```json" in content:
        m = re.search(r'```json\s*([\s\S]*?)```', content)
        if m:
            content = m.group(1).strip()
    elif "```" in content:
        m = re.search(r'```\s*([\s\S]*?)```', content)
        if m:
            content = m.group(1).strip()
    content = content.strip()

    if _LOG_LLM:
        prompt_preview = " | ".join(f"{m['role']}: {str(m.get('content',''))[:120]}" for m in messages)
        _log.info("[LLM] PROMPT (%d msgs): %s", len(messages), prompt_preview)
        _log.info("[LLM] RESPONSE (%d chars, %s): %s", len(content), model, content[:500])

    try:
        return json.loads(content), model, elapsed, usage
    except json.JSONDecodeError:
        try:
            return json.loads(content + "}"), model, elapsed, usage
        except json.JSONDecodeError:
            pass

    _log.warning("[LLM] JSON parse failed (%d chars): %s", len(content), content[:2000])
    # Model returned text around the JSON — try to extract embedded array or object.
    # Strategy 1: greedy regex (handles noise before/after when note has no braces)
    for pattern in (r'\[[\s\S]*\]', r'\{[\s\S]*\}'):
        m = re.search(pattern, content)
        if m:
            try:
                return json.loads(m.group(0)), model, elapsed, usage
            except json.JSONDecodeError:
                pass
    # Strategy 2: scan backward from each } — handles "JSON then note that contains }"
    idx = content.rfind('}')
    while idx >= 0:
        try:
            return json.loads(content[:idx + 1]), model, elapsed, usage
        except json.JSONDecodeError:
            idx = content.rfind('}', 0, idx)
    return {}, model, elapsed, usage


def _strip(text: str) -> str:
    """Remove reasoning-model artifacts (<think> blocks, ```thinking fences, HTML comments)
    that some models emit before/around their actual answer."""
    text = text.strip()

    import re

    # Strip completed <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip unclosed <think> block — model hit max_tokens while still reasoning
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)

    text = re.sub(r"```thinking.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)

    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    text = text.strip()

    while text.startswith("\n\n") or text.startswith("...\n\n"):
        text = text.lstrip("\n").lstrip(".")

    return text.strip()


def _visible_answer(raw: str) -> str:
    """The part of a streamed completion to show: nothing while the model is still inside its
    leading <think> block, everything after </think> once it closes."""
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[1].lstrip()
    if raw.lstrip().startswith("<think>") or "<think>".startswith(raw.lstrip()):
        return ""  # thinking (or not yet enough text to tell)
    return raw


def _llm_stream(messages: list[dict], temperature: float = 0.3, max_tokens: int = MAX_TOKENS_CEILING):
    """Streaming variant of _llm for the paper chat. Yields the visible answer so far (a growing string,
    empty while the model is still thinking), a ("retry", next model, attempt) tuple before each retry
    (so the chat can say the model isn't answering instead of "Thinking..."), and finally a dict
    {"model", "usage", "finish_reason"}.

    Falls back through MODELS like _llm, but only until the first chunk arrives: a stream that breaks
    after that raises, since the user has already seen part of the answer. A request the model rejects
    as too long (400) is not retried on the same model."""
    last_err = None
    for model in MODELS:
        for attempt in range(3):
            if last_err is not None:
                yield ("retry", model, attempt + 1)
            got_chunk, raw, counted = False, "", False
            try:
                with _slot(), httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as http:
                    with http.stream(
                        "POST", f"{API_BASE}/chat/completions",
                        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                        json={"model": model, "messages": messages, "temperature": temperature,
                              "max_tokens": max_tokens, "stream": True,
                              "stream_options": {"include_usage": True}},
                    ) as r:
                        if r.status_code >= 400:
                            r.read()
                            raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:300]}", request=r.request, response=r)
                        raw, usage, finish = "", None, None
                        for line in r.iter_lines():
                            if not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            chunk = json.loads(line[6:])
                            usage = chunk.get("usage") or usage
                            if not chunk.get("choices"):
                                continue
                            choice = chunk["choices"][0]
                            finish = choice.get("finish_reason") or finish
                            delta = (choice.get("delta") or {}).get("content") or ""
                            if delta:
                                got_chunk = True
                                raw += delta
                                yield _visible_answer(raw)
                        if not got_chunk:
                            raise ValueError("empty completion from server")
                        if _LOG_LLM:
                            _log.info("[LLM] STREAM RESPONSE (%d chars, %s): %s", len(raw), model, raw[:500])
                        final = {"model": model, "usage": usage or _estimate_usage(messages, raw),
                                 "finish_reason": finish}
                        _count(final["usage"])
                        counted = True
                        yield final
                        return
            except (httpx.HTTPError, ValueError) as e:
                if got_chunk:
                    raise
                last_err = e
                too_long = isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 400
                _log.warning("[LLM] stream %s attempt %d failed: %s", model, attempt + 1, str(e)[:300])
                if too_long:
                    break
                time.sleep((2**attempt) + random.uniform(0, 1))
            finally:
                if got_chunk and not counted:  # stopped by the user or broken mid-answer: estimate it
                    _count(_estimate_usage(messages, raw))
    raise LLMUnavailableError(str(last_err))
