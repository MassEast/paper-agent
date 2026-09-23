import json
import logging
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


def _llm(
    messages: list[dict], temperature: float = 0.9, max_tokens: int = 4096
) -> tuple[str, str, int, dict]:
    """Single entry point for all LLM calls — an OpenAI-compatible chat-completions request.

    Tries each model in MODELS in order; for each, retries up to 5 times with exponential
    backoff on connection errors, timeouts, HTTP errors, or an empty/think-only completion.
    Falls through to the next model only after exhausting retries on the current one.

    Returns (response_text, model_used, elapsed_ms, usage_dict). Raises LLMUnavailableError
    if every model/attempt combination fails.
    """
    t0 = time.monotonic()
    last_err = None

    for model in MODELS:
        for attempt in range(5):
            try:
                with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as http:
                    r = http.post(
                        f"{API_BASE}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {API_KEY}",
                            "Content-Type": "application/json",
                            # Some OpenAI-compatible gateways (including ours) check this
                            # instead of, or in addition to, the Authorization header.
                            # Harmless extra header for providers that only look at Bearer auth.
                            "x-api-key": API_KEY,
                        },
                        json={
                            "model": model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                    )
                    r.raise_for_status()
                    payload = r.json()
                    raw = payload["choices"][0]["message"]["content"]
                    if not raw or not raw.strip():
                        last_err = ValueError("empty completion from server")
                        if attempt < 4:
                            time.sleep((2**attempt) + random.uniform(0, 2))
                        continue
                    stripped = _strip(raw)
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
