import re
import threading
import time
from typing import Optional

import httpx

# Identifies this tool to arXiv/publisher servers instead of the default "python-httpx/x.xx" —
# an unlabeled bot signature with no contact info is exactly what gets IPs blocked.
USER_AGENT = "paper-agent/1.0 (+https://github.com/MassEast/paper-agent)"


def _http_client(**kwargs) -> httpx.Client:
    """httpx.Client with the project's User-Agent set. Use instead of a bare httpx.Client(...)
    for any outbound request (arXiv, ar5iv, Semantic Scholar, publisher sites)."""
    headers = kwargs.pop("headers", {}) or {}
    headers = {"User-Agent": USER_AGENT, **headers}
    return httpx.Client(headers=headers, **kwargs)


# Global arXiv/ar5iv rate-limit enforcement: at most one request in flight at a time, with a
# 3.5 s gap enforced after each one releases the lock — covers export.arxiv.org (API search),
# ar5iv.labs.arxiv.org (full-text fetch), and arxiv.org/pdf|html|abs (figure, page count).
# arXiv's bulk-access policy expects exactly this kind of single-connection throttling; prior
# to this, only the export.arxiv.org API calls in crawl.py went through it — ar5iv/PDF/HTML
# fetches were unthrottled and ran with up to 10-way parallelism.
_arxiv_lock = threading.Lock()
_arxiv_waiters = 0  # GIL-safe: number of threads currently waiting to acquire _arxiv_lock


class _ArxivLockCtx:
    """Context manager for _arxiv_lock that tracks the waiter count and enforces a 3.5 s gap
    between requests (sleeps before releasing, so the next caller never violates the limit).

    Usage (blocking):    with _ArxivLockCtx(): ...
    Usage (with timeout): with _ArxivLockCtx(timeout=10.0) as ok:
                              if not ok: <handle busy>
    """

    def __init__(self, timeout: Optional[float] = None):
        self._timeout = timeout
        self.acquired = False

    def __enter__(self) -> bool:
        global _arxiv_waiters
        _arxiv_waiters += 1
        self.acquired = False
        try:
            if self._timeout is not None:
                self.acquired = _arxiv_lock.acquire(timeout=self._timeout)
            else:
                _arxiv_lock.acquire()
                self.acquired = True
        finally:
            _arxiv_waiters -= 1  # no longer waiting (held, timed-out, or exception)
        return self.acquired

    def __exit__(self, *_):
        if self.acquired:
            time.sleep(3.5)  # enforce ≥3.5s gap between arXiv/ar5iv requests
            _arxiv_lock.release()


def arxiv_queue_depth() -> int:
    """Return how many threads are currently waiting to acquire the arXiv lock."""
    return _arxiv_waiters


def extract_arxiv_id(text: str) -> str | None:
    patterns = [
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?",
        r"(?:^|[\s/])([0-9]{4}\.[0-9]{4,5})(?:v\d+)?(?:$|[\s\.])",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def fetch_arxiv_title(arxiv_id: str) -> str | None:
    try:
        with _ArxivLockCtx():
            with _http_client(timeout=10.0) as http:
                r = http.get(
                    "https://export.arxiv.org/api/query",
                    params={"id_list": arxiv_id, "max_results": 1},
                )
                r.raise_for_status()
                candidates = re.findall(r"<title>([^<]+)</title>", r.text)
                return candidates[1].strip() if len(candidates) > 1 else None
    except Exception:
        pass
    return None


def fetch_paper_full_text(arxiv_id: str, max_chars: int = 80000) -> str | None:
    try:
        with _ArxivLockCtx():
            with _http_client(timeout=30.0, follow_redirects=True) as http:
                r = http.get(f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}")
                if r.status_code != 200:
                    return None
                html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", r.text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", html)
                return re.sub(r"\s+", " ", text).strip()[:max_chars]
    except Exception as e:
        print(f"ar5iv fetch failed for {arxiv_id}: {e}")
    return None


def fetch_paper_figure(arxiv_id: str) -> str | None:
    try:
        with _ArxivLockCtx():
            with _http_client(timeout=15.0) as http:
                resp = http.get(f"https://arxiv.org/abs/{arxiv_id}")
        for src in re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', resp.text, re.IGNORECASE):
            if any(x in src.lower() for x in ['figure', 'fig', '.png', '.jpg']) or src.startswith('http'):
                return ('https:' + src if src.startswith('//') else
                        'https://arxiv.org' + src if src.startswith('/') else src)
    except Exception:
        pass
    return None
