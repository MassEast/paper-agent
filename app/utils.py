import io
import re
import threading
import time
from typing import Optional

import httpx

# Identifies this tool to arXiv/publisher servers instead of the default "python-httpx/x.xx" —
# an unlabeled bot signature with no contact info is exactly what gets IPs blocked.
USER_AGENT = "paper-agent/1.0 (+https://github.com/MassEast/paper-agent)"


class UnsafeURLError(httpx.HTTPError):
    """A request to an address that is not on the public internet (see _is_safe_external_url)."""


def _is_safe_external_url(url: str) -> bool:
    """Basic SSRF guard for user-supplied URLs (add-by-URL, /extract-from-url): reject
    non-http(s) schemes and hosts that resolve to a private/loopback/link-local/reserved
    address (RFC1918, 127.0.0.0/8, 169.254.0.0/16 incl. the cloud-metadata address, etc.).
    Fails closed — any resolution error is treated as unsafe."""
    import socket
    import ipaddress
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        for family, _, _, _, sockaddr in socket.getaddrinfo(parsed.hostname, None):
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True
    except Exception:
        return False


def _refuse_internal(request: httpx.Request) -> None:
    # Request hook: httpx runs it for the first request and again for every redirect hop.
    if not _is_safe_external_url(str(request.url)):
        raise UnsafeURLError(f"refused non-public URL: {request.url}")


def _http_client(public_only: bool = False, **kwargs) -> httpx.Client:
    """httpx.Client with the project's User-Agent set. Use instead of a bare httpx.Client(...)
    for any outbound request (arXiv, ar5iv, Semantic Scholar, publisher sites).

    public_only=True for URLs a user supplied (or a page they supplied pointed to): every request,
    redirects included, must go to a public address, so a public page can't redirect the server
    into the cluster. Checking only the first URL missed that (idea from external_resources/
    research-toolset-main/src/core/url_security.py, make_request_guard)."""
    headers = kwargs.pop("headers", {}) or {}
    headers = {"User-Agent": USER_AGENT, **headers}
    if public_only:
        kwargs["event_hooks"] = {"request": [_refuse_internal]}
    return httpx.Client(headers=headers, **kwargs)


# Global arXiv/ar5iv rate-limit enforcement: at most one request in flight at a time, with a
# 5 s gap enforced after each one releases the lock — covers export.arxiv.org (API search),
# ar5iv.labs.arxiv.org (full-text fetch), and arxiv.org/pdf|html|abs (figure, page count).
# arXiv's bulk-access policy expects exactly this kind of single-connection throttling; prior
# to this, only the export.arxiv.org API calls in crawl.py went through it — ar5iv/PDF/HTML
# fetches were unthrottled and ran with up to 10-way parallelism.
_arxiv_lock = threading.Lock()
_arxiv_waiters = 0  # GIL-safe: number of threads currently waiting to acquire _arxiv_lock


class _ArxivLockCtx:
    """Context manager for _arxiv_lock that tracks the waiter count and enforces a 5 s gap
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
            time.sleep(5)  # enforce ≥5s gap between arXiv/ar5iv requests
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


MIN_TEXT_CHARS = 2000  # anything shorter is an error page or a stub, not the paper

# arXiv's own HTML has nearly every paper from the day it is announced; ar5iv converts with a lag of
# weeks and, until then, redirects to the abstract page (which once passed for the full text)
FULL_TEXT_HTML_URLS = ("https://arxiv.org/html/{}", "https://ar5iv.labs.arxiv.org/html/{}")


def _html_to_text(html: str) -> str:
    article = re.search(r"<article\b.*</article>", html, flags=re.DOTALL | re.IGNORECASE)
    html = article.group(0) if article else html  # the paper only, not the site's menus
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def fetch_paper_full_text(arxiv_id: str, max_chars: int = 80000) -> str | None:
    """The paper's text from its HTML version, or None if there is none (yet)."""
    for url in FULL_TEXT_HTML_URLS:
        url = url.format(arxiv_id)
        try:
            with _ArxivLockCtx():
                with _http_client(timeout=30.0, follow_redirects=True) as http:
                    r = http.get(url)
        except httpx.HTTPError as e:
            print(f"HTML fetch failed for {url}: {e}")
            continue
        if r.status_code != 200 or not r.url.path.startswith("/html/"):  # e.g. redirected to /abs/
            continue
        text = _html_to_text(r.text)
        if len(text) >= MIN_TEXT_CHARS:
            return text[:max_chars]
    return None


def fetch_pdf_text(url: str, arxiv: bool, max_chars: int = 80000) -> str | None:
    """Text extracted from a PDF, or None if it can't be downloaded or read. arxiv=True waits for
    the arXiv rate limit."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    def _get():
        # A non-arXiv PDF link comes from a page a user added
        with _http_client(public_only=not arxiv, timeout=httpx.Timeout(60.0), follow_redirects=True) as http:
            return http.get(url)

    try:
        if arxiv:
            with _ArxivLockCtx():
                r = _get()
        else:
            r = _get()
    except httpx.HTTPError as e:
        print(f"PDF download failed for {url}: {e}")
        return None
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        print(f"no PDF at {url} (HTTP {r.status_code})")
        return None
    try:
        reader = PdfReader(io.BytesIO(r.content), strict=False)
        text = " ".join(page.extract_text() or "" for page in reader.pages)
    except PdfReadError as e:  # a broken PDF: the caller falls back to the abstract
        print(f"unreadable PDF at {url}: {e}")
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] if len(text) >= MIN_TEXT_CHARS else None


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


def base_arxiv_id(arxiv_id: str) -> str:
    """Strip a trailing version suffix: '2605.23872v2' -> '2605.23872'. Non-arXiv ids pass through."""
    return re.sub(r"v\d+$", "", arxiv_id)
