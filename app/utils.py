import html as html_lib
import io
import os
import re
import threading
import time
from contextlib import contextmanager
from typing import Optional
from urllib.parse import urlparse

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


# The nightly job (scripts/nightly_crawl.py) keeps this file while it runs. The logs folder is on the volume the
# web app shares with it, so the web app can see the nightly run and leave it most of the shared LLM quota
# (llm._limit) and Semantic Scholar rate (_ScholarLockCtx). Older than NIGHTLY_MARKER_MAX_AGE_S: left behind by
# a run that was killed, ignored.
NIGHTLY_MARKER = os.path.join(
    os.environ.get("LOGS_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"),
    "nightly.running")
NIGHTLY_MARKER_MAX_AGE_S = 12 * 3600
_holding_marker = False  # True in the nightly job's own process


def nightly_running() -> bool:
    """Whether the nightly job is running in another process (for the web app)."""
    if _holding_marker:
        return False
    try:
        return time.time() - os.path.getmtime(NIGHTLY_MARKER) < NIGHTLY_MARKER_MAX_AGE_S
    except OSError:
        return False


@contextmanager
def nightly_marker():
    """Held by the nightly job for its whole run (see NIGHTLY_MARKER)."""
    global _holding_marker
    os.makedirs(os.path.dirname(NIGHTLY_MARKER), exist_ok=True)
    with open(NIGHTLY_MARKER, "w") as f:
        f.write(f"{os.getpid()}\n")
    _holding_marker = True
    try:
        yield
    finally:
        _holding_marker = False
        os.remove(NIGHTLY_MARKER)


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
# What the model reads of a paper (screening, summary, contributions, why-line, chat), after the reference list is
# left out: ~100k tokens, so the large model (196k context) has room for the rest of the prompt and the answer.
# Before 2026-11-12 the text was cut at 80,000 characters with the references in it: about half of all papers
# (median 20 pages on prod) were read only in part.
FULL_TEXT_CHARS = 400_000
_REF_HEADING_RE = re.compile(r"\b(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\b")
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
REFS_YEARS_PER_1000 = 2.5  # a reference list has a year every few hundred characters; prose rarely does
REFS_END_GAP = 1500  # this long without a year after one: the reference list is over (an appendix follows)
# An appendix heading ("Appendix A Details of ...", "Appendix B.1 Setup"); not a back-link in an arXiv HTML reference
# ("Appendix E .", "Appendix B , §7")
_APPENDIX_HEADING_RE = re.compile(r"\bAppendix\s+[A-Z](?:\.\d+)*\s+[A-Z][a-z]")


def _years_per_1000(text: str) -> float:
    return 1000 * len(_YEAR_RE.findall(text)) / max(len(text), 1)


def drop_references(text: str) -> str:
    """The paper without its reference list, which the model doesn't need. It starts at the first References
    heading in the last 70% of the text (one in the table of contents comes earlier) that is followed by what
    looks like references (a year every few hundred characters). It ends at an appendix heading, or where the years
    stop for REFS_END_GAP characters, whichever comes first: what follows (the appendix) stays, as does everything
    before (the conclusion, also on the same page). When nothing looks like a reference list, nothing is cut."""
    for m in _REF_HEADING_RE.finditer(text):
        start = m.start()
        if start < 0.3 * len(text) or _years_per_1000(text[start:start + 3000]) < REFS_YEARS_PER_1000:
            continue
        years = [y.end() for y in _YEAR_RE.finditer(text, start)]
        end = len(text)
        for a, b in zip(years, years[1:] + [len(text)]):
            if b - a > REFS_END_GAP:
                stop = text.find(". ", a, a + 300)  # finish the last entry: up to its next full stop
                end = stop + 2 if stop >= 0 else a
                break
        heading = _APPENDIX_HEADING_RE.search(text, start + 200, end)  # an appendix that cites a lot has years too
        if heading:
            end = heading.start()
        return text[:start] + " [References left out] " + text[end:]
    return text
    return text

# arXiv's own HTML has nearly every paper from the day it is announced; ar5iv converts with a lag of
# weeks and, until then, redirects to the abstract page (which once passed for the full text)
FULL_TEXT_HTML_URLS = ("https://arxiv.org/html/{}", "https://ar5iv.labs.arxiv.org/html/{}")


_LINK_SKIP_HOSTS = ("arxiv.org", "doi.org", "creativecommons.org", "latexml", "nist.gov")


def _show_link_target(m: re.Match) -> str:
    """An <a href> whose text hides where it goes ("Code", "Project page", an icon) keeps its URL in the text, as
    "Code (https://github.com/...)", so the paper's own links can be found in the text (app/resources.py). Not
    for the bibliography's links (ltx_bib_external) or arXiv/DOI links, which are no use there."""
    attrs, inner = m.group(1), m.group(2)
    href = re.search(r'href="(https?://[^"]+)"', attrs)
    if not href or "ltx_bib_external" in attrs:
        return m.group(0)
    url = html_lib.unescape(href.group(1))
    host = (urlparse(url).hostname or "").lower()
    if any(h in host for h in _LINK_SKIP_HOSTS):
        return m.group(0)
    text = re.sub(r"<[^>]+>", " ", inner)
    if re.sub(r"^https?://(www\.)?", "", url).rstrip("/") in text:
        return m.group(0)
    return f"<a{attrs}>{inner}</a> ({url})"


def _html_to_text(html: str) -> str:
    article = re.search(r"<article\b.*</article>", html, flags=re.DOTALL | re.IGNORECASE)
    html = article.group(0) if article else html  # the paper only, not the site's menus
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<a\b([^>]*)>(.*?)</a>", _show_link_target, html, flags=re.DOTALL | re.IGNORECASE)
    return html_lib.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))).strip()


def fetch_paper_full_text(arxiv_id: str, max_chars: int = FULL_TEXT_CHARS) -> str | None:
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
            return drop_references(text)[:max_chars]
    return None


# bioRxiv PDFs: one at a time, 7 s apart (Crawl-delay in biorxiv.org/robots.txt). The wait is before a
# download, for what is left of the gap since the last one, so a single download (a chat) doesn't wait
_biorxiv_lock = threading.Lock()
_biorxiv_last = [float("-inf")]  # time.monotonic() when the last bioRxiv download ended
BIORXIV_GAP_S = 7


def _is_biorxiv(url: str) -> bool:
    return (urlparse(url).hostname or "").endswith("biorxiv.org")


@contextmanager
def _biorxiv_gap():
    with _biorxiv_lock:
        time.sleep(max(0.0, _biorxiv_last[0] + BIORXIV_GAP_S - time.monotonic()))
        try:
            yield
        finally:
            _biorxiv_last[0] = time.monotonic()


def _biorxiv_html_to_text(html: str) -> str:
    """The paper's text from a bioRxiv .full page: from its fulltext-view div to the "Back to top" link after
    the references (the site's menus, share links and subject list are outside), without the "Download
    figure" buttons and the reference link labels. Cleaner than the PDF's text, which has line numbers and
    the license line of every page in it (compared on 6 papers, 2026-10-20)."""
    m = re.search(r'<div[^>]*class="[^"]*\bfulltext-view\b', html)
    if not m:
        return ""
    end = html.find("Back to top", m.start())
    text = _html_to_text(html[m.start():end if end > 0 else len(html)])
    text = re.sub(r"Download figure|Open in new tab|OpenUrl|CrossRef|PubMed|Google Scholar|Web of Science|↵", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# A crawl reads a bioRxiv page for the text (deep check) and again for figure 1 (once accepted); keeping the
# answer for an hour makes that one download. A missing HTML version is kept too, so it isn't asked twice.
_biorxiv_pages: dict[str, tuple[float, Optional[str]]] = {}
BIORXIV_PAGE_TTL_S = 3600


def fetch_biorxiv_page(pdf_url: str) -> str | None:
    """The HTML version of a bioRxiv paper (pdf_url without ".pdf"), or None if there is none yet: bioRxiv
    adds it about 4 to 6 days after posting (none of 17 papers 1 to 3 days old had it, all of 12 papers 6 and
    10 days old did, checked 2026-10-20)."""
    url = pdf_url.removesuffix(".pdf")
    now = time.monotonic()
    for k in [k for k, (t, _) in _biorxiv_pages.items() if now - t > BIORXIV_PAGE_TTL_S]:
        del _biorxiv_pages[k]
    if url in _biorxiv_pages:
        return _biorxiv_pages[url][1]
    try:
        with _biorxiv_gap():
            with _http_client(timeout=30.0, follow_redirects=True) as http:
                r = http.get(url)
    except httpx.HTTPError as e:
        print(f"HTML fetch failed for {url}: {e}")
        return None  # not kept: a network error is tried again
    page = r.text if r.status_code == 200 and re.search(r'<div[^>]*class="[^"]*\bfulltext-view\b', r.text) else None
    _biorxiv_pages[url] = (now, page)
    return page


def fetch_biorxiv_full_text(pdf_url: str, max_chars: int = FULL_TEXT_CHARS) -> str | None:
    """A bioRxiv paper's text from its HTML version, or None while there is none (then the PDF is read)."""
    page = fetch_biorxiv_page(pdf_url)
    text = _biorxiv_html_to_text(page) if page else ""
    return drop_references(text)[:max_chars] if len(text) >= MIN_TEXT_CHARS else None


_BIORXIV_F1 = re.compile(r'<div id="F1"[^>]*>.*?<a href="(https://www\.biorxiv\.org/[^"?]+/F1\.large\.jpg)', re.S)


def biorxiv_figure(page: str) -> tuple[Optional[str], Optional[str]]:
    """Figure 1 of a bioRxiv HTML page: (image URL, caption with its "Figure 1:" label), or (None, None).
    Every one of 20 pages 6 to 14 days old had it, and the image loads when linked from another site
    (checked 2026-10-20)."""
    m = _BIORXIV_F1.search(page or "")
    if not m:
        return None, None
    cap = re.search(r'<div class="fig-caption"[^>]*>(.*?)<div class="sb-div caption-clear"', page[m.end():m.end() + 20000], re.S)
    caption = html_lib.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", cap.group(1)))).strip() if cap else None
    return m.group(1), caption


def fetch_biorxiv_figure(pdf_url: str) -> tuple[Optional[str], Optional[str]]:
    return biorxiv_figure(fetch_biorxiv_page(pdf_url))


PDF_LINK_PAGES = 3  # the paper's own links (code, project page) are on its first pages


def _pdf_page_links(page, text: str) -> str:
    """The web links of a PDF page whose URL isn't in its text (a "Code" or icon link), as " (url)" each, so
    app/resources.py can find them, like _show_link_target for the HTML version."""
    out = []
    for annot in page.get("/Annots") or []:
        a = annot.get_object()
        uri = (a.get("/A") or {}).get("/URI") if a.get("/Subtype") == "/Link" else None
        if not isinstance(uri, str) or not uri.startswith(("http://", "https://")):
            continue
        host = (urlparse(uri).hostname or "").lower()
        if any(h in host for h in _LINK_SKIP_HOSTS) or re.sub(r"^https?://(www\.)?", "", uri).rstrip("/") in text \
                or uri in out:
            continue
        out.append(uri)
    return "".join(f" ({u})" for u in out)


def fetch_pdf_text(url: str, arxiv: bool, max_chars: int = FULL_TEXT_CHARS) -> str | None:
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
        elif _is_biorxiv(url):
            with _biorxiv_gap():  # bioRxiv's robots.txt asks for 7 s between requests
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
        pages = [page.extract_text() or "" for page in reader.pages]
        text = " ".join(t + (_pdf_page_links(page, t) if i < PDF_LINK_PAGES else "")
                        for i, (page, t) in enumerate(zip(reader.pages, pages)))
    except PdfReadError as e:  # a broken PDF: the caller falls back to the abstract
        print(f"unreadable PDF at {url}: {e}")
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return drop_references(text)[:max_chars] if len(text) >= MIN_TEXT_CHARS else None


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


_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s?&#]+")


def doi_from_url(url: Optional[str]) -> Optional[str]:
    """The DOI in a paper's URL (doi.org/10..., a publisher path or query with the DOI in it), or None."""
    m = _DOI_RE.search(url or "")
    if not m:
        return None
    doi = re.sub(r"/(pdf|full|abstract|epdf)$", "", m.group(0))
    doi = re.sub(r"(v\d+)?(\.full)?(\.pdf)?$", "", doi) if doi.startswith(("10.1101/", "10.64898/")) else doi  # bioRxiv
    return re.sub(r"\.pdf$", "", doi).rstrip("/.")


def base_arxiv_id(arxiv_id: str) -> str:
    """Strip a trailing version suffix: '2605.23872v2' -> '2605.23872'. Non-arXiv ids pass through."""
    return re.sub(r"v\d+$", "", arxiv_id)
