"""bioRxiv as a crawl source.

bioRxiv's public API (api.biorxiv.org/details) lists every preprint posted in a date range but has no keyword
search, so a search here reads the whole window page by page and keeps the papers whose title or abstract
contain a keyword. A nightly window of five days is a few hundred papers, a handful of requests.

The result has the same shape as an arXiv candidate. A bioRxiv paper has no arXiv id, so it is stored under a
synthetic "web:biorxiv-<hash of the DOI>" id like any non-arXiv paper (Paper.is_arxiv is False). The id has no
slash because it goes into URLs. Its DOI is in the landing page link kept as pdf_url. Only the abstract is
screened: bioRxiv is not on OPEN_ACCESS_PDF_DOMAINS, so no full text is fetched.
"""

import hashlib
import logging
import re
import time
from datetime import date, datetime
from typing import Optional

from app.utils import _http_client

_log = logging.getLogger(__name__)

API_URL = "https://api.biorxiv.org/details/biorxiv/{start}/{end}/{cursor}/json"
LANDING_URL = "https://www.biorxiv.org/content/{doi}v{version}"
REQUEST_TRIES = 3


def biorxiv_paper_id(doi: str) -> str:
    return "web:biorxiv-" + hashlib.sha256(doi.lower().encode()).hexdigest()[:12]


_STOPWORDS = {"and", "for", "the", "with", "from", "into", "via"}


def _stem(word: str) -> str:
    return word[:6]  # crude: "resistance" and "resistant" meet, "microscopy" and "microscope" meet


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def matched_keywords(text: str, keywords: list[str]) -> list[str]:
    """Keywords whose words all appear in text (any order, word endings ignored). Looser than arXiv's phrase
    search on purpose: the screening decides, and a phrase like "antimicrobial resistance morphology" would
    almost never appear verbatim."""
    present = {_stem(w) for w in _words(text)}
    out = []
    for k in keywords:
        stems = [_stem(w) for w in _words(k) if len(w) > 2 and w not in _STOPWORDS]
        if stems and all(s in present for s in stems):
            out.append(k)
    return out


def _get_page(http, start: date, end: date, cursor: int) -> Optional[dict]:
    """One page of the interval listing, with a few retries; None if it kept failing."""
    url = API_URL.format(start=start.isoformat(), end=end.isoformat(), cursor=cursor)
    for attempt in range(REQUEST_TRIES):
        try:
            r = http.get(url)
            if r.status_code == 200:
                return r.json()
            _log.warning("[bioRxiv] %s -> HTTP %s", url, r.status_code)
        except Exception as e:
            _log.warning("[bioRxiv] %s failed: %s", url, e)
        if attempt < REQUEST_TRIES - 1:
            time.sleep(2 * 3 ** attempt)
    return None


def fetch_window(date_from: date, date_to: date, warnings: Optional[list[str]] = None) -> list[dict]:
    """Every first version posted from date_from to date_to (inclusive), as the API returns it. If the
    listing breaks off, what was read so far is returned and a warning says the window is incomplete."""
    items: list[dict] = []
    cursor = 0
    with _http_client(timeout=30.0, follow_redirects=True) as http:
        while True:
            page = _get_page(http, date_from, date_to, cursor)
            if page is None:
                if warnings is not None:
                    warnings.append(f"bioRxiv did not answer after {len(items)} papers; the rest of "
                                    f"{date_from} to {date_to} was not searched")
                break
            batch = page.get("collection") or []
            items += batch
            total = int((page.get("messages") or [{}])[0].get("total") or 0)
            cursor += len(batch)
            if not batch or cursor >= total:
                break
    return [it for it in items if str(it.get("version", "1")) == "1"]


def _candidate(item: dict, names: list[str]) -> Optional[dict]:
    doi, title = item.get("doi"), (item.get("title") or "").strip()
    if not doi or not title:
        return None
    try:
        posted = datetime.strptime(item["date"], "%Y-%m-%d").date()
    except (KeyError, ValueError):
        posted = None
    institution = (item.get("author_corresponding_institution") or "").strip()
    return {
        "arxiv_id": biorxiv_paper_id(doi),
        "doi": doi,
        "title": " ".join(title.split()),
        "authors": [a.strip() for a in (item.get("authors") or "").split(";") if a.strip()],
        "abstract": " ".join((item.get("abstract") or "").split()),
        "pdf_url": LANDING_URL.format(doi=doi, version=item.get("version") or 1),
        "published_date": posted,
        "year": posted.year if posted else None,
        "institutions": [institution] if institution else [],
        "categories": [item["category"]] if item.get("category") else [],
        "source": "biorxiv",
        "found_via": [{"via": "biorxiv", "name": k} for k in names],
    }


def search_biorxiv(keywords: list[str], date_from: date, date_to: date,
                   warnings: Optional[list[str]] = None) -> list[dict]:
    """bioRxiv preprints first posted in the window whose title or abstract match a keyword."""
    if not keywords:
        return []
    found, seen = [], set()
    for item in fetch_window(date_from, date_to, warnings):
        names = matched_keywords((item.get("title") or "") + " " + (item.get("abstract") or ""), keywords)
        paper = _candidate(item, names) if names else None
        if paper and paper["arxiv_id"] not in seen:
            seen.add(paper["arxiv_id"])
            found.append(paper)
    _log.info("[bioRxiv] %d matching papers, %s to %s", len(found), date_from, date_to)
    return found
