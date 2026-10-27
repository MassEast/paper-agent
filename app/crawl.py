import json
import logging
import re
import threading
import os
import time
import hashlib
import html
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from typing import Optional

_log = logging.getLogger(__name__)

import contextvars

import arxiv
import httpx

from app import db
from app.models import Project, Paper, PaperSummary, ProjectPaper, CrawlLog, ScreenedPaper, compute_screening_hash
from app.llm import _llm, _llm_json, LLMUnavailableError, MODELS as LLM_MODELS
import app.prompts as prompts
from app.usage import counted, counts
from app.biorxiv import search_biorxiv, biorxiv_paper_id, fetch_versions as biorxiv_versions, PDF_URL as BIORXIV_PDF_URL
from app.utils import _ArxivLockCtx, arxiv_queue_depth, _http_client, _is_safe_external_url, USER_AGENT, base_arxiv_id, \
    _is_biorxiv, _biorxiv_gap, fetch_biorxiv_figure, doi_from_url, nightly_running

API_KEY = os.environ.get("LLM_API_KEY")
SCHOLAR_API_KEY = os.environ.get("SCHOLAR_API_KEY")

# How many My Collection papers to include as LLM reference/calibration context (keyword
# extraction + relevance screening) — see _select_reference_papers. This prefix is identical
# across every candidate paper screened in a crawl, so with prefix caching on the inference
# server it's paid once per crawl rather than recomputed per paper; it still adds to KV-cache
# memory pressure for the crawl's duration though, so don't set this arbitrarily high.
REFERENCE_PAPERS_LIMIT = int(os.environ.get("REFERENCE_PAPERS_LIMIT", "15"))

# Domains we'll actually download a PDF from for a manually-added ("web:") paper — everything
# else only gets its landing-page metadata (title/abstract/authors/doi) read, never a PDF
# download. Publisher PDFs (ACM, IEEE, Elsevier, ScienceDirect, ...) are excluded on purpose:
# a server-side bot repeatedly downloading them, possibly from a licensed institutional IP, is
# exactly the pattern that gets flagged by publishers and can risk the license. Starting list —
# extend as needed.
OPEN_ACCESS_PDF_DOMAINS = {
    "arxiv.org",
    "ar5iv.labs.arxiv.org",
    "aclanthology.org",
    "proceedings.mlr.press",
    "openreview.net",
    "proceedings.neurips.cc",
    "openaccess.thecvf.com",  # CVPR / ICCV / WACV — Computer Vision Foundation's own OA repo
    "ojs.aaai.org",  # AAAI / IAAI proceedings — open access by AAAI policy
    "ijcai.org",  # IJCAI proceedings
    "jmlr.org",  # Journal of Machine Learning Research — always open
    "biorxiv.org",  # welcomes machine reading, link back instead of re-hosting (biorxiv.org/tdm); 7 s apart, see utils
}


def _is_open_access_pdf_domain(url: str) -> bool:
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    return host in OPEN_ACCESS_PDF_DOMAINS or any(
        host.endswith("." + d) for d in OPEN_ACCESS_PDF_DOMAINS
    )


MAX_WEB_ENRICH_ATTEMPTS = 3
BIORXIV_FIGURE_DAYS = 30  # the nightly backfill looks for a bioRxiv paper's figure 1 until this long after posting
# Nightly full-text re-read of New Papers screened from the abstract (refresh_from_full_text)
MAX_FULL_TEXT_ATTEMPTS = 3
FULL_TEXT_REFRESH_PER_NIGHT = int(os.environ.get("FULL_TEXT_REFRESH_PER_NIGHT", "50"))
# Backfill Pass 2 retries a paper's citation count at most once per this many days. Without it
# the same ~100 zero-citation papers were re-fetched on each of the 4 nightly cron runs.
CITATION_BACKFILL_COOLDOWN_DAYS = 3


def _enrich_one_web_paper(paper) -> None:
    """One enrichment attempt for a 'web:' paper: refresh pdf_url from the landing page, then
    (if the resolved PDF is on an open-access domain) download it for page_count + institutions.
    Always increments web_enrich_attempts / sets web_enrich_last_attempt_at, regardless of
    outcome — shared by the Pass-3 nightly backfill and the manual "Retry" button in the UI.
    Caller must db.session.commit()."""
    paper.web_enrich_attempts = (paper.web_enrich_attempts or 0) + 1
    paper.web_enrich_last_attempt_at = datetime.utcnow()

    info = extract_paper_info_from_url(paper.page_url or paper.pdf_url)
    if not paper.abstract and info.get("abstract"):
        # Added while the page couldn't be read (a publisher's block page became the title, no abstract):
        # take what the metadata services know now, and rewrite the summary made from the block page
        paper.title = info.get("title") or paper.title
        paper.abstract = info["abstract"]
        if info.get("authors"):
            paper.authors = json.dumps(info["authors"])
        paper.doi = paper.doi or info.get("doi")
        _log.info("[web-enrich] %s: filled in title and abstract (%s)", paper.arxiv_id, paper.title[:60])
        write_summary(paper, paper.abstract)
    if info.get("pdf_url"):
        paper.pdf_url = info["pdf_url"]
    if info.get("page_count") and not paper.page_count:
        paper.page_count = info["page_count"]
    if info.get("pdf_url"):
        _enrich_web_paper_pdf_metadata(paper, info["pdf_url"])


_crawl_lock = threading.Lock()

# _arxiv_lock / _ArxivLockCtx / arxiv_queue_depth now live in app.utils (shared with the
# ar5iv/arxiv.org HTML+PDF fetches there) — imported above. Re-exported under these names
# so existing `from app.crawl import _ArxivLockCtx` / `arxiv_queue_depth` call sites
# (app/routes/projects.py) keep working unchanged.

# Global Semantic Scholar rate-limit enforcement: 1 req/s with API key.
# Same pattern as _arxiv_lock — all Scholar HTTP calls must go through _ScholarLockCtx.
_scholar_lock = threading.Lock()
SCHOLAR_GAP_DURING_NIGHTLY_S = 12.0
_scholar_waiters = 0


class _ScholarLockCtx:
    """Context manager for Semantic Scholar rate-limit lock (1 req/s).

    Acquires the global lock before making any Semantic Scholar API request and sleeps 1.1 s
    after the request completes so the next caller never violates the 1 req/s limit.
    """

    def __init__(self, timeout: Optional[float] = None):
        self._timeout = timeout
        self.acquired = False

    def __enter__(self) -> bool:
        global _scholar_waiters
        _scholar_waiters += 1
        self.acquired = False
        try:
            if self._timeout is not None:
                self.acquired = _scholar_lock.acquire(timeout=self._timeout)
            else:
                _scholar_lock.acquire()
                self.acquired = True
        finally:
            _scholar_waiters -= 1
        return self.acquired

    def __exit__(self, *_):
        if self.acquired:
            # ≥1 s between Semantic Scholar requests (1 per second with a key). While the nightly job, a separate
            # process with its own lock, uses the key at that rate, the web app keeps SCHOLAR_GAP_DURING_NIGHTLY_S
            # so that both together stay below 1 per second.
            time.sleep(SCHOLAR_GAP_DURING_NIGHTLY_S if nightly_running() else 1.1)
            _scholar_lock.release()


def _scholar_headers() -> dict:
    """Return auth headers for Semantic Scholar API (User-Agent + API key if configured)."""
    headers = {"User-Agent": USER_AGENT}
    if SCHOLAR_API_KEY:
        headers["x-api-key"] = SCHOLAR_API_KEY
    return headers


# Semantic Scholar refuses many requests with 429 (or a 5xx) when it is busy, at random and no matter how far
# apart our requests are: waiting longer between tries doesn't raise the odds, so retry after a few seconds.
SCHOLAR_ATTEMPTS = 8
SCHOLAR_RETRY_WAIT_S = 2.0


def _scholar_call(method: str, url: str, timeout: float = 30.0, **kwargs) -> httpx.Response:
    """One Semantic Scholar request through _ScholarLockCtx, tried up to SCHOLAR_ATTEMPTS times while it answers
    429 or 5xx. Returns the last response, so the caller sees a 429 that never went through."""
    for attempt in range(SCHOLAR_ATTEMPTS):
        with _ScholarLockCtx():
            with _http_client(timeout=httpx.Timeout(timeout)) as http:
                r = http.request(method, url, headers=_scholar_headers(), **kwargs)
        if (r.status_code != 429 and r.status_code < 500) or attempt == SCHOLAR_ATTEMPTS - 1:
            return r
        time.sleep(SCHOLAR_RETRY_WAIT_S + SCHOLAR_RETRY_WAIT_S * (attempt % 2))
    return r

# Papers whose details are being fetched in the background right now (start_collection_add_async,
# start_web_paper_add_async): paper id -> {"by": which task, "step": what it is doing}. The paper's card
# shows the step and redraws itself as it changes (/project-paper/<id>/card), so a freshly added paper
# fills in without a reload. In memory: needs the single gunicorn process (see CLAUDE.md).
_enriching: dict[int, dict] = {}


def enrich_step(paper_id: int) -> str | None:
    state = _enriching.get(paper_id)
    return state["step"] if state else None


def _set_step(paper_id: int, by: str, step: str) -> None:
    _enriching[paper_id] = {"by": by, "step": step}


def _end_steps(paper_id: int, by: str) -> None:
    """Only the task that set the status clears it: a web add that turns out to be an arXiv paper hands
    the same paper over to start_collection_add_async, which has set its own status by then."""
    if _enriching.get(paper_id, {}).get("by") == by:
        _enriching.pop(paper_id, None)


def start_collection_add_async(project_id: int, arxiv_id: str, paper_id: int, app) -> None:
    """Background thread: fully enrich a paper that was just added directly to My Collection.

    The Paper row already exists with a placeholder title. This thread fills in abstract,
    authors, figure, institutions, citation count, and generates summary + contributions.
    """
    def _run():
        try:
            _steps()
        finally:
            _end_steps(paper_id, "arxiv")

    def _steps():
        with app.app_context(), counted(project_id, "add"):
            # --- Step 1: Fetch arXiv metadata (needs the global lock) ---
            _set_step(paper_id, "arxiv", "Fetching title, authors and abstract from arXiv")
            with _ArxivLockCtx():
                try:
                    results = list(arxiv.Client(delay_seconds=3.1).results(
                        arxiv.Search(id_list=[arxiv_id])
                    ))
                    if results:
                        r = results[0]
                        paper = Paper.query.get(paper_id)
                        if paper:
                            paper.title = r.title
                            paper.abstract = r.summary
                            paper.authors = json.dumps([str(a) for a in r.authors])
                            paper.pdf_url = r.pdf_url
                            pub_date = r.published.date()
                            paper.published_date = pub_date
                            paper.year = pub_date.year
                            paper.categories = json.dumps(r.categories)
                            paper.source = "manual"
                            paper.doi = paper.doi or r.doi or None  # the journal version's, when the authors gave one
                            db.session.commit()
                            _log.info("[collection-add] arXiv metadata fetched for %s: %s", arxiv_id, r.title)
                except Exception as e:
                    _log.warning("[collection-add] arXiv fetch failed for %s: %s", arxiv_id, e)
                # _ArxivLockCtx enforces the inter-request gap on release — no manual sleep needed.

            paper = Paper.query.get(paper_id)
            if not paper:
                return

            # --- Step 2: Figure ---
            _set_step(paper_id, "arxiv", "Fetching the figure")
            try:
                figure_url, figure_caption = get_paper_figure(arxiv_id)
                if figure_url:
                    paper.figure_url = figure_url
                    paper.figure_caption = figure_caption
                    db.session.commit()
            except Exception as e:
                _log.warning("[collection-add] figure fetch failed for %s: %s", arxiv_id, e)

            # --- Step 3: Citation count ---
            _set_step(paper_id, "arxiv", "Fetching the citation count")
            try:
                _count = get_citation_count(arxiv_id)
                if _count is not None:
                    paper.citation_count = _count
                    paper.citation_fetched_at = datetime.utcnow()
                    db.session.commit()
            except Exception as e:
                _log.warning("[collection-add] citation fetch failed for %s: %s", arxiv_id, e)

            # --- Step 4: Page count ---
            _set_step(paper_id, "arxiv", "Counting the pages")
            try:
                enrich_page_counts([paper])
                db.session.commit()
            except Exception as e:
                _log.warning("[collection-add] page count failed for %s: %s", arxiv_id, e)

            # --- Step 5: LLM summary + contributions ---
            _set_step(paper_id, "arxiv", "Writing the AI summary and main contributions")
            db.session.refresh(paper)
            if paper.summary:
                _log.info("[collection-add] summary already exists for %s, skipping", arxiv_id)
            else:
                try:
                    paper_content = get_paper_content(arxiv_id) or paper.abstract or ""
                    summary_data, summary_model, summary_elapsed, summary_tokens = generate_summary_with_full_content(
                        title=paper.title,
                        abstract=paper.abstract or "",
                        paper_content=paper_content,
                    )
                    contributions, _, contrib_elapsed, contrib_tokens = extract_main_contributions(
                        title=paper.title,
                        paper_content=paper_content,
                        abstract=paper.abstract or "",
                    )
                    total_elapsed = (summary_elapsed or 0) + (contrib_elapsed or 0)
                    total_tokens = (summary_tokens or 0) + (contrib_tokens or 0)
                    summary_data = summary_data if isinstance(summary_data, dict) else {}
                    summary = PaperSummary(
                        paper_id=paper.id,
                        content_source=summary_source(paper_content, paper.abstract or ""),
                        summary_text=summary_data.get("summary", ""),
                        main_contributions=contributions,
                        model_used=summary_model or LLM_MODELS[0],
                        elapsed_ms=total_elapsed if total_elapsed > 0 else None,
                        tokens_used=total_tokens if total_tokens > 0 else None,
                        introduces_dataset=summary_data.get("introduces_dataset", False),
                        introduces_architecture=summary_data.get("introduces_architecture", False),
                        introduces_method=summary_data.get("introduces_method", False),
                        is_survey=summary_data.get("is_survey", False),
                        is_benchmark=summary_data.get("is_benchmark", False),
                    )
                    db.session.add(summary)
                    db.session.commit()
                    _log.info("[collection-add] summary generated for %s", arxiv_id)
                except Exception as e:
                    _log.warning("[collection-add] summary failed for %s: %s", arxiv_id, e)

            # --- Step 6: Institutions (Scholar batch) ---
            _set_step(paper_id, "arxiv", "Finding the authors' institutions")
            try:
                enrich_institutions_from_scholar([paper])
                db.session.commit()
            except Exception as e:
                _log.warning("[collection-add] institution enrichment failed for %s: %s", arxiv_id, e)

            # Mark complete only if arXiv fetch succeeded (authors populated)
            paper = Paper.query.get(paper_id)
            if paper and paper.authors and paper.authors != "[]":
                paper.enrich_complete = 1
                db.session.commit()
            if paper and paper.citation_fetched_at:
                _log.info("[collection-add] ✓ fully enriched %s", arxiv_id)
            else:
                _log.info("[collection-add] ✓ enriched %s (citations pending — nightly backfill will retry)", arxiv_id)

    _set_step(paper_id, "arxiv", "Waiting to start")  # before the thread, so the page rendered next shows it
    threading.Thread(target=_run, daemon=True).start()


def start_web_paper_add_async(project_id: int, url: str, paper_id: int, app) -> None:
    """Background thread: enrich a web paper added directly to My Collection.

    Pipeline:
    1. Fetch URL → extract title/abstract/detected-arXiv-links.
    2. For each detected arXiv link: fetch its title and check if it matches the page title.
       If match → convert this paper record to an arXiv paper and hand off to
       start_collection_add_async for full enrichment.
    3. No arXiv match → store extracted metadata + generate LLM summary from the abstract.
    """
    import hashlib as _hashlib

    def _run():
        try:
            _steps()
        finally:
            _end_steps(paper_id, "web")

    def _steps():
        with app.app_context(), counted(project_id, "add"):
            paper = Paper.query.get(paper_id)
            if not paper:
                return

            # Step 1: fetch page metadata
            _set_step(paper_id, "web", "Reading the web page")
            info = extract_paper_info_from_url(url)
            page_title = info.get("title") or url
            page_abstract = info.get("abstract") or ""  # reliable only (citation_abstract / #abstract)
            page_authors = info.get("authors") or []
            arxiv_links = info.get("page_arxiv_links") or []
            # full_text: stripped body content — richer than meta abstract, better for LLM
            page_full_text = info.get("full_text") or page_abstract

            # Step 2: check if any in-page arXiv link corresponds to the same paper
            _set_step(paper_id, "web", "Looking for the paper on arXiv")
            matched_arxiv_id = None
            match_source = None
            for candidate_id in arxiv_links[:3]:  # check at most 3 links
                try:
                    with _ArxivLockCtx():
                        results = list(arxiv.Client(delay_seconds=3.1).results(
                            arxiv.Search(id_list=[candidate_id])
                        ))
                        if results and page_title and _titles_match(page_title, results[0].title):
                            matched_arxiv_id = candidate_id
                            match_source = "in-page link"
                            _log.info("[web-add] arXiv match found for %s → %s (%s)",
                                      url, candidate_id, results[0].title)
                except Exception as e:
                    _log.warning("[web-add] arXiv lookup failed for %s: %s", candidate_id, e)
                if matched_arxiv_id:
                    break

            # Step 2b: DOI → arXiv lookup
            # 10.48550/arXiv.{id} DOIs encode the arXiv ID directly — no Scholar call needed.
            # For other DOIs, fall through to Scholar.
            doi = info.get("doi")
            if not matched_arxiv_id and doi:
                direct = re.match(r"10\.48550/arXiv\.(\d{4}\.\d{4,5})", doi)
                if direct:
                    matched_arxiv_id = direct.group(1)
                    match_source = "DOI (direct)"
                    _log.info("[web-add] DOI %s → arXiv %s (direct)", doi, matched_arxiv_id)
                else:
                    try:
                        r = _scholar_call("GET", f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                                          timeout=10.0, params={"fields": "externalIds"})
                        if r.status_code == 200:
                            ext = r.json().get("externalIds") or {}
                            arxiv_via_doi = ext.get("ArXiv")
                            if arxiv_via_doi:
                                matched_arxiv_id = arxiv_via_doi
                                match_source = "DOI (Scholar)"
                                _log.info("[web-add] DOI %s → Scholar → arXiv %s", doi, matched_arxiv_id)
                    except Exception as _doi_e:
                        _log.warning("[web-add] DOI Scholar lookup failed for %s: %s", doi, _doi_e)

            # Step 2c: no explicit link or DOI match — search arXiv by title as a last resort.
            # Many CS/ML papers published elsewhere (IEEE, ACM, Elsevier, ...) also have an
            # arXiv preprint that just isn't linked from the publisher's landing page. Finding
            # it recovers full enrichment (institutions/page-count come from the arXiv PDF,
            # which is on OPEN_ACCESS_PDF_DOMAINS, instead of the paywalled original never
            # being downloadable at all). Same _titles_match safety check as the other paths —
            # a title-only search is the least reliable signal of the three, so this only
            # accepts a near-exact title match, same threshold as everything else here.
            if not matched_arxiv_id and page_title and page_title != url:
                try:
                    with _ArxivLockCtx():
                        results = list(arxiv.Client(delay_seconds=3.1).results(
                            arxiv.Search(
                                query=f'ti:"{page_title}"',
                                max_results=3,
                                sort_by=arxiv.SortCriterion.Relevance,
                            )
                        ))
                    for candidate in results:
                        if _titles_match(page_title, candidate.title):
                            matched_arxiv_id = base_arxiv_id(candidate.entry_id.split("/")[-1])
                            match_source = "title search"
                            _log.info("[web-add] arXiv title-search match for %s → %s (%s)",
                                      url, matched_arxiv_id, candidate.title)
                            break
                except Exception as e:
                    _log.warning("[web-add] arXiv title search failed for %r: %s", page_title, e)

            if matched_arxiv_id:
                # Convert the web paper record to an arXiv paper (if no conflict)
                existing_arxiv = Paper.query.filter_by(arxiv_id=matched_arxiv_id).first()
                if existing_arxiv and existing_arxiv.id != paper_id:
                    # Real arXiv paper already in DB — relink ProjectPaper and remove placeholder
                    pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper_id).first()
                    if pp:
                        # Check if already linked to the real paper
                        dup = ProjectPaper.query.filter_by(
                            project_id=project_id, paper_id=existing_arxiv.id
                        ).first()
                        if not dup:
                            pp.paper_id = existing_arxiv.id
                        else:
                            db.session.delete(pp)
                    db.session.delete(paper)
                    db.session.commit()
                    _log.info("[web-add] relinked to existing arXiv paper %s (via %s)", matched_arxiv_id, match_source)
                    return
                # Update arxiv_id on the placeholder paper then enrich
                paper.arxiv_id = matched_arxiv_id
                db.session.commit()
                start_collection_add_async(project_id, matched_arxiv_id, paper_id, app)
                return

            # Step 3: No arXiv match — store as web paper with extracted metadata
            paper.title = page_title
            paper.abstract = page_abstract
            paper.authors = json.dumps(page_authors) if page_authors else "[]"
            # Prefer the publisher's direct PDF link (citation_pdf_url) over the landing page URL
            paper.pdf_url = info.get("pdf_url") or url
            if info.get("page_count"):
                paper.page_count = info["page_count"]
            paper.source = "web"
            paper.doi = info.get("doi") or doi_from_url(url) or doi_from_url(paper.pdf_url)
            db.session.commit()
            _log.info("[web-add] stored web paper: %s → %s (pdf=%s)", url, page_title, paper.pdf_url)

            # A bioRxiv link: read like a bioRxiv crawl find (full text, figure 1), not as a web page
            if _is_biorxiv(paper.pdf_url or url):
                _set_step(paper_id, "web", "Reading the paper on bioRxiv")
                if upgrade_biorxiv_paper(paper):
                    paper.enrich_complete = 1
                    db.session.commit()
                    return

            # Step 3b: download the actual PDF (if we found a direct link) for institutions +
            _set_step(paper_id, "web", "Reading the PDF")
            # page-count fallback — same idea as the arXiv pipeline's page-1 extraction
            # (_enrich_institutions_llm_fallback), generalized to an arbitrary PDF URL instead
            # of assuming arxiv.org/pdf/{id}.pdf.
            if info.get("pdf_url"):
                _enrich_web_paper_pdf_metadata(paper, info["pdf_url"])
                db.session.commit()
                _log.info("[web-add] PDF metadata: page_count=%s institutions=%s",
                          paper.page_count, paper.institutions)

            # Step 4: generate LLM summary from page content (best-effort)
            _set_step(paper_id, "web", "Writing the AI summary and main contributions")
            # Use full_text (stripped body) so blogs/articles get a real summary, not meta noise.
            if page_full_text:
                try:
                    summary_data, _, _, _ = generate_summary_with_full_content(
                        title=page_title,
                        abstract=page_abstract,
                        paper_content=page_full_text,
                    )
                    contributions, _, _, _ = extract_main_contributions(
                        title=page_title,
                        paper_content=page_full_text,
                        abstract=page_abstract,
                    )
                    summary_data = summary_data if isinstance(summary_data, dict) else {}
                    summary = PaperSummary(
                        paper_id=paper.id,
                        content_source="web_page" if summary_source(page_full_text, page_abstract) == "full_text" else "abstract",
                        summary_text=summary_data.get("summary", ""),
                        main_contributions=contributions,
                        model_used=LLM_MODELS[0],
                        introduces_dataset=summary_data.get("introduces_dataset", False),
                        introduces_architecture=summary_data.get("introduces_architecture", False),
                        introduces_method=summary_data.get("introduces_method", False),
                        is_survey=summary_data.get("is_survey", False),
                        is_benchmark=summary_data.get("is_benchmark", False),
                    )
                    db.session.add(summary)
                    db.session.commit()
                    _log.info("[web-add] summary generated for web paper %s", url)
                except Exception as e:
                    _log.warning("[web-add] summary failed for %s: %s", url, e)

            # Web papers can't be further enriched from arXiv — mark complete
            try:
                paper = Paper.query.get(paper_id)
                if paper:
                    paper.enrich_complete = 1
                    db.session.commit()
            except Exception:
                db.session.rollback()

    _set_step(paper_id, "web", "Waiting to start")  # before the thread, so the page rendered next shows it
    threading.Thread(target=_run, daemon=True).start()


# In-memory: maps crawl_log_id → title of paper currently being screened
# Written by worker threads, read by crawl-status route. Simple dict; GIL protects scalar writes.
_crawl_current_papers: dict[int, str] = {}
# In-memory: maps crawl_log_id → number of parallel workers currently active
_crawl_in_flight: dict[int, int] = {}


class TimeoutError(Exception):
    pass


def _titles_match(t1: str, t2: str) -> bool:
    """True if the two titles share ≥60% of their significant words (case/punct insensitive)."""
    stop = {"a", "an", "the", "of", "in", "on", "for", "with", "and", "or", "to", "is", "are", "that"}
    def _words(t):
        return {w for w in re.sub(r"[^\w\s]", "", t.lower()).split() if w not in stop and len(w) > 2}
    w1, w2 = _words(t1), _words(t2)
    if not w1 or not w2:
        return False
    return len(w1 & w2) / max(len(w1), len(w2)) >= 0.6


def extract_paper_info_from_url(url: str) -> dict:
    """Extract paper metadata from an arbitrary URL for the "add by URL" flow.

    Detects arXiv URLs (and NASA ADS's encoded arXiv IDs) and fetches structured
    metadata via the arXiv API. For any other URL, scrapes citation_* meta tags
    (works for ACL Anthology, NeurIPS, ICLR, PMLR, etc.) and falls back to
    <title>/og:title and a stripped-HTML full-text dump for the LLM to summarize.

    Returns a dict with title/abstract/full_text/arxiv_id/authors/page_arxiv_links/
    doi/pdf_url/page_count keys — any that couldn't be determined are left None
    (or [] for page_arxiv_links). Never raises; failures just leave fields unset.
    """
    result: dict[str, object] = {
        "title": None, "abstract": None, "full_text": None,
        "arxiv_id": None, "authors": None, "page_arxiv_links": [], "doi": None,
        "pdf_url": None, "page_count": None,
    }

    if not _is_safe_external_url(url):
        _log.warning("[extract-url] rejected unsafe URL: %s", url)
        return result

    arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)
    # NASA ADS URL: https://ui.adsabs.harvard.edu/abs/2026arXiv260309600B/abstract
    # Encoded arXiv ID: {year}arXiv{YYMM}{NNNNN}{letter} → {YYMM}.{NNNNN}
    nasa_ads_match = re.search(r"adsabs\.harvard\.edu/abs/\d{4}arXiv(\d{4})(\d{5})\w", url)

    detected_arxiv_id = None
    if arxiv_match:
        detected_arxiv_id = arxiv_match.group(1)
    elif nasa_ads_match:
        detected_arxiv_id = f"{nasa_ads_match.group(1)}.{nasa_ads_match.group(2)}"

    if detected_arxiv_id:
        result["arxiv_id"] = detected_arxiv_id
        with _ArxivLockCtx():
            try:
                arxiv_id = result["arxiv_id"]

                def _fetch():
                    search = arxiv.Search(id_list=[arxiv_id])
                    return list(arxiv.Client(delay_seconds=3.1).results(search))

                with ThreadPoolExecutor(max_workers=1) as executor:
                    papers = executor.submit(_fetch).result(timeout=15)

                if papers:
                    result["title"] = papers[0].title
                    result["abstract"] = papers[0].summary
                    result["authors"] = [str(a) for a in papers[0].authors]
            except Exception:
                pass
            # _ArxivLockCtx enforces the inter-request gap on release — no manual sleep needed.
        return result

    # For PDF URLs: strip the .pdf suffix and fetch the HTML abstract page instead.
    # Academic repositories (ACL Anthology, Semantic Scholar, etc.) host metadata on the HTML page.
    fetch_url = url
    if fetch_url.lower().endswith(".pdf"):
        fetch_url = fetch_url[:-4]

    def _meta_value(html: str, attr_name: str, attr_val: str) -> str | None:
        """Extract content= from any <meta> tag containing name/property=attr_val.
        Works regardless of attribute order and with or without quotes (e.g. ACL Anthology)."""
        for tag in re.findall(r'<meta\b[^>]+>', html, re.IGNORECASE):
            # Match the target attribute with optional quoting
            if re.search(
                r'(?:name|property)\s*=\s*["\']?' + re.escape(attr_val) + r'["\']?(?:\s|>|/)',
                tag, re.IGNORECASE,
            ):
                m = re.search(r'content\s*=\s*["\']([^"\']+)["\']', tag, re.IGNORECASE)
                if m:
                    return m.group(1).strip()
        return None

    def _meta_all(html: str, attr_name: str, attr_val: str) -> list[str]:
        """Like _meta_value but returns all matching content= values."""
        results = []
        for tag in re.findall(r'<meta\b[^>]+>', html, re.IGNORECASE):
            if re.search(
                r'(?:name|property)\s*=\s*["\']?' + re.escape(attr_val) + r'["\']?(?:\s|>|/)',
                tag, re.IGNORECASE,
            ):
                m = re.search(r'content\s*=\s*["\']([^"\']+)["\']', tag, re.IGNORECASE)
                if m:
                    results.append(m.group(1).strip())
        return results

    try:
        with _http_client(public_only=True, timeout=15.0, follow_redirects=True) as http:
            resp = http.get(fetch_url)
        # A 403/404 body is a block or error page (ScienceDirect: Cloudflare), not the paper.
        content = "" if resp.status_code >= 400 else resp.text[:100000]

        # <title> fallback
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", content, re.IGNORECASE)
        if title_match:
            result["title"] = title_match.group(1).strip()

        # og:title
        og_title = _meta_value(content, "property", "og:title")
        if og_title:
            result["title"] = og_title

        # citation_title — preferred (ACL, CVF, NeurIPS, ICLR, etc.)
        cit_title = _meta_value(content, "name", "citation_title")
        if cit_title:
            result["title"] = cit_title

        # citation_abstract — reliable academic abstract
        cit_abs = _meta_value(content, "name", "citation_abstract")
        if cit_abs:
            result["abstract"] = cit_abs

        # <div id="abstract"> or <div class="*abstract*"> — ACL Anthology, many conference sites
        if not result["abstract"]:
            div_abs = re.search(
                r'<div[^>]+(?:id=["\']abstract["\']|class=["\'][^"\']*abstract[^"\']*["\'])[^>]*>'
                r'(.*?)</div>',
                content, re.IGNORECASE | re.DOTALL,
            )
            if div_abs:
                result["abstract"] = re.sub(r'<[^>]+>', '', div_abs.group(1)).strip()

        # citation_doi — used for Scholar DOI lookup
        cit_doi = _meta_value(content, "name", "citation_doi")
        if cit_doi:
            result["doi"] = cit_doi

        # citation_author — list of authors
        cit_authors = _meta_all(content, "name", "citation_author")
        if cit_authors:
            result["authors"] = cit_authors

        # citation_pdf_url — direct PDF link (PMLR, NeurIPS, ACL Anthology, etc.)
        cit_pdf_url = _meta_value(content, "name", "citation_pdf_url")
        if cit_pdf_url:
            result["pdf_url"] = cit_pdf_url

        # citation_firstpage / citation_lastpage — compute page count
        cit_firstpage = _meta_value(content, "name", "citation_firstpage")
        cit_lastpage = _meta_value(content, "name", "citation_lastpage")
        if cit_firstpage and cit_lastpage:
            try:
                result["page_count"] = int(cit_lastpage) - int(cit_firstpage) + 1
            except ValueError:
                pass

        # Scan for arXiv links (blog posts referencing a paper)
        arxiv_links = list(dict.fromkeys(
            re.findall(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)', content, re.IGNORECASE)
        ))
        if arxiv_links:
            result["page_arxiv_links"] = arxiv_links

        # Full-text extraction: strip scripts/styles/nav, then strip tags.
        # Used by the LLM for blog posts and pages without a citation_abstract.
        _body = re.sub(
            r'<(script|style|nav|header|footer|svg)[^>]*>.*?</\1>', '', content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        _body = re.sub(r'<[^>]+>', ' ', _body)
        _body = re.sub(r'\s+', ' ', _body).strip()
        if _body:
            result["full_text"] = _body[:200000]

    except Exception:
        pass

    if not result["abstract"]:
        _fill_from_doi_apis(result, url)

    return result


# An Elsevier article id (PII) in a link: ScienceDirect ".../pii/S0263224125022432", Cell ".../fulltext/
# S2589-0042(25)01350-1", Lancet ".../article/PIIS0140-6736(24)00001-2/fulltext". Crossref knows it without the
# punctuation; 17 characters: S, the journal's ISSN, year, article number, check digit.
_ELSEVIER_PII = re.compile(r"(?:/pii/|/fulltext/|/article/PII)(S[0-9X()\-]{16,24})", re.IGNORECASE)


def _elsevier_pii(url: str) -> Optional[str]:
    m = _ELSEVIER_PII.search(url)
    pii = re.sub(r"[^0-9X]", "", m.group(1)[1:].upper()) if m else ""
    return "S" + pii if len(pii) == 16 else None


def _fill_from_doi_apis(result: dict, url: str) -> None:
    """Fill title/abstract/authors/doi from OpenAlex and Crossref when the landing page gave none
    (ScienceDirect and other publishers block server-side fetches). Metadata APIs only; no
    publisher page or PDF is touched. Never raises; fields it can't fill stay as they were."""
    try:
        with _http_client(public_only=True, timeout=15.0, follow_redirects=True) as http:
            doi = result.get("doi")
            if not doi:
                doi = doi_from_url(url)
            if not doi:
                pii = _elsevier_pii(url)
                if pii:
                    r = http.get("https://api.crossref.org/works", params={
                        "filter": f"alternative-id:{pii}", "select": "DOI", "rows": 1})
                    items = r.json()["message"]["items"] if r.status_code == 200 else []
                    doi = items[0]["DOI"] if items else None
            if not doi:
                return
            result["doi"] = doi

            r = http.get(f"https://api.openalex.org/works/doi:{doi}",
                         params={"select": "title,abstract_inverted_index,authorships"})
            if r.status_code == 200:
                work = r.json()
                result["title"] = work.get("title") or result["title"]
                index = work.get("abstract_inverted_index") or {}
                if index:
                    words = sorted((pos, w) for w, positions in index.items() for pos in positions)
                    result["abstract"] = " ".join(w for _, w in words)
                names = [a["author"]["display_name"] for a in work.get("authorships") or []
                         if (a.get("author") or {}).get("display_name")]
                if names:
                    result["authors"] = names
            if not result["title"] or not result["authors"]:
                r = http.get(f"https://api.crossref.org/works/{doi}")
                if r.status_code == 200:
                    msg = r.json()["message"]
                    if not result["title"] and msg.get("title"):
                        result["title"] = msg["title"][0]
                    names = [f"{a.get('given', '')} {a.get('family', '')}".strip()
                             for a in msg.get("author") or []]
                    if not result["authors"] and names:
                        result["authors"] = names
    except Exception as e:
        _log.warning("[extract-url] DOI API fallback failed for %s: %s", url, e)


def generate_research_interest_from_collection(
    collection_pps: list, existing_text: str = ""
) -> str:
    """Generate or improve the research interest from My Collection papers."""
    selected_pps = _select_reference_papers(collection_pps, limit=REFERENCE_PAPERS_LIMIT)
    papers_text = "\n\n".join(
        f"Paper {i+1}: {pp.paper.title}\nAbstract: {(pp.paper.abstract or '')[:500]}"
        for i, pp in enumerate(selected_pps)
    ) if selected_pps else ""

    if existing_text:
        collection_section = (
            f"\nAlso incorporate these papers from the researcher's collection:\n{papers_text}\n"
            if papers_text else ""
        )
        prompt = prompts.RESEARCH_INTEREST_IMPROVE.format(
            existing_text=existing_text,
            refs_section=collection_section,
        )
    else:
        prompt = prompts.RESEARCH_INTEREST_FROM_COLLECTION.format(papers_text=papers_text)

    try:
        result, _, _, _ = _llm(
            [{"role": "user", "content": prompt}], temperature=0.7, max_tokens=4096
        )
        return result.strip()
    except LLMUnavailableError:
        raise
    except Exception:
        if selected_pps:
            titles = [pp.paper.title for pp in selected_pps[:5]]
            return f"Research related to: {', '.join(titles)}"
        return existing_text


_FIGURE_SKIP = (
    "icon", "logo", "social", "badge", "static/browse", "static/base", "static/v", "favicon",
    "arxiv-logotype", "twitter", "linkedin", "facebook", "mathjax", "header",
    "banner", "button", "arrow", "sprite", "funder", "smileybones",
)


def get_paper_figure(arxiv_id: str) -> tuple[Optional[str], Optional[str]]:
    """Return (figure_url, caption_text) for the first suitable figure in the paper.

    Tries arXiv HTML first (has inline figures + captions), falls back to abs page.
    Returns (None, None) if no figure found.

    Note: we intentionally do not fall back to the PDF. Papers without an arXiv HTML
    version may have figures in their PDF, but PDF extraction requires a heavy dependency
    (e.g. pymupdf) and is significantly slower. Accepted trade-off: those papers get
    no figure rather than incurring that cost.

    Note: for papers where Figure 1 is split into subfigures (a), (b), …, arXiv latexml
    emits each subfigure as a separate <figure> block with no parent wrapper. We return
    only the first subfigure. Stitching multiple images side-by-side would require
    fetching all of them and compositing with PIL — not worth the complexity for now.
    """
    for page_url in [
        f"https://arxiv.org/html/{arxiv_id}",
        f"https://arxiv.org/abs/{arxiv_id}",
    ]:
        try:
            with _ArxivLockCtx():
                with _http_client(timeout=15.0, follow_redirects=True) as http:
                    resp = http.get(page_url)
            if resp.status_code != 200:
                continue
            content = resp.text
            # Use the final URL after redirects as base for resolving relative paths.
            # urljoin requires the base to end with / to treat the last segment as a directory.
            from urllib.parse import urljoin as _urljoin
            import re as _re_fig
            final_url = str(resp.url)
            if not final_url.endswith("/"):
                final_url += "/"
            # arXiv HTML pages use relative src paths that can be either:
            #   - paper-dir-relative: "model.jpg" → resolve against final_url
            #   - html-root-relative: "2605.30322v1/x1.png" → resolve against "https://arxiv.org/html/"
            # Detect the second case by checking if src starts with an arXiv ID pattern.
            _ARXIV_HTML_ROOT = "https://arxiv.org/html/"
            def _resolve_src(src: str) -> str:
                if src.startswith("//"):
                    return "https:" + src
                if src.startswith("/"):
                    return "https://arxiv.org" + src
                if src.startswith("http"):
                    return src
                if _re_fig.match(r'\d{4}\.\d{4,5}(?:v\d+)?/', src):
                    return _ARXIV_HTML_ROOT + src
                return _urljoin(final_url, src)

            # Try to find figures with captions.
            # arXiv HTML (latexml) sometimes places the <img> in a sibling <div> immediately
            # before the <figure> tag rather than inside it. We handle both layouts:
            # (a) img inside <figure>...</figure>, (b) img in the ~1000 chars before <figure>.
            for fig_m in re.finditer(r'<figure[^>]*>(.*?)</figure>', content, re.IGNORECASE | re.DOTALL):
                block = fig_m.group(1)
                img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', block, re.IGNORECASE)
                if not img_match:
                    # Check content immediately preceding this figure tag for an adjacent img
                    pre = content[max(0, fig_m.start() - 1000):fig_m.start()]
                    img_match = re.search(r'(?:.*)<img[^>]+src=["\']([^"\']+)["\']', pre, re.IGNORECASE | re.DOTALL)
                if not img_match:
                    continue
                src = img_match.group(1)
                src_lower = src.lower()
                if any(skip in src_lower for skip in _FIGURE_SKIP):
                    continue
                if not (
                    src_lower.endswith(".png") or src_lower.endswith(".jpg")
                    or src_lower.endswith(".jpeg") or src_lower.endswith(".gif")
                    or src_lower.endswith(".webp") or "fig" in src_lower
                ):
                    continue
                src = _resolve_src(src)
                if not src.startswith("http"):
                    continue

                caption = None
                cap_match = re.search(r'<figcaption[^>]*>(.*?)</figcaption>', block, re.IGNORECASE | re.DOTALL)
                if cap_match:
                    raw_cap = cap_match.group(1)
                    raw_cap = re.sub(r'<math[^>]*>.*?</math>', '', raw_cap, flags=re.IGNORECASE | re.DOTALL)
                    raw_cap = re.sub(r'<[^>]+>', ' ', raw_cap)
                    # Strip residual LaTeX commands that appear outside <math> tags
                    # e.g. \dotsc, \text{voc}, \mathbb{R} in some arXiv HTML outputs
                    raw_cap = re.sub(r'\\[a-zA-Z]+(\{[^}]{0,60}\})*', '', raw_cap)
                    caption = re.sub(r'\s+', ' ', raw_cap).strip()
                    if len(caption) > 1200:
                        caption = caption[:1200].rsplit(' ', 1)[0] + '…'

                return src, caption

            # Fallback: no <figure> blocks, scan all imgs without caption
            img_matches = re.findall(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', content, re.IGNORECASE)
            for src in img_matches:
                src_lower = src.lower()
                if any(skip in src_lower for skip in _FIGURE_SKIP):
                    continue
                if not (
                    src_lower.endswith(".png") or src_lower.endswith(".jpg")
                    or src_lower.endswith(".jpeg") or src_lower.endswith(".gif")
                    or src_lower.endswith(".webp") or "fig" in src_lower
                ):
                    continue
                src = _resolve_src(src)
                if src.startswith("http"):
                    return src, None
        except Exception:
            continue
    return None, None


def _fetch_figure_html(figure_url: str, caption: Optional[str] = None) -> str:
    """Download figure URL, add white padding, return embedded HTML block. Returns '' on failure."""
    try:
        with _biorxiv_gap() if _is_biorxiv(figure_url) else _ArxivLockCtx():  # each site's own rate limit
            with _http_client(timeout=15, follow_redirects=True) as http:
                img_resp = http.get(figure_url)
        if img_resp.status_code != 200 or len(img_resp.content) <= 1000:
            return ""
        img_data = img_resp.content
        content_type = img_resp.headers.get("content-type", "image/png")
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(img_data))
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            w, h = img.size
            pad_x, pad_y = int(w * 0.05), int(h * 0.05)
            bg = Image.new("RGBA", (w + pad_x * 2, h + pad_y * 2), (255, 255, 255, 255))
            bg.paste(img, (pad_x, pad_y), img)
            bg = bg.convert("RGB")
            buf = io.BytesIO()
            bg.save(buf, format="PNG")
            img_data, content_type = buf.getvalue(), "image/png"
        except ImportError:
            pass
        import base64
        data_uri = f"data:{content_type};base64,{base64.b64encode(img_data).decode()}"
        caption_html = (
            f'<p style="margin: 8px 0 0 0; font-size: 11px; color: #6b7280; line-height: 1.5;">'
            f'{html.escape(caption)}</p>'
            if caption else ""
        )
        return (
            '<div style="margin-top: 20px; padding: 15px; background-color: #ffffff;'
            ' border-radius: 8px; border: 1px solid #e5e7eb;">'
            '<p style="color: #6b21a8; margin: 0 0 10px 0; font-weight: 700; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;">📊 FEATURED FIGURE</p>'
            f'<img src="{data_uri}" alt="Paper Figure" style="max-width: 100%; height: auto;'
            ' max-height: 280px; object-fit: contain; display: block; border-radius: 4px;" />'
            + caption_html
            + '</div>'
        )
    except Exception as e:
        print(f"Failed to fetch figure: {e}")
        return ""


def send_notification(
    subject: str,
    message: str,
    notification_type: str = "email",
    figure_url: Optional[str] = None,
    project_emails: Optional[str] = None,
    project_settings_url: Optional[str] = None,
    content_html: Optional[str] = None,
):
    """Send a notification email via the configured backend (NOTIFIER_TYPE: "resend" or "email"/SMTP).

    Builds a branded HTML email (content_html, if given, replaces the default message-based
    body) with an optional embedded figure. Returns True on success, False on any failure
    (missing recipient emails, missing credentials, send error) — never raises.
    """
    notifier = os.environ.get("NOTIFIER_TYPE", "email").lower()
    server_url = os.environ.get("SERVER_URL", "http://localhost:5001")

    if project_settings_url is None:
        project_settings_url = server_url

    global_emails = os.environ.get("NOTIFY_EMAILS", "")
    emails_source = project_emails or global_emails
    if not emails_source:
        _log.warning("[email] send skipped — no recipient emails (project_emails=%r, NOTIFY_EMAILS=%r)", project_emails, global_emails)
        return False
    notify_emails = [e.strip() for e in emails_source.split(",") if e.strip()]

    # Build HTML body (shared by both backends)
    if content_html:
        body_html = content_html
        inline_figure = ""
    else:
        html_lines = []
        for line in message.split("\n"):
            if line.strip():
                html_lines.append(f'<p style="margin: 0 0 10px 0;">{html.escape(line)}</p>')
            else:
                html_lines.append("<br>")
        body_html = "".join(html_lines)
        inline_figure = _fetch_figure_html(figure_url) if figure_url else ""

    html_body = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; color: #374151; background-color: #f9fafb; margin: 0; padding: 20px;">
  <div style="max-width: 580px; margin: 0 auto;">
    <div style="background: linear-gradient(135deg, #7c3aed 0%, #9333ea 100%); color: white; padding: 24px; border-radius: 12px 12px 0 0;">
      <h1 style="margin: 0; font-size: 22px; font-weight: 600;">✨ Related Work Agent</h1>
    </div>
    <div style="background-color: #ffffff; padding: 24px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 12px 12px;">
      <div style="margin-bottom: 20px;">{body_html}</div>
      {inline_figure}
    </div>
  </div>
</body>
</html>"""

    if notifier == "resend":
        api_key = os.environ.get("RESEND_API_KEY")
        from_addr = os.environ.get("RESEND_FROM", "Related Work Agent <notifications@example.com>")
        if not api_key:
            _log.warning("[email] RESEND_API_KEY not set")
            return False
        try:
            resp = httpx.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"from": from_addr, "to": notify_emails, "subject": subject, "html": html_body},
                timeout=30,
            )
            if resp.status_code in (200, 201):
                _log.info("[email] Resend: sent to %s (id=%s)", notify_emails, resp.json().get("id"))
                return True
            _log.error("[email] Resend: HTTP %d — %s", resp.status_code, resp.text[:500])
            return False
        except Exception as e:
            import traceback
            _log.error("[email] Resend send failed: %s — %s\n%s", type(e).__name__, e, traceback.format_exc())
            return False

    if notifier == "email":
        smtp_host = os.environ.get("SMTP_HOST")
        smtp_port = int(os.environ.get("SMTP_PORT", "465"))
        smtp_user = os.environ.get("SMTP_USER")
        smtp_password = os.environ.get("SMTP_PASSWORD")
        missing = [name for v, name in [(smtp_host, "SMTP_HOST"), (smtp_user, "SMTP_USER"), (smtp_password, "SMTP_PASSWORD")] if not v]
        if missing:
            _log.warning("[email] send skipped — missing env vars: %s", missing)
            return False
        _log.info("[email] SMTP: sending to %s via %s:%s", notify_emails, smtp_host, smtp_port)
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            msg = MIMEMultipart("alternative")
            msg["From"] = smtp_user
            msg["To"] = ", ".join(notify_emails)
            msg["Subject"] = subject
            msg.attach(MIMEText(html_body, "html", "utf-8"))
            import socket as _socket
            smtp_ip = _socket.getaddrinfo(smtp_host, smtp_port, _socket.AF_INET)[0][4][0]
            if smtp_port == 465:
                with smtplib.SMTP_SSL(smtp_ip, smtp_port, timeout=30) as server:
                    server.login(smtp_user, smtp_password)
                    server.sendmail(smtp_user, notify_emails, msg.as_string())
            else:
                with smtplib.SMTP(smtp_ip, smtp_port, timeout=30) as server:
                    server.ehlo(); server.starttls(); server.ehlo()
                    server.login(smtp_user, smtp_password)
                    server.sendmail(smtp_user, notify_emails, msg.as_string())
            _log.info("[email] SMTP: sent successfully to %s", notify_emails)
            return True
        except Exception as e:
            import traceback
            _log.error("[email] SMTP send failed: %s — %s\n%s", type(e).__name__, e, traceback.format_exc())
            return False

    _log.warning("[email] send_notification fell through — notifier=%r", notifier)
    return False


def notify_important_paper(
    paper, project_name: str, project_slug: str, figure_url: Optional[str] = None,
    figure_caption: Optional[str] = None,
    project_emails: Optional[str] = None, notes: Optional[str] = None
):
    """Build and send the "paper flagged as important" email (single-paper card + notes).

    Returns the bool from send_notification (True = sent).
    """
    from zoneinfo import ZoneInfo
    berlin = ZoneInfo("Europe/Berlin")
    now_str = datetime.now(berlin).strftime("%H:%M on %A, %d.%m.%Y")

    server_url = os.environ.get("SERVER_URL", "http://localhost:5001")
    dashboard_url = f"{server_url}/projects/{project_slug}"

    authors_list = paper.authors_list
    authors_str = html.escape(
        ", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else "")
        if authors_list else "Unknown"
    )
    year = getattr(paper, "year", None)
    year_span = f' <span style="font-weight:400;color:#6b7280;font-size:15px;">({year})</span>' if year else ""
    citations_html = ""
    meta_parts = []
    if paper.citation_count:
        meta_parts.append(f"{paper.citation_count:,} citations")
    if paper.page_count:
        meta_parts.append(f"{paper.page_count} pages")
    if meta_parts:
        citations_html = f'<p style="margin:3px 0 0 0;font-size:12px;color:#6b7280;">{" · ".join(meta_parts)}</p>'

    institutions_list = paper.institutions_list
    institutions_html = (
        f'<p style="margin:3px 0 0 0;font-size:12px;color:#6b7280;">'
        f'{html.escape(", ".join(institutions_list[:3]))}</p>'
    ) if institutions_list else ""

    summary = getattr(paper, "summary", None)
    badge_labels = []
    if summary:
        if getattr(summary, "introduces_dataset", False): badge_labels.append("Dataset")
        if getattr(summary, "introduces_architecture", False): badge_labels.append("Architecture")
        if getattr(summary, "introduces_method", False): badge_labels.append("Method")
        if getattr(summary, "is_survey", False): badge_labels.append("Survey")
        if getattr(summary, "is_benchmark", False): badge_labels.append("Benchmark")
    badges_html = ""
    if badge_labels:
        spans = "".join(
            f'<span style="display:inline-block;padding:2px 8px;background-color:#ede9fe;color:#6d28d9;'
            f'font-size:11px;border-radius:4px;font-weight:600;margin-right:4px;border:1px solid #c4b5fd;">{b}</span>'
            for b in badge_labels
        )
        badges_html = f'<div style="margin:8px 0 0 0;">{spans}</div>'

    summary_text = getattr(summary, "summary_text", None) if summary else None
    summary_html = ""
    if summary_text and isinstance(summary_text, str) and summary_text.strip():
        summary_html = (
            '<div style="margin-top:14px;border-top:1px solid #ddd6fe;padding-top:12px;">'
            '<p style="margin:0 0 6px 0;font-size:11px;font-weight:700;color:#7c3aed;'
            'text-transform:uppercase;letter-spacing:0.05em;">AI Summary</p>'
            f'<p style="margin:0;font-size:13px;color:#374151;line-height:1.6;">{html.escape(summary_text.strip())}</p>'
            '</div>'
        )

    mc = getattr(summary, "main_contributions", None) if summary else None
    contributions_html = ""
    if mc and isinstance(mc, str):
        items = "".join(
            f'<li style="margin-bottom:6px;color:#374151;font-size:13px;">'
            f'{html.escape(line.lstrip("- ").strip())}</li>'
            for line in mc.split("\n") if line.strip()
        )
        contributions_html = (
            '<div style="margin-top:14px;border-top:1px solid #ddd6fe;padding-top:12px;">'
            '<p style="margin:0 0 8px 0;font-size:11px;font-weight:700;color:#7c3aed;'
            'text-transform:uppercase;letter-spacing:0.05em;">Main Findings</p>'
            f'<ul style="margin:0;padding-left:18px;line-height:1.6;">{items}</ul>'
            '</div>'
        )

    notes_html = ""
    if notes and notes.strip():
        notes_html = (
            '<div style="margin-top:14px;border-top:1px solid #ddd6fe;padding-top:12px;">'
            '<p style="margin:0 0 6px 0;font-size:11px;font-weight:700;color:#7c3aed;'
            'text-transform:uppercase;letter-spacing:0.05em;">Personal Notes</p>'
            f'<p style="margin:0;font-size:13px;color:#374151;line-height:1.6;">{html.escape(notes.strip())}</p>'
            '</div>'
        )

    paper_card = (
        '<div style="background-color:#f5f3ff;border:1px solid #ddd6fe;border-radius:10px;'
        'padding:18px;margin:4px 0 16px 0;">'
        '<p style="font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:#7c3aed;'
        'margin:0 0 12px 0;font-weight:700;">📄 Flagged Paper</p>'
        f'<h2 style="margin:0 0 6px 0;font-size:17px;font-weight:700;color:#111827;line-height:1.35;">'
        f'{html.escape(paper.title)}{year_span}</h2>'
        f'<p style="margin:0;font-size:13px;color:#4b5563;">{authors_str}</p>'
        + citations_html + institutions_html + badges_html
        + (
            f'<p style="margin:10px 0 0 0;font-size:12px;">'
            f'<a href="{html.escape(paper.landing_url)}" style="color:#7c3aed;text-decoration:none;">'
            f'{html.escape(paper.landing_url.removeprefix("https://"))}</a></p>'
            if paper.landing_url else ""
        )
        + notes_html + summary_html + contributions_html
        + '</div>'
    )

    figure_html = _fetch_figure_html(figure_url, caption=figure_caption) if figure_url else ""

    view_link = (
        f'<p style="text-align:center;margin:20px 0;">'
        f'<a href="{dashboard_url}" style="color:#7c3aed;font-size:13px;font-weight:600;'
        f'text-decoration:none;">View in project →</a></p>'
    )

    project_link = (
        f'<a href="{dashboard_url}" style="color:#7c3aed;text-decoration:none;font-weight:700;">'
        f'{html.escape(project_name)}</a>'
    )
    content_html = (
        f'<p style="font-size:13px;color:#6b7280;margin:0 0 16px 0;">'
        f'Flagged at {now_str} · in project {project_link}</p>'
        + paper_card + figure_html + view_link
    )

    subject = f"📄 Paper flagged: {paper.title[:60]}{'…' if len(paper.title) > 60 else ''}"
    return send_notification(subject, "", content_html=content_html, figure_url=None, project_emails=project_emails, project_settings_url=dashboard_url)


def notify_crawl_complete(
    project_name: str,
    project_slug: str,
    papers_found: int,
    papers_added: int,
    top_papers: list = None,
    project_id: int = None,
    keywords_used: list = None,
    papers_screened: int = None,
    paper_limit: int = None,
    source_counts: Optional[dict] = None,
    date_from=None,
    date_to=None,
    keywords_auto_generated: bool = False,
    warning: Optional[str] = None,
):
    """Build and send the crawl-complete summary email (stats table + featured paper card).

    Returns the bool from send_notification (True = sent).
    """
    server_url = os.environ.get("SERVER_URL", "http://localhost:5001")
    dashboard_url = f"{server_url}/projects/{project_slug}"
    project_settings_url = dashboard_url

    subject = f"✨ Crawl complete: {papers_added} new paper{'s' if papers_added != 1 else ''} for {project_name}"

    screened = papers_screened if papers_screened is not None else papers_found
    screened_note = ""
    if paper_limit and paper_limit > 0 and papers_found > paper_limit:
        screened_note = f" (limit: {paper_limit} of {papers_found} found)"

    # Source breakdown rows: a paper found several ways counts in each of its rows, the total counts it once
    source_rows = ""
    if source_counts:
        for via, label in _email_source_rows(source_counts):
            c = source_counts.get(via) or {"found": 0, "added": 0}
            source_rows += (
                f'<tr style="border-bottom: 1px solid #f3f4f6;">'
                f'<td style="padding: 7px 0; color: #6b7280;">{label}</td>'
                f'<td style="padding: 7px 0; text-align: right; font-weight: 600; color: #111827; white-space: nowrap;">{c["found"]}'
                f'<span style="font-weight: 400; color: #6b7280;"> ({c["added"]} added)</span></td>'
                f'</tr>'
            )

    stats_html = (
        '<table cellpadding="0" cellspacing="0" style="width: 100%; border-collapse: collapse;'
        ' margin: 12px 0 20px 0; font-size: 14px;">'
        '<tbody>'
        + source_rows
        + f'<tr style="border-bottom: 2px solid #e5e7eb;">'
        f'<td style="padding: 9px 0; color: #374151; font-weight: 600;">Total found'
        + ('<div style="font-weight: 400; font-size: 12px; color: #9ca3af;">A paper found several ways counts in'
           ' each row, so the rows add up to more than the total, but nothing is screened twice.</div>'
           if source_counts else '')
        + '</td>'
        f'<td style="padding: 9px 0; text-align: right; font-weight: 700; color: #111827;">{papers_found}</td>'
        f'</tr>'
        f'<tr style="border-bottom: 1px solid #f3f4f6;">'
        f'<td style="padding: 7px 0; color: #6b7280;">Screened with AI{html.escape(screened_note)}</td>'
        f'<td style="padding: 7px 0; text-align: right; font-weight: 600; color: #111827;">{screened}</td>'
        f'</tr>'
        f'<tr style="border-bottom: 1px solid #f3f4f6;">'
        f'<td style="padding: 7px 0; color: #6b7280;">Added to project</td>'
        f'<td style="padding: 7px 0; text-align: right; font-weight: 700; color: #7c3aed;">{papers_added}</td>'
        f'</tr>'
        f'<tr>'
        f'<td style="padding: 7px 0; color: #6b7280;">Skipped (irrelevant)</td>'
        f'<td style="padding: 7px 0; text-align: right; font-weight: 600; color: #111827;">{screened - papers_added}</td>'
        f'</tr>'
        '</tbody></table>'
    )

    # Featured paper card
    featured_html = ""
    figure_url: Optional[str] = None
    figure_caption: Optional[str] = None
    if top_papers and len(top_papers) > 0:
        top = top_papers[0]
        if isinstance(top, tuple):
            top = top[0]

        raw_fig = getattr(top, "figure_url", None)
        if isinstance(raw_fig, str):
            figure_url = raw_fig
        raw_cap = getattr(top, "figure_caption", None)
        if isinstance(raw_cap, str) and raw_cap.strip():
            figure_caption = raw_cap.strip()

        title = html.escape(getattr(top, "title", "Unknown Title"))
        year = getattr(top, "year", None)
        year_span = f' <span style="font-weight: 400; color: #6b7280; font-size: 15px;">({year})</span>' if year else ""
        citation_count = getattr(top, "citation_count", None) or 0
        arxiv_id = getattr(top, "arxiv_id", "")
        # A non-arXiv paper (bioRxiv) links to its own page
        paper_link = getattr(top, "landing_url", None) or f"https://arxiv.org/abs/{arxiv_id}"

        authors_list = getattr(top, "authors_list", [])
        if not authors_list:
            authors_raw = getattr(top, "authors", None)
            if authors_raw:
                try:
                    authors_list = json.loads(authors_raw)
                except Exception:
                    authors_list = [str(authors_raw)]
        authors_str = html.escape(", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else ""))

        page_count = getattr(top, "page_count", None) or 0
        meta_parts = []
        if citation_count:
            meta_parts.append(f"{citation_count:,} citations")
        if page_count:
            meta_parts.append(f"{page_count} pages")
        citations_html = (
            f'<p style="margin: 3px 0 0 0; font-size: 12px; color: #6b7280;">{" · ".join(meta_parts)}</p>'
            if meta_parts else ""
        )

        institutions_list = getattr(top, "institutions_list", [])
        institutions_html = (
            f'<p style="margin: 3px 0 0 0; font-size: 12px; color: #6b7280;">'
            f'{html.escape(", ".join(institutions_list[:3]))}</p>'
            if institutions_list else ""
        )

        summary = getattr(top, "summary", None)
        badge_labels = []
        if summary:
            if getattr(summary, "introduces_dataset", False): badge_labels.append("Dataset")
            if getattr(summary, "introduces_architecture", False): badge_labels.append("Architecture")
            if getattr(summary, "introduces_method", False): badge_labels.append("Method")
            if getattr(summary, "is_survey", False): badge_labels.append("Survey")
            if getattr(summary, "is_benchmark", False): badge_labels.append("Benchmark")
        badges_html = ""
        if badge_labels:
            spans = "".join(
                f'<span style="display: inline-block; padding: 2px 8px; background-color: #ede9fe;'
                f' color: #6d28d9; font-size: 11px; border-radius: 9999px; font-weight: 600; margin-right: 4px;">{b}</span>'
                for b in badge_labels
            )
            badges_html = f'<div style="margin: 8px 0 0 0;">{spans}</div>'

        mc = getattr(summary, "main_contributions", None) if summary else None
        main_contributions_html = ""
        if mc and isinstance(mc, str):
            items = "".join(
                f'<li style="margin-bottom: 6px; color: #374151; font-size: 13px;">'
                f'{html.escape(line.lstrip("- ").strip())}</li>'
                for line in mc.split("\n") if line.strip()
            )
            main_contributions_html = (
                '<div style="margin-top: 14px; border-top: 1px solid #ddd6fe; padding-top: 12px;">'
                '<p style="margin: 0 0 8px 0; font-size: 11px; font-weight: 700; color: #7c3aed;'
                ' text-transform: uppercase; letter-spacing: 0.05em;">Main Findings</p>'
                f'<ul style="margin: 0; padding-left: 18px; line-height: 1.6;">{items}</ul>'
                '</div>'
            )

        featured_html = (
            '<div style="background-color: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 10px;'
            ' padding: 18px; margin: 4px 0 16px 0;">'
            '<p style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #7c3aed;'
            ' margin: 0 0 12px 0; font-weight: 700;">✨ Featured Paper</p>'
            f'<h2 style="margin: 0 0 6px 0; font-size: 17px; font-weight: 700; color: #111827; line-height: 1.35;">'
            f'{title}{year_span}</h2>'
            f'<p style="margin: 0; font-size: 13px; color: #4b5563;">{authors_str}</p>'
            + citations_html
            + institutions_html
            + badges_html
            + f'<p style="margin: 10px 0 0 0; font-size: 12px;">'
            f'<a href="{html.escape(paper_link)}" style="color: #7c3aed; text-decoration: none;">'
            f'{html.escape(paper_link.removeprefix("https://"))}</a></p>'
            + main_contributions_html
            + '</div>'
        )

    figure_html = _fetch_figure_html(figure_url, caption=figure_caption) if figure_url else ""

    view_link = (
        f'<div style="margin-top: 20px; text-align: center;">'
        f'<a href="{dashboard_url}" style="color: #7c3aed; font-size: 14px; font-weight: 600;'
        f' text-decoration: none;">View and tag papers &#8594;</a>'
        f'</div>'
    )

    if date_from and date_to:
        date_range_str = f"{date_from.strftime('%b %d, %Y')} – {date_to.strftime('%b %d, %Y')}"
        date_range_html = (
            f'<p style="margin: 2px 0 10px 0; font-size: 12px; color: #6b7280;">'
            f'{date_range_str}</p>'
        )
    else:
        date_range_html = ""

    auto_kw_html = ""
    if keywords_auto_generated and keywords_used:
        kw_list = ", ".join(html.escape(k) for k in keywords_used)
        settings_url = f"{dashboard_url}#keywords"
        auto_kw_html = (
            f'<div style="margin: 16px 0; padding: 10px 14px; background-color: #faf5ff;'
            f' border: 1px solid #e9d5ff; border-radius: 8px; font-size: 13px; color: #6b21a8;">'
            f'<strong>Keywords auto-generated</strong> — no keywords were saved, so they were generated from'
            f' your research interest for this crawl: <em>{kw_list}</em>.'
            f' <a href="{settings_url}" style="color: #7c3aed; font-weight: 600;">Review or edit them &rarr;</a>'
            f'</div>'
        )

    warning_html = (
        f'<div style="margin: 12px 0; padding: 10px 14px; background-color: #fffbeb;'
        f' border: 1px solid #fde68a; border-radius: 8px; font-size: 13px; color: #92400e;">'
        f'<strong>Incomplete search:</strong> {html.escape(warning)}</div>'
        if warning else ""
    )

    content_html = (
        f'<p style="margin: 0 0 4px 0; font-size: 15px; color: #374151;">'
        f'Crawl finished for <strong>{html.escape(project_name)}</strong></p>'
        + date_range_html
        + warning_html
        + stats_html
        + featured_html
        + figure_html
        + auto_kw_html
        + view_link
    )

    # Plain-text fallback
    date_range_text = (
        f'{date_from.strftime("%b %d, %Y")} – {date_to.strftime("%b %d, %Y")}\n'
        if date_from and date_to else ""
    )
    message = (
        f'Crawl finished for "{project_name}"\n'
        + date_range_text
        + ''.join(f'\n{label}: {(source_counts.get(via) or {}).get("found", 0)} '
                  f'({(source_counts.get(via) or {}).get("added", 0)} added)'
                  for via, label in _email_source_rows(source_counts) if source_counts)
        + f'\nTotal found: {papers_found}\n'
        f'Added: {papers_added} · Skipped: {screened - papers_added}\n\n'
        f'View and tag papers: {dashboard_url}'
        + (f'\n\nIncomplete search: {warning}' if warning else '')
    )

    project_emails = None
    if project_id:
        project = Project.query.get(project_id)
        if project and project.notification_emails:
            project_emails = project.notification_emails

    return send_notification(
        subject,
        message,
        figure_url=None,
        project_emails=project_emails,
        project_settings_url=project_settings_url,
        content_html=content_html,
    )


_COLLECTION_TAG_PRIORITY = {"important": 0, "to_discuss": 1, "to_read": 2, "related": 3}


def _collection_tag_tier(pp) -> int:
    tags = pp.tags_list
    for tag in ("important", "to_discuss", "to_read"):
        if tag in tags:
            return _COLLECTION_TAG_PRIORITY[tag]
    return _COLLECTION_TAG_PRIORITY["related"]


def _select_reference_papers(pps: list, limit: int) -> list:
    """Pick up to `limit` My Collection papers to use as LLM reference/calibration context.

    Priority: important > to_discuss > to_read > related — fill from the top tier down.
    Within a tier that has more papers than the remaining slots, alternates newest/oldest
    by collected_at (newest, oldest, 2nd-newest, 2nd-oldest, ...) instead of taking
    whichever the DB happens to return first, so the selection stays a representative
    spread rather than skewing toward one end of the collection's history.
    """
    valid = [pp for pp in pps if pp.paper and not pp.trashed_at and pp.manual_tag]
    tiers: dict[int, list] = {}
    for pp in valid:
        tiers.setdefault(_collection_tag_tier(pp), []).append(pp)

    selected = []
    for tier_key in sorted(tiers):
        remaining = limit - len(selected)
        if remaining <= 0:
            break
        tier_pps = sorted(tiers[tier_key], key=lambda pp: pp.collected_at or datetime.min)
        if len(tier_pps) <= remaining:
            selected.extend(tier_pps)
            continue
        lo, hi = 0, len(tier_pps) - 1
        take_newest = True
        while len(selected) < limit and lo <= hi:
            if take_newest:
                selected.append(tier_pps[hi])
                hi -= 1
            else:
                selected.append(tier_pps[lo])
                lo += 1
            take_newest = not take_newest
    return selected


def _collection_context_text(collection_pps: list) -> str:
    """Build a context string from My Collection (ProjectPaper objects) for keyword/RI generation.

    Uses stored abstracts — no arXiv fetches needed. See _select_reference_papers for selection order.
    """
    entries = []
    for pp in _select_reference_papers(collection_pps, limit=REFERENCE_PAPERS_LIMIT):
        abstract = (pp.paper.abstract or "")[:500]
        entries.append(f"- {pp.paper.title}" + (f": {abstract}" if abstract else ""))
    return "\n".join(entries) if entries else "(none yet — add papers to My Collection to improve keyword extraction)"


def extract_keywords_with_usage(
    research_interest: str, collection_papers: list
) -> tuple[list[str], dict]:
    """Extract search keywords from research interest + My Collection.

    collection_papers: list of ProjectPaper ORM objects (My Collection papers).
    """
    collection_context = _collection_context_text(collection_papers)

    prompt = prompts.KEYWORD_EXTRACTION.format(
        research_interest=research_interest or "(not specified)",
        collection_context=collection_context,
    )

    try:
        result, model, elapsed_ms, usage = _llm_json(
            [{"role": "user", "content": prompt}], temperature=0.3, max_tokens=16384
        )
        _log.info("[keywords] LLM returned type=%s value=%r model=%s elapsed=%sms", type(result).__name__, result, model, elapsed_ms)
        if isinstance(result, list):
            keywords = [str(k) for k in result[:6]]
            _log.info("[keywords] Extracted %d keywords: %s", len(keywords), keywords)
            return keywords, usage
        if isinstance(result, dict) and "keywords" in result:
            keywords = result["keywords"][:6]
            _log.info("[keywords] Extracted %d keywords from dict: %s", len(keywords), keywords)
            return keywords, usage
        _log.warning("[keywords] LLM returned unexpected format, returning empty. result=%r", result)
        return [], usage
    except LLMUnavailableError:
        raise
    except Exception as e:
        _log.warning("[keywords] LLM call failed (%s)", e)
        return [], {"total_tokens": 0}


def extract_keywords(research_interest: str, collection_papers: list) -> list[str]:
    keywords, _ = extract_keywords_with_usage(research_interest, collection_papers)
    return keywords


def _keyword_clause(keyword: str) -> str:
    return f'abs:"{keyword}"' if " " in keyword else keyword


def _keywords_label(keywords: list[str]) -> str:
    return keywords[0] if len(keywords) == 1 else f"{len(keywords)} keywords"


def _fetch_arxiv_window(keywords: list[str], d_from: date, d_to: date, max_results: int) -> tuple[list[dict], bool, bool, bool]:
    """One arXiv query for all `keywords` (OR-ed) restricted server-side to submittedDate in [d_from, d_to].

    One query per project instead of one per keyword: arXiv answers ~1 in 4 of our nightly queries with
    a 429 even at one request per 5 s, so fewer requests means fewer failures. Same papers as the union
    of per-keyword queries (checked live on 3 projects, 2026-10-14).

    Returns (papers, hit_cap, timed_out, errored). `hit_cap` means arXiv returned max_results papers, so
    older ones in this window may have been cut off. `errored` means the query failed outright
    (429/503/other) after retries — "no answer", as opposed to "answered, nothing matched".
    """
    label = _keywords_label(keywords)
    base = " OR ".join(_keyword_clause(k) for k in keywords)
    query = f"({base}) AND submittedDate:[{d_from:%Y%m%d}0000 TO {d_to:%Y%m%d}2359]"
    papers: list[dict] = []
    timed_out = False
    errored = False
    with _ArxivLockCtx():
        for attempt in range(3):
            papers = []
            timed_out = False
            errored = False
            try:
                start_time = datetime.now()
                search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
                client = arxiv.Client(num_retries=1, delay_seconds=3.1)
                for result in client.results(search):
                    if (datetime.now() - start_time).total_seconds() > 45:
                        timed_out = True
                        break
                    pub_date = result.published.date()
                    papers.append(
                        {
                            "arxiv_id": base_arxiv_id(result.entry_id.split("/")[-1]),
                            "title": result.title,
                            "authors": [str(a) for a in result.authors],
                            "abstract": result.summary,
                            "pdf_url": re.sub(r"v\d+$", "", result.pdf_url or ""),
                            "published_date": pub_date,
                            "year": pub_date.year,
                            "categories": [cat for cat in result.categories],
                            "doi": result.doi or None,  # the journal version's, when the authors gave one
                        }
                    )
                break  # success — don't retry
            except arxiv.HTTPError as e:
                # 503 is arXiv's other "slow down / busy" answer — wait it out like a 429
                if getattr(e, "status", 0) in (429, 503) and attempt < 2:
                    wait = 120 * (attempt + 1)  # 120s, then 240s
                    _log.warning("[arXiv] %s for '%s', waiting %ds (attempt %d/3)", e.status, label, wait, attempt + 1)
                    time.sleep(wait)
                else:
                    _log.warning("[arXiv] HTTP error for '%s': %s", label, e)
                    errored = True
                    break
            except Exception as e:
                _log.warning("[arXiv] Error searching '%s': %s", label, e)
                errored = True
                break
        # _ArxivLockCtx enforces the inter-request gap on release — no manual sleep needed.
    return papers, len(papers) >= max_results, timed_out, errored


def _collect_arxiv(
    keywords: list[str], d_from: date, d_to: date, max_results: int, warnings: list[str], failed: Optional[list[str]] = None
) -> list[dict]:
    """All papers matching any of `keywords` in [d_from, d_to], newest first.

    arXiv caps each query at `max_results`. When a query hits the cap, the oldest paper we got
    tells us how far back it reached: the next window is [d_from, that date] (the boundary day is
    re-queried since it may have been cut mid-day; duplicates are dropped by id).
    A query that errors out (429/503) adds all `keywords` to `failed`, and a message to `warnings`.
    """
    keyword = _keywords_label(keywords)
    papers: list[dict] = []
    seen: set[str] = set()
    cur_to = d_to
    while cur_to >= d_from:
        batch, hit_cap, timed_out, errored = _fetch_arxiv_window(keywords, d_from, cur_to, max_results)
        for p in batch:
            if p["arxiv_id"] not in seen:
                seen.add(p["arxiv_id"])
                papers.append(p)
        if errored:
            msg = f"'{keyword}': arXiv search failed for {d_from}..{cur_to} — no results"
            _log.warning("[arXiv] %s", msg)
            warnings.append(msg)
            if failed is not None:
                failed.extend(keywords)
            break
        if timed_out:
            msg = f"'{keyword}': arXiv search timed out for {d_from}..{cur_to} — results may be incomplete"
            _log.warning("[arXiv] %s", msg)
            warnings.append(msg)
        if not hit_cap:
            break
        oldest = min(p["published_date"] for p in batch)
        if oldest >= cur_to:  # the cap was filled by a single day — can't page further back by date
            msg = f"'{keyword}': more than {max_results} papers on {cur_to} — some were cut off"
            _log.warning("[arXiv] %s", msg)
            warnings.append(msg)
            cur_to = cur_to - timedelta(days=1)
        else:
            _log.info("[arXiv] '%s' hit %d-result cap, oldest %s — continuing with %s..%s", keyword, max_results, oldest, d_from, oldest)
            cur_to = oldest
    return papers


def _apply_search_outcome(crawl_log, failed_keywords: list[str], keywords: list[str]) -> None:
    """Set the final status. Every arXiv query erroring out (429/503) means the window was never
    searched — that's an "error", not a quiet day. Some failing is a success with a warning message."""
    if failed_keywords and len(failed_keywords) >= len(keywords):
        crawl_log.status = "error"
        crawl_log.error_message = f"arXiv search failed for all {len(keywords)} keywords (rate limited or unavailable) — this date range was not searched"
    else:
        crawl_log.status = "success"
        if failed_keywords:
            crawl_log.error_message = (
                f"arXiv search failed for {len(failed_keywords)}/{len(keywords)} keywords: {', '.join(failed_keywords)}"
            )


def search_arxiv_with_full_papers(
    keywords: list[str],
    date_from: date,
    date_to: date,
    max_results: int = 200,
    warnings: Optional[list[str]] = None,
    failed_keywords: Optional[list[str]] = None,
) -> list[dict]:
    """Search arXiv for all keywords at once (one OR query, see _fetch_arxiv_window) and return
    deduplicated papers in date range.
    Returns list of paper dicts without LLM processing.
    Used for the search phase to show user how many papers were found.

    `max_results` is the per-query cap; when it is hit, the search continues from the oldest paper's date
    back to `date_from`, so the full range is covered. Anything still truncated (single day over
    the cap, or a 45s timeout) is appended to `warnings` if given; if the query errored out
    entirely, all keywords are appended to `failed_keywords` if given.
    """
    if warnings is None:
        warnings = []
    if not keywords:
        return []
    papers = []
    seen_ids = set()
    for paper in _collect_arxiv(keywords, date_from, date_to, max_results, warnings, failed_keywords):
        if paper["arxiv_id"] in seen_ids:
            continue
        seen_ids.add(paper["arxiv_id"])
        papers.append(paper)
    return papers


def enrich_institutions_from_scholar(papers: list) -> None:
    """Batch-fetch author affiliations from Semantic Scholar and store in paper.institutions.

    `papers` is a list of Paper ORM objects. Updates each paper.institutions in-place (caller
    must db.session.commit() afterwards). Skips papers that already have institutions set.
    Rate limit: 1 req/sec without API key; batch endpoint handles up to 500 IDs per call.
    """
    to_enrich = [p for p in papers if not p.institutions]
    if not to_enrich:
        return

    ids = [f"ArXiv:{base_arxiv_id(p.arxiv_id)}" for p in to_enrich]
    try:
        r = _scholar_call("POST", "https://api.semanticscholar.org/graph/v1/paper/batch",
                          params={"fields": "authors.affiliations"}, json={"ids": ids})
        if r.status_code == 200:
            id_map = {base_arxiv_id(p.arxiv_id): p for p in to_enrich}
            for item in r.json():
                if not item:
                    continue
                ext = (item.get("externalIds") or {})
                aid = ext.get("ArXiv") or ""
                paper = id_map.get(base_arxiv_id(aid))
                if not paper:
                    continue
                affs: set[str] = set()
                for author in (item.get("authors") or []):
                    for aff in (author.get("affiliations") or []):
                        if aff:
                            affs.add(aff)
                if affs:
                    paper.institutions = json.dumps(sorted(affs))
        else:
            _log.warning("[Scholar] Affiliation batch returned %d: %s", r.status_code, r.text[:200])
    except Exception as e:
        _log.warning("[Scholar] Affiliation batch failed: %s", e)

    # Always fall back to LLM for papers still missing institutions
    still_missing = [p for p in to_enrich if not p.institutions]
    _enrich_institutions_llm_fallback(still_missing)


def _enrich_institutions_llm_fallback(papers: list) -> None:
    """Per-paper LLM institution extraction for papers the Scholar batch call missed.

    Downloads each paper's PDF and extracts page-1 text (title page has affiliations),
    falling back to the abstract if the PDF fetch fails. Updates paper.institutions
    in-place; caller must db.session.commit() afterwards.
    """
    if not papers:
        return

    def _get_page1_text(arxiv_id: str) -> str:
        """Download PDF and extract text from page 1 (title page has affiliations)."""
        try:
            import io
            from pypdf import PdfReader
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            with _ArxivLockCtx():
                with _http_client(timeout=httpx.Timeout(30.0), follow_redirects=True) as http:
                    r = http.get(url)
            if r.status_code == 200 and r.content:
                reader = PdfReader(io.BytesIO(r.content), strict=False)
                if reader.pages:
                    return reader.pages[0].extract_text() or ""
        except Exception as e:
            _log.warning("[institutions] PDF fetch failed for %s: %s", arxiv_id, e)
        return ""

    for paper in papers:
        page1 = _get_page1_text(paper.arxiv_id) if paper.arxiv_id else ""
        text_for_llm = page1[:5000] if page1 else (paper.abstract or "")[:600]
        source_label = "first page of the PDF" if page1 else "abstract"
        prompt = prompts.INSTITUTION_EXTRACTION.format(
            source_label=source_label,
            title=paper.title,
            text=text_for_llm,
        )
        try:
            result, _, _, _ = _llm_json(
                [{"role": "user", "content": prompt}], temperature=0, max_tokens=4096
            )
            if isinstance(result, list):
                insts = [str(i).strip() for i in result if i and str(i).strip()]
                if insts:
                    paper.institutions = json.dumps(insts)
                    _log.info("[institutions] LLM extracted %d institutions for %s: %s", len(insts), paper.arxiv_id, insts)
        except Exception as e:
            _log.warning("[institutions] LLM fallback failed for %s: %s", paper.arxiv_id, e)


def _enrich_web_paper_pdf_metadata(paper, pdf_url: str) -> None:
    """Download pdf_url once; fill paper.page_count (if missing) and paper.institutions
    (via LLM extraction from page 1 text) — the same approach as the arXiv pipeline's
    _enrich_institutions_llm_fallback, generalized to an arbitrary PDF URL instead of
    assuming arxiv.org/pdf/{id}.pdf. Updates paper in-place; caller must commit.

    Only actually downloads the PDF for domains in OPEN_ACCESS_PDF_DOMAINS — paper.pdf_url
    itself (a landing-page-resolved link, for the user's own use) is set regardless by the
    caller; this function is specifically the server-side bulk-download step, which is what
    the allowlist restricts. No-ops (leaves page_count/institutions unset) for anything else.
    """
    if not _is_open_access_pdf_domain(pdf_url):
        _log.info("[web-enrich] skipping PDF download for non-open-access domain: %s", pdf_url)
        return
    try:
        import io
        from pypdf import PdfReader
        with _http_client(public_only=True, timeout=httpx.Timeout(30.0), follow_redirects=True) as http:
            pdf_resp = http.get(pdf_url)
        if pdf_resp.status_code == 200 and pdf_resp.content:
            reader = PdfReader(io.BytesIO(pdf_resp.content), strict=False)
            if not paper.page_count:
                paper.page_count = len(reader.pages)
            if reader.pages and not paper.institutions:
                page1_text = reader.pages[0].extract_text() or ""
                if page1_text:
                    prompt = prompts.INSTITUTION_EXTRACTION.format(
                        source_label="first page of the PDF",
                        title=paper.title,
                        text=page1_text[:5000],
                    )
                    result, _, _, _ = _llm_json(
                        [{"role": "user", "content": prompt}], temperature=0, max_tokens=4096
                    )
                    if isinstance(result, list):
                        insts = [str(i).strip() for i in result if i and str(i).strip()]
                        if insts:
                            paper.institutions = json.dumps(insts)
    except Exception as e:
        _log.warning("[web-enrich] PDF metadata fetch failed for %s: %s", pdf_url, e)


def enrich_page_counts(papers: list) -> None:
    """Fetch PDF page count for papers that have an arxiv_id but no page_count yet.

    Downloads each PDF and counts pages with pypdf. Runs up to 4 in parallel.
    Updates paper.page_count in-place; caller must db.session.commit() afterwards.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        _log.warning("[page_count] pypdf not installed — skipping page count enrichment")
        return

    to_enrich = [p for p in papers if p.arxiv_id and not p.page_count]
    if not to_enrich:
        return

    def fetch_count(arxiv_id: str) -> Optional[int]:
        import io
        try:
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            with _ArxivLockCtx():
                with _http_client(timeout=httpx.Timeout(45.0), follow_redirects=True) as http:
                    r = http.get(url)
            if r.status_code == 200 and r.content:
                return len(PdfReader(io.BytesIO(r.content), strict=False).pages)
        except Exception as e:
            _log.warning("[page_count] Failed for %s: %s", arxiv_id, e)
        return None

    arxiv_ids = [p.arxiv_id for p in to_enrich]
    id_to_paper = {p.arxiv_id: p for p in to_enrich}

    with ThreadPoolExecutor(max_workers=4) as pool:
        for arxiv_id, count in zip(arxiv_ids, pool.map(fetch_count, arxiv_ids)):
            if count:
                id_to_paper[arxiv_id].page_count = count
                _log.info("[page_count] %s → %d pages", arxiv_id, count)


def select_featured_paper_llm(papers: list, research_interest: str) -> tuple:
    """Return (paper, tokens_used). paper may be None if papers is empty.

    Only papers with a figure are candidates when any has one, so the crawl email shows an image
    whenever one is available at all."""
    if not papers:
        return None, 0
    papers = [p for p in papers if p.figure_url] or papers
    if len(papers) == 1:
        return papers[0], 0

    items = []
    for i, p in enumerate(papers):
        snippet = (p.abstract or "")[:200]
        citations = getattr(p, "citation_count", None)
        cit_str = f" · {citations:,} citations" if citations else ""
        items.append(f"{i + 1}. {p.title}{cit_str}\n   {snippet}")

    prompt = prompts.FEATURED_PAPER_SELECTION.format(
        research_interest=research_interest or "(not specified)",
        papers="\n\n".join(items),
    )
    try:
        result, _, _, usage = _llm_json(
            [{"role": "user", "content": prompt}], temperature=0.1, max_tokens=4096
        )
        tokens = int(usage.get("total_tokens", 0))
        idx = int(result.get("selected", 1)) - 1
        if 0 <= idx < len(papers):
            _log.info("[featured] LLM selected paper %d: %s", idx + 1, papers[idx].title)
            return papers[idx], tokens
    except Exception as e:
        _log.warning("[featured] LLM selection failed: %s", e)

    return papers[0], 0


SCHOLAR_PAPER_FIELDS = "externalIds,title,authors,abstract,year,publicationDate,venue,openAccessPdf"


def journal_paper_id(doi: str) -> str:
    """Id of a non-arXiv, non-bioRxiv paper found through Semantic Scholar (a journal article), from its DOI.
    No slash or dot: it goes into URLs and DOM ids."""
    return "web:doi-" + hashlib.sha256(doi.lower().encode()).hexdigest()[:12]


def _scholar_paper(item: dict, pub_date: Optional[date]) -> Optional[dict]:
    """A Semantic Scholar result as a candidate. An arXiv paper by its arXiv id; a bioRxiv preprint like a
    bioRxiv keyword find (same id, read from its PDF on bioRxiv); any other paper with a DOI and an abstract
    (mostly journal articles, most of what bio and chemistry seeds bring) under journal_paper_id, read from
    its open-access PDF only if that is on OPEN_ACCESS_PDF_DOMAINS, else screened on the abstract. None for
    the rest (no DOI, or no abstract to screen)."""
    ext = item.get("externalIds") or {}
    aid, doi = ext.get("ArXiv"), ext.get("DOI") or None
    paper = {
        "title": item.get("title") or "",
        "authors": [a.get("name", "") for a in (item.get("authors") or [])],
        "abstract": item.get("abstract") or "",
        "doi": doi,
        "published_date": pub_date,
        "year": pub_date.year if pub_date else (item.get("year") or None),
        "categories": [],
        "source": "semantic_scholar",
    }
    if aid:
        return {**paper, "arxiv_id": base_arxiv_id(aid), "pdf_url": f"https://arxiv.org/pdf/{base_arxiv_id(aid)}"}
    if not doi or not paper["abstract"]:
        return None
    if (item.get("venue") or "").lower() == "biorxiv":
        return {**paper, "arxiv_id": biorxiv_paper_id(doi), "pdf_url": BIORXIV_PDF_URL.format(doi=doi, version=1)}
    oa = (item.get("openAccessPdf") or {}).get("url") or ""
    return {**paper, "arxiv_id": journal_paper_id(doi),
            "pdf_url": oa if oa and _is_open_access_pdf_domain(oa) else f"https://doi.org/{doi}"}


# Semantic Scholar's recommendations endpoint takes at most 100 seed papers and returns at most 500
# recommendations per call, ranked by similarity, not date, and spread over the last few months. The crawl keeps
# only the ones published in its window (5 days nightly), so a single call with all seeds has few of those.
# Smaller seed groups each get their own top 500, and together they reach many more recent papers, which pass
# screening about as often. One call per seed (the "recent" pool) brings many more still, but mostly unrelated
# ones, so it isn't used. Groups of 10: at most 10 calls per crawl.
RECOMMENDATION_MAX_SEEDS = 100
RECOMMENDATION_GROUP_SIZE = 10
RECOMMENDATION_LIMIT = 500


def get_semantic_scholar_recommendations(arxiv_ids: list[str], limit: int = RECOMMENDATION_LIMIT) -> list[dict]:
    """Recommended papers (_scholar_paper) for the seeds (collection_seeds, best first, the first
    RECOMMENDATION_MAX_SEEDS used), asked for in groups of RECOMMENDATION_GROUP_SIZE seeds with `limit` results
    each (why: see the comment above).
    A paper recommended for several groups comes back once. Returns a list of paper dicts compatible with the
    arXiv search result format."""
    seeds = [_scholar_ref(s) for s in arxiv_ids[:RECOMMENDATION_MAX_SEEDS]]
    groups = [seeds[i:i + RECOMMENDATION_GROUP_SIZE] for i in range(0, len(seeds), RECOMMENDATION_GROUP_SIZE)]
    papers: dict[str, dict] = {}
    for g, group in enumerate(groups, 1):
        try:
            r = _scholar_call("POST", "https://api.semanticscholar.org/recommendations/v1/papers/", timeout=60.0,
                              json={"positivePaperIds": group},
                              params={"fields": SCHOLAR_PAPER_FIELDS, "limit": limit})
            if r.status_code != 200:
                _log.warning("[Scholar] Recommendations API returned %d for group %d/%d: %s",
                             r.status_code, g, len(groups), r.text[:200])
                continue
            for item in r.json().get("recommendedPapers", []):
                pub_date_str = item.get("publicationDate") or ""
                try:
                    pub_date = date.fromisoformat(pub_date_str) if pub_date_str else None
                except ValueError:
                    pub_date = None
                paper = _scholar_paper(item, pub_date)
                if paper and paper["arxiv_id"] not in papers:  # repeats from earlier groups come back once
                    papers[paper["arxiv_id"]] = paper
        except Exception as e:
            _log.warning("[Scholar] Recommendations error for group %d/%d: %s", g, len(groups), e)
        _log.info("[Scholar] recommendations: group %d/%d done, %d papers so far", g, len(groups), len(papers))

    _log.info("[Scholar] Got %d recommendations from %d seeds in %d groups", len(papers), len(seeds), len(groups))
    return list(papers.values())


# At most this many My Collection papers are checked for new citing papers per crawl, the ones checked longest
# ago first (citation_seeds), so a larger collection is covered in turn over a few nights. Semantic Scholar
# answers many of these calls with 429s at night, so checking a large collection in one crawl took hours and
# held up the projects scheduled after it.
CITATION_MAX_SEEDS = 50


def get_citing_papers(seeds: list[tuple[str, str]], date_from: date, date_to: date,
                      progress=None, checked: Optional[list[str]] = None) -> list[dict]:
    """Papers (_scholar_paper) published in [date_from, date_to] that cite any of the seeds ((seed, title)
    pairs, from citation_seeds; only the first CITATION_MAX_SEEDS are used). One Semantic Scholar call per seed: it
    lists citing papers newest first, so a page usually covers the whole window; it pages on while the page
    still reaches into the window. A paper citing several seeds comes back once, with one found_via entry per
    seed. Same dict shape as get_semantic_scholar_recommendations. progress(i, n) is called after each seed.
    checked (optional list) gets the id of every seed Semantic Scholar answered for (not one it gave up on
    with 429s), for mark_citations_checked."""
    if len(seeds) > CITATION_MAX_SEEDS:
        _log.info("[citations] %d My Collection papers, checking the %d checked longest ago", len(seeds), CITATION_MAX_SEEDS)
        seeds = seeds[:CITATION_MAX_SEEDS]
    found: dict[str, dict] = {}
    started = time.time()
    for i, (seed_id, seed_title) in enumerate(seeds, 1):
        offset = 0
        while True:
            r = _scholar_call("GET", f"https://api.semanticscholar.org/graph/v1/paper/{_scholar_ref(seed_id)}/citations",
                              params={"fields": SCHOLAR_PAPER_FIELDS, "limit": 100, "offset": offset})
            if r.status_code != 429 and checked is not None and seed_id not in checked:
                checked.append(seed_id)
            if r.status_code != 200:  # 404: Semantic Scholar doesn't know the seed
                _log.info("[citations] %s: Semantic Scholar answered %d", seed_id, r.status_code)
                break
            body = r.json()
            oldest = None
            for c in body.get("data") or []:
                item = c.get("citingPaper") or {}
                try:
                    pub_date = date.fromisoformat(item.get("publicationDate") or "")
                except ValueError:
                    continue
                oldest = pub_date if oldest is None else min(oldest, pub_date)
                if not (date_from <= pub_date <= date_to):
                    continue
                cand = _scholar_paper(item, pub_date)
                if not cand:
                    continue
                via = ({"via": "cites", "seed": seed_id, "title": seed_title} if _scholar_ref(seed_id) == seed_id
                       else {"via": "cites", "arxiv_id": base_arxiv_id(seed_id), "title": seed_title})
                paper = found.setdefault(cand["arxiv_id"], {**cand, "found_via": []})
                if via not in paper["found_via"]:
                    paper["found_via"].append(via)
            if body.get("next") is None or oldest is None or oldest < date_from:
                break
            offset = body["next"]
        if progress:
            progress(i, len(seeds))
        if i % 10 == 0 or i == len(seeds):
            elapsed = time.time() - started
            _log.info("[citations] %d/%d seeds, %d citing papers so far, %.0fs (ETA %.0fs)",
                      i, len(seeds), len(found), elapsed, elapsed / i * (len(seeds) - i))
    _log.info("[citations] %d papers in %s–%s cite %d My Collection papers", len(found), date_from, date_to, len(seeds))
    return list(found.values())


def _matched_keywords(paper: dict, keywords: list[str]) -> list[str]:
    """Which keywords appear in the title or abstract. The arXiv search sends all keywords as one OR query,
    so arXiv doesn't say which matched; this is the local guess. arXiv also matches word variants, so it
    can come back empty for a paper arXiv did find."""
    norm = lambda t: re.sub(r"\s+", " ", (t or "").lower().replace("-", " "))
    text = norm(paper.get("title", "") + " " + paper.get("abstract", ""))
    return [k for k in keywords if norm(k).strip() and norm(k).strip() in text]


def scholar_seed(paper) -> Optional[str]:
    """How Semantic Scholar knows a My Collection paper, as a seed: its arXiv id; else "DOI:<doi>"; else the
    Semantic Scholar id found by title (resolve_scholar_ids); None if none of these (a blog post, say)."""
    if paper.is_arxiv:
        return base_arxiv_id(paper.arxiv_id)
    if paper.doi:
        return f"DOI:{paper.doi}"
    return paper.semantic_scholar_id or None


def _scholar_ref(seed: str) -> str:
    """The Semantic Scholar paper id for a seed: "ArXiv:<id>" for an arXiv id; a "DOI:" or Semantic Scholar
    id as it is."""
    if seed.startswith("DOI:") or re.fullmatch(r"[0-9a-f]{40}", seed):
        return seed
    return f"ArXiv:{base_arxiv_id(seed)}"


def resolve_scholar_ids(collection: list) -> None:
    """Give the non-arXiv My Collection papers a seed for the Semantic Scholar sources: the DOI in their URL,
    else a Semantic Scholar title match (checked with _titles_match). Each paper is looked up once
    (scholar_lookup_at); after a 429 it is left for the next crawl."""
    for pp in collection:
        paper = pp.paper
        if not paper or paper.is_arxiv or paper.doi or paper.semantic_scholar_id or paper.scholar_lookup_at:
            continue
        doi = doi_from_url(paper.pdf_url)
        if doi:
            paper.doi = doi
            continue
        try:
            r = _scholar_call("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match",
                              params={"query": paper.title, "fields": "title,externalIds"})
        except httpx.HTTPError as e:
            _log.warning("[Scholar] title lookup failed for %s: %s", paper.arxiv_id, e)
            continue
        if r.status_code == 429:
            continue
        paper.scholar_lookup_at = datetime.utcnow()
        match = ((r.json().get("data") or [{}])[0]) if r.status_code == 200 else {}
        if match.get("paperId") and _titles_match(paper.title, match.get("title") or ""):
            paper.semantic_scholar_id = match["paperId"]
            paper.doi = (match.get("externalIds") or {}).get("DOI") or None
        _log.info("[Scholar] %s %r: %s", paper.arxiv_id, paper.title[:60],
                  f"found ({paper.doi or paper.semantic_scholar_id})" if paper.semantic_scholar_id else "not found")
    db.session.commit()


def collection_seeds(collection: list) -> list[tuple[str, str]]:
    """(seed, title) of the My Collection papers Semantic Scholar can know (scholar_seed): the seeds for
    recommendations and citations. collection: ProjectPaper links, best first (_collection_tag_tier)."""
    return [(seed, pp.paper.title) for pp in collection if pp.paper and (seed := scholar_seed(pp.paper))]


def citation_seeds(collection: list) -> list[tuple[str, str]]:
    """collection_seeds() ordered for the citations source: never checked first (in collection order), then
    the ones checked longest ago (ProjectPaper.citations_checked_at)."""
    ordered = sorted(collection, key=lambda pp: (pp.citations_checked_at is not None,
                                                 pp.citations_checked_at or datetime.min))
    return collection_seeds(ordered)


def mark_citations_checked(project_id: int, arxiv_ids: list[str]) -> None:  # seeds, see collection_seeds
    """Stamp the My Collection papers the citations source just got an answer for, so the next crawl starts
    with the others (citation_seeds)."""
    if not arxiv_ids:
        return
    ids = {_scholar_ref(a) for a in arxiv_ids}
    now = datetime.utcnow()
    for pp in ProjectPaper.query.filter_by(project_id=project_id).all():
        if pp.paper and (seed := scholar_seed(pp.paper)) and _scholar_ref(seed) in ids:
            pp.citations_checked_at = now
    db.session.commit()


def gather_candidates(keywords: list[str], seeds: list[tuple[str, str]], date_from: date, date_to: date,
                      use_arxiv: bool = True, use_scholar: bool = True, use_citations: bool = True,
                      failed_keywords: Optional[list[str]] = None, warnings: Optional[list[str]] = None,
                      status: Optional[dict] = None, cite_seeds: Optional[list[tuple[str, str]]] = None,
                      citations_checked: Optional[list[str]] = None, use_biorxiv: bool = False) -> list[dict]:
    """Candidates from the sources, in this order: arXiv keyword search, bioRxiv keyword search (off unless
    asked for), Semantic Scholar recommendations (seeded with My Collection), papers citing My Collection.
    A paper found more than once comes back once, and its "found_via" lists every way it was found.
    seeds: collection_seeds(), for the recommendations; cite_seeds: citation_seeds() for the citations
    (defaults to seeds). citations_checked: see get_citing_papers. status (optional dict) gets
    "scholar_status" and "citations" progress for the manual count step."""
    sources = []
    if use_arxiv:
        found = search_arxiv_with_full_papers(keywords, date_from, date_to, warnings=warnings,
                                              failed_keywords=failed_keywords)
        for p in found:
            names = _matched_keywords(p, keywords)
            p["found_via"] = [{"via": "keyword", "name": k} for k in names] or [{"via": "keyword"}]
        sources.append(found)
    if use_biorxiv:
        sources.append(search_biorxiv(keywords, date_from, date_to, warnings=warnings))
    if use_scholar and seeds:
        if status is not None:
            status["scholar_status"] = "running"
        recs = [p for p in get_semantic_scholar_recommendations([s[0] for s in seeds])
                # The recommendations API has no date filter, so it returns old papers too
                if p.get("published_date") is not None and date_from <= p["published_date"] <= date_to]
        for p in recs:
            p["found_via"] = [{"via": "recommended"}]
        sources.append(recs)
    cite_seeds = seeds if cite_seeds is None else cite_seeds
    if use_citations and cite_seeds:
        def _progress(i, n):
            if status is not None:
                status["citations"] = f"{i}/{n}"
        sources.append(get_citing_papers(cite_seeds, date_from, date_to, progress=_progress,
                                         checked=citations_checked))

    merged: dict[str, dict] = {}
    for found in sources:
        for p in found:
            key = base_arxiv_id(p["arxiv_id"])
            if key in merged:
                merged[key]["found_via"] += [v for v in p["found_via"] if v not in merged[key]["found_via"]]
            else:
                merged[key] = {**p, "found_via": list(p["found_via"])}
    return list(merged.values())


_citation_cache: dict[str, int] = {}
_CITATION_429_GIVE_UP = 5  # consecutive papers that exhausted retries on 429 → pause citation fetches
_CITATION_PAUSE_S = 900
_citation_429_streak = 0
_citation_paused_until = 0.0


def _fetch_citations(paper_ref: str) -> Optional[int]:
    """Core Scholar citation fetch. paper_ref is 'arXiv:{id}' or a raw Scholar paper ID.

    Returns the citation count, or None on any failure (429, timeout, 404, etc.).
    None means "don't update the stored value" — callers must check before writing to DB.
    Only caches successful results so transient failures are retried on the next call.
    """
    paper_ref = re.sub(r"^(arXiv:.+?)v\d+$", r"\1", paper_ref)  # Scholar 404s on versioned arXiv ids
    if paper_ref in _citation_cache:
        return _citation_cache[paper_ref]

    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_ref}?fields=citationCount"
    global _citation_429_streak, _citation_paused_until
    if time.time() < _citation_paused_until:
        return None  # Scholar has been refusing us for many papers in a row; stop hammering it for a while
    try:
        r = _scholar_call("GET", url)
    except httpx.HTTPError as e:
        _log.warning("[Scholar] Exception fetching citations for %s: %s", paper_ref, e)
        return None
    if r.status_code == 200:
        count = r.json().get("citationCount", 0)
        _citation_cache[paper_ref] = count
        _citation_429_streak = 0
        return count
    if r.status_code == 429:
        _citation_429_streak += 1
        if _citation_429_streak >= _CITATION_429_GIVE_UP:
            _citation_429_streak = 0
            _citation_paused_until = time.time() + _CITATION_PAUSE_S
            _log.warning("[Scholar] %d papers in a row rate limited — pausing citation fetches for %ds", _CITATION_429_GIVE_UP, _CITATION_PAUSE_S)
    _log.warning("[Scholar] Error %d fetching citations for %s", r.status_code, paper_ref)
    return None


SCHOLAR_BATCH_SIZE = 500  # the most ids /paper/batch takes in one request


def get_citation_counts(refs: list[str]) -> dict[str, int]:
    """Citation counts for many papers ("arXiv:<id>", "DOI:<doi>" or Semantic Scholar ids) in one /paper/batch
    request per SCHOLAR_BATCH_SIZE. Papers Semantic Scholar doesn't know, and every paper of a batch it kept
    refusing, are left out."""
    counts: dict[str, int] = {}
    for i in range(0, len(refs), SCHOLAR_BATCH_SIZE):
        chunk = refs[i:i + SCHOLAR_BATCH_SIZE]
        try:
            r = _scholar_call("POST", "https://api.semanticscholar.org/graph/v1/paper/batch", timeout=60.0,
                              params={"fields": "citationCount"}, json={"ids": chunk})
        except httpx.HTTPError as e:
            _log.warning("[Scholar] citation batch failed: %s", e)
            continue
        if r.status_code != 200:
            _log.warning("[Scholar] citation batch of %d returned %d", len(chunk), r.status_code)
            continue
        for ref, item in zip(chunk, r.json()):  # one entry per id, in order; None for an unknown paper
            if item is not None:
                counts[ref] = item.get("citationCount") or 0
    return counts


def get_citation_count(arxiv_id: str) -> Optional[int]:
    return _fetch_citations(f"arXiv:{arxiv_id}")


def get_citation_count_by_scholar_id(scholar_id: str) -> Optional[int]:
    return _fetch_citations(scholar_id)


def get_paper_content(arxiv_id: str, pdf_url: Optional[str] = None) -> Optional[str]:
    """Return the paper's full text (arXiv HTML, else the PDF), or None. Every caller already has the
    abstract and falls back to it, so there is no extra arXiv API request for it here. A non-arXiv paper is
    read only from an open-access domain: a bioRxiv paper like an arXiv one (its HTML, else its PDF), any
    other only from its PDF."""
    from app.utils import fetch_paper_full_text, fetch_pdf_text, fetch_biorxiv_full_text, _is_biorxiv

    if arxiv_id.startswith("web:") and pdf_url and _is_biorxiv(pdf_url):
        return fetch_biorxiv_full_text(pdf_url) or fetch_pdf_text(pdf_url, arxiv=False)
    if arxiv_id.startswith("web:"):
        return fetch_pdf_text(pdf_url, arxiv=False) if pdf_url and _is_open_access_pdf_domain(pdf_url) else None
    return fetch_paper_full_text(arxiv_id) or fetch_pdf_text(f"https://arxiv.org/pdf/{arxiv_id}", arxiv=True)


def _relevance_ref_text(collection_papers: list[dict]) -> str:
    if not collection_papers:
        return ""
    return "Researcher's collection (for context on what is relevant):\n" + "\n".join(
        [f"- {p.get('title', '')}: {p.get('abstract', '')[:400]}" for p in collection_papers]
    )


def _screen_paper_quick(
    title: str,
    abstract: str,
    research_interest: str,
    collection_papers: list[dict],
) -> tuple[bool, str, str, int, int]:
    """Phase 1 of paper relevance screening: title + abstract only, no full-text fetch —
    this is what keeps screening cheap (and avoids an ar5iv fetch per candidate) across
    hundreds of candidates per crawl. Callers on the hot crawl path should only fetch
    paper_content and call _screen_paper_deep if this returns relevant.

    Returns (is_relevant, reason, model_used, elapsed_ms, tokens_used). reason is one of
    "not-relevant", "relevant", "parse-failed" — a parse failure is treated as not-relevant
    (paper is skipped) rather than raising.
    """
    ref_text = _relevance_ref_text(collection_papers)

    quick_prompt = prompts.RELEVANCE_QUICK.format(
        research_interest=research_interest,
        ref_text=ref_text,
        title=title,
        abstract=abstract,
    )

    result, model, elapsed, usage = _llm_json(
        [{"role": "user", "content": quick_prompt}], temperature=0.3, max_tokens=4096
    )
    tokens_used = int(usage.get("total_tokens", 0))

    if not isinstance(result, dict) or "relevant" not in result:
        print(f"[relevance] Quick check parse failed for '{title[:80]}' — skipping paper. result={result!r}")
        return False, "parse-failed", model, elapsed, tokens_used

    relevant = bool(result.get("relevant"))
    return relevant, ("relevant" if relevant else "not-relevant"), model, elapsed, tokens_used


def _screen_paper_deep(
    title: str,
    abstract: str,
    paper_content: str,
    research_interest: str,
    collection_papers: list[dict],
) -> tuple[bool, str, str, int, int]:
    """Phase 2 of paper relevance screening: re-evaluates with up to 150k chars of full
    paper text. Only meaningful to call for a candidate that already passed the quick
    check — see _screen_paper_quick.

    Returns (is_relevant, reason, model_used, elapsed_ms, tokens_used, why). reason is one of
    "not-relevant", "relevant", "deep-parse-failed"; why is the LLM's one sentence on why the
    paper matters for this research interest ("" if not relevant or not given), stored per
    project as ProjectPaper.relevance_reason.
    """
    ref_text = _relevance_ref_text(collection_papers)

    deep_prompt = prompts.RELEVANCE_DEEP.format(
        research_interest=research_interest,
        ref_text=ref_text,
        title=title,
        abstract=abstract,
        paper_content=paper_content[:150000],
    )

    result, model, elapsed, usage = _llm_json(
        [{"role": "user", "content": deep_prompt}], temperature=0.3, max_tokens=4096
    )
    tokens_used = int(usage.get("total_tokens", 0))

    if not isinstance(result, dict) or "relevant" not in result:
        print(f"[relevance] Deep check parse failed for '{title[:80]}' — treating as not relevant. result={result!r}")
        return False, "deep-parse-failed", model, elapsed, tokens_used, ""

    relevant = bool(result.get("relevant"))
    why = str(result.get("reason") or "").strip() if relevant else ""
    return relevant, ("relevant" if relevant else "not-relevant"), model, elapsed, tokens_used, why


def summary_source(paper_content: str | None, abstract: str) -> str:
    """What generate_summary_with_full_content / extract_main_contributions actually read for these
    inputs (they fall back to the abstract unless the content is longer): "full_text" or "abstract"."""
    return "full_text" if paper_content and len(paper_content) > len(abstract or "") else "abstract"


_BIORXIV_POST_RE = re.compile(r"(?<!\d)(\d{4}\.\d{2}\.\d{2}\.\d{5,})")


def upgrade_biorxiv_paper(paper) -> bool:
    """Make a paper added by hand from a bioRxiv link a full bioRxiv paper, as if the crawl had found it: DOI,
    posting date and corresponding author's institution from the bioRxiv API, the crawl's id
    (biorxiv_paper_id, so a crawl that finds it again sees it is already there; kept if another paper has
    it), its latest PDF as pdf_url, a summary from the full text (HTML, else PDF), figure 1, and citations by
    DOI. Before 2026-10-20 bioRxiv was not on the open-access list, so such papers got a summary of the web
    page and their PDF never read. False, paper unchanged, if bioRxiv doesn't know it. Caller commits."""
    post = _BIORXIV_POST_RE.search(f"{paper.doi or ''} {paper.pdf_url or ''}")
    dois = [paper.doi] if (paper.doi or "").startswith(("10.1101/", "10.64898/")) else []
    dois += [f"{prefix}/{post.group(1)}" for prefix in ("10.1101", "10.64898") if post]  # prefix changed in 2026
    versions = next((v for v in (biorxiv_versions(d) for d in dict.fromkeys(dois)) if v), [])
    if not versions:
        _log.warning("[bioRxiv] %s (%s): bioRxiv does not know it", paper.arxiv_id, paper.pdf_url)
        return False
    first, last = versions[0], versions[-1]
    doi = last["doi"]
    new_id = biorxiv_paper_id(doi)
    if new_id != paper.arxiv_id and Paper.query.filter_by(arxiv_id=new_id).first():
        _log.warning("[bioRxiv] %s: a crawl already stored %s as %s, both kept", paper.arxiv_id, doi, new_id)
    else:
        paper.arxiv_id = new_id
    paper.doi, paper.source = doi, "biorxiv"
    paper.pdf_url = BIORXIV_PDF_URL.format(doi=doi, version=last.get("version") or 1)
    paper.published_date = datetime.strptime(first["date"], "%Y-%m-%d").date()
    paper.year = paper.published_date.year
    paper.abstract = paper.abstract or " ".join((last.get("abstract") or "").split())
    if not paper.institutions_list and (last.get("author_corresponding_institution") or "").strip():
        paper.institutions = json.dumps([last["author_corresponding_institution"].strip()])
    paper.web_enrich_attempts = 0  # the card's "PDF metadata failed": Pass 3 is for papers from other sites
    content = get_paper_content(paper.arxiv_id, paper.pdf_url)
    if content:
        write_summary(paper, content)
    if not paper.figure_url:
        paper.figure_url, paper.figure_caption = fetch_biorxiv_figure(paper.pdf_url)
    count = _fetch_citations(f"DOI:{doi}")
    if count is not None:
        paper.citation_count, paper.citation_fetched_at = count, datetime.utcnow()
    _log.info("[bioRxiv] %s upgraded: %s, %s, figure %s", paper.arxiv_id, doi,
              "summary from the full text" if content else "no full text", "yes" if paper.figure_url else "not yet")
    return True


def write_summary(paper, content: str) -> None:
    """Write (or rewrite) a paper's shared summary, badges and main contributions from content (the full text,
    or the abstract if there is none). Used by Regenerate and the bioRxiv upgrade. Caller commits."""
    abstract = paper.abstract or ""
    summary_data, model, elapsed, tokens = generate_summary_with_full_content(
        title=paper.title, abstract=abstract, paper_content=content)
    contributions, _, contrib_elapsed, contrib_tokens = extract_main_contributions(
        title=paper.title, paper_content=content, abstract=abstract)
    summary_data = summary_data if isinstance(summary_data, dict) else {}
    fields = dict(
        content_source=summary_source(content, abstract),
        summary_text=summary_data.get("summary", ""),
        model_used=model,
        introduces_dataset=summary_data.get("introduces_dataset", False),
        introduces_architecture=summary_data.get("introduces_architecture", False),
        introduces_method=summary_data.get("introduces_method", False),
        is_survey=summary_data.get("is_survey", False),
        is_benchmark=summary_data.get("is_benchmark", False),
        elapsed_ms=elapsed + contrib_elapsed,
        tokens_used=tokens + contrib_tokens,
    )
    if paper.summary:
        for k, v in fields.items():
            setattr(paper.summary, k, v)
        if contributions:
            paper.summary.main_contributions = contributions
    else:
        db.session.add(PaperSummary(paper_id=paper.id, main_contributions=contributions, **fields))


def extract_main_contributions(
    title: str, paper_content: str, abstract: str
) -> tuple[str, str, int, int]:
    """Returns (contributions_text, contribution_summary, elapsed_ms, tokens)."""
    content = paper_content if paper_content and len(paper_content) > len(abstract) else abstract

    prompt = prompts.MAIN_CONTRIBUTIONS.format(
        title=title,
        content=(content[:100000] if content else abstract),
    )

    try:
        result, model, elapsed, usage = _llm_json(
            [{"role": "user", "content": prompt}], temperature=0.2, max_tokens=4096
        )
        if isinstance(result, dict):
            contributions = result.get("main_contributions", [])
            if isinstance(contributions, list):
                formatted = "\n".join(f"- {c}" for c in contributions[:4])
                return (
                    formatted,
                    result.get("contribution_summary", ""),
                    elapsed,
                    int(usage.get("total_tokens", 0)),
                )
        return "", "", 0, int(usage.get("total_tokens", 0))
    except LLMUnavailableError:
        raise
    except Exception:
        return "", "", 0, 0


def generate_summary_with_full_content(
    title: str, abstract: str, paper_content: str
) -> tuple[dict, str, int, int]:
    """Returns (summary_dict, model, elapsed_ms, tokens). summary_dict has keys:
    summary, introduces_dataset,
    introduces_architecture, introduces_method, is_survey, is_benchmark
    (see prompts.SUMMARY_WITH_FULL_CONTENT). On failure, returns that same shape
    with empty/False defaults rather than raising (except LLMUnavailableError)."""
    content = paper_content if paper_content and len(paper_content) > len(abstract) else abstract

    prompt = prompts.SUMMARY_WITH_FULL_CONTENT.format(
        title=title,
        abstract=abstract,
        content=(content[:150000] if content else abstract),
    )

    try:
        result, model, elapsed, usage = _llm_json(
            [{"role": "user", "content": prompt}], temperature=0.3, max_tokens=4096
        )
        if isinstance(result, dict) and isinstance(result.get("summary"), list):
            result["summary"] = " ".join(result["summary"])
        return result, model, elapsed, int(usage.get("total_tokens", 0))
    except LLMUnavailableError:
        raise
    except Exception:
        return (
            {
                "summary": "",
                "introduces_dataset": False,
                "introduces_architecture": False,
                "introduces_method": False,
                "is_survey": False,
                "is_benchmark": False,
            },
            LLM_MODELS[0],
            0,
            0,
        )


EMAIL_SOURCE_LABELS = [("keyword", "arXiv keyword search"), ("biorxiv", "bioRxiv keyword search"),
                       ("recommended", "Semantic Scholar recommendations"),
                       ("cites", "Papers citing My Collection")]


def _email_source_rows(source_counts: Optional[dict]) -> list[tuple[str, str]]:
    """The sources an email lists: all of them, except bioRxiv when the crawl did not find or add anything
    from it (most projects never search it)."""
    return [(via, label) for via, label in EMAIL_SOURCE_LABELS
            if via != "biorxiv" or any((source_counts or {}).get(via, {}).values())]


def _count_sources(papers: list[dict], counts: Optional[dict] = None, key: str = "found") -> dict:
    """Add each paper once to every source it was found by ({via: {"found": n, "added": m}})."""
    counts = counts if counts is not None else {via: {"found": 0, "added": 0} for via, _ in EMAIL_SOURCE_LABELS}
    for p in papers:
        for via in {fv.get("via") for fv in p.get("found_via") or []}:
            if via in counts:
                counts[via][key] += 1
    return counts


def _merge_sources(a: Optional[dict], b: Optional[dict]) -> dict:
    out = {via: {"found": 0, "added": 0} for via, _ in EMAIL_SOURCE_LABELS}
    for c in (a or {}, b or {}):
        for via, n in c.items():
            if via in out:
                out[via]["found"] += n.get("found", 0)
                out[via]["added"] += n.get("added", 0)
    return out


def _make_carry(crawl_log, papers_added, papers_checked, papers, source_counts) -> dict:
    """First-pass numbers handed to the retry pass so both produce one combined email."""
    return {
        "papers_found": crawl_log.papers_found, "papers_added": papers_added, "papers_checked": papers_checked,
        "papers": list(papers), "keywords": list(crawl_log.keywords_list or []),
        "sources": source_counts,
    }


def _merge_carry(earlier: Optional[dict], later: dict) -> dict:
    """Add up two deferred passes (first pass + a retry round that failed again and deferred too)."""
    if not earlier:
        return later
    merged = {k: earlier.get(k, 0) + later.get(k, 0) for k in ("papers_found", "papers_added", "papers_checked")}
    merged["sources"] = _merge_sources(earlier.get("sources"), later.get("sources"))
    merged["papers"] = list(earlier.get("papers", [])) + list(later.get("papers", []))
    merged["keywords"] = list(dict.fromkeys(list(earlier.get("keywords", [])) + list(later.get("keywords", []))))
    return merged


def _search_warning(failed: list[str]) -> str:
    """Email banner text when arXiv never answered. All keywords go in one query, so it is all or nothing."""
    days_back = int(os.environ.get("CRAWL_DAYS_BACK", "5"))
    return (f"arXiv did not answer tonight (rate limited or unavailable), so papers from the arXiv keyword search "
            f"({', '.join(failed)}) may be missing; the papers here come from Semantic Scholar (recommendations and "
            f"papers citing My Collection). "
            f"Each night searches the last {days_back} days, so the next nights cover these days again.")


@counts("crawl")
def send_carry_notification(project_id: int, carry: dict, date_from, date_to, warning: str) -> bool:
    """Send the deferred first-pass email when the retry pass could not (it errored before screening)."""
    project = Project.query.get(project_id)
    featured, _ = select_featured_paper_llm(carry["papers"], project.research_interest or "")
    return notify_crawl_complete(
        project.name, project.slug, carry["papers_found"], carry["papers_added"],
        [featured] if featured else [], project_id=project_id, keywords_used=carry["keywords"],
        papers_screened=carry["papers_checked"], source_counts=carry.get("sources"),
        date_from=date_from, date_to=date_to, warning=warning,
    )


@counts("crawl")
def run_crawl(
    project_id: int,
    date_from: date,
    date_to: date,
    triggered_by: str = "manual",
    paper_limit: int = 500,
    crawl_log_id: Optional[int] = None,
    keywords_override: Optional[list[str]] = None,
    cached_papers: Optional[list[dict]] = None,
    use_arxiv: bool = True,
    use_scholar: bool = True,
    use_citations: bool = True,
    defer_notification: bool = False,
    carry: Optional[dict] = None,
    use_biorxiv: bool = False,
) -> Optional[CrawlLog]:
    """Run one crawl end-to-end: keywords → search → screen → summarize → enrich → notify.

    Always called via start_crawl_async (a daemon thread), never directly from a request
    handler — this function blocks for the full crawl duration. See AGENTS.md's
    "Crawl pipeline" section for the numbered step-by-step flow.

    Creates (or reuses, if crawl_log_id is passed) a CrawlLog row and updates it live as
    the crawl progresses — that row is what the crawl-status UI polls. Any existing
    "running" CrawlLog for this project is marked "cancelled" first (only one crawl per
    project at a time). defer_notification: if some keywords failed, skip the email and stash the
    numbers in crawl_log.notify_carry; pass that dict as `carry` to the retry run so the two passes
    produce ONE combined email. keywords_override/cached_papers let the UI skip re-running
    keyword extraction / arXiv search when the user already previewed a count.

    Returns the CrawlLog on completion (status "success", "cancelled", or "error"), or
    None if a lock/setup issue prevented the crawl from starting at all.
    """
    crawl_log = None
    with _crawl_lock:
        if crawl_log_id:
            crawl_log = CrawlLog.query.get(crawl_log_id)
            if crawl_log and crawl_log.status != "running":
                crawl_log = None

        if not crawl_log:
            existing = CrawlLog.query.filter_by(project_id=project_id, status="running").first()

            if existing:
                existing.status = "cancelled"
                db.session.commit()

            crawl_log = CrawlLog(
                project_id=project_id,
                triggered_by=triggered_by,
                started_at=datetime.utcnow(),
                status="running",
                date_from=date_from,
                date_to=date_to,
            )
            db.session.add(crawl_log)
            # Commit (not just flush) so the "running" row is durable before any real
            # work starts — otherwise a rollback() in the except handlers below (needed
            # to recover a broken session after a mid-crawl DB error) would also wipe out
            # the crawl_log row itself if it hadn't been committed yet.
            db.session.commit()

        print(f"[CRAWL] Using crawl id={crawl_log.id}, paper_limit={paper_limit}")

    failed_keywords: list[str] = []  # keywords whose arXiv query errored out (429/503)
    try:
        project = Project.query.get(project_id)
        if not project:
            raise ValueError("Project not found")

        # My Collection papers — used for keyword extraction, screening context, and Scholar seeds.
        # _collection_pps (full, tag-priority sorted) feeds the Scholar seed list — Scholar seeding wants
        # as many candidates as possible, not just the capped LLM-context selection below.
        _collection_pps = sorted(
            [pp for pp in ProjectPaper.query.filter_by(project_id=project_id).all()
             if pp.manual_tag and not pp.trashed_at and pp.paper],
            key=_collection_tag_tier,
        )
        collection_ref_dicts = [
            {"title": pp.paper.title, "abstract": (pp.paper.abstract or "")[:400]}
            for pp in _select_reference_papers(_collection_pps, limit=REFERENCE_PAPERS_LIMIT)
        ]

        if not project.research_interest and not _collection_pps:
            raise ValueError(
                "Cannot crawl: Please set a research interest or add papers to My Collection first."
            )

        tokens_used = 0

        keywords_auto_generated = False
        if keywords_override is not None:
            keywords = [k for k in keywords_override if k]
        else:
            keywords, usage = extract_keywords_with_usage(
                project.research_interest or "", _collection_pps
            )
            tokens_used += int(usage.get("total_tokens", 0))
            # Persist for future runs — never overwrite keywords already saved by the user
            if keywords and not project.saved_keywords:
                project.saved_keywords = json.dumps(keywords)
                keywords_auto_generated = True
                _log.info("[crawl] saved %d auto-generated keywords for project %s", len(keywords), project.name)

        if not keywords:
            raise ValueError("No keywords available for crawl.")
        crawl_log.keywords_used = json.dumps(keywords)
        crawl_log.paper_limit = paper_limit
        db.session.commit()  # persist keywords + limit so UI can show them during arXiv search

        if cached_papers is not None:
            print(f"[CRAWL] Using {len(cached_papers)} cached papers (skipping re-search).")
            papers_data = cached_papers
        else:
            _log.info("[crawl] searching arXiv for %d keywords, date range %s–%s", len(keywords), date_from, date_to)
            citations_checked: list[str] = []
            search_warnings: list[str] = []
            if use_scholar or use_citations:
                resolve_scholar_ids(_collection_pps)
            papers_data = gather_candidates(keywords, collection_seeds(_collection_pps), date_from, date_to, use_arxiv=use_arxiv,
                                            use_scholar=use_scholar, use_citations=use_citations,
                                            failed_keywords=failed_keywords, cite_seeds=citation_seeds(_collection_pps),
                                            citations_checked=citations_checked, warnings=search_warnings,
                                            use_biorxiv=use_biorxiv)
            mark_citations_checked(project_id, citations_checked)
            for w in search_warnings:
                _log.warning("[crawl] search warning: %s", w)
            if failed_keywords:
                _log.error("[crawl] arXiv search FAILED for %d/%d keywords: %s",
                           len(failed_keywords), len(keywords), failed_keywords)
        # Subtract papers already linked to this project (any state) — same logic as the count step
        existing_project_ids = {
            base_arxiv_id(pp.paper.arxiv_id)
            for pp in ProjectPaper.query.filter_by(project_id=project_id).all()
            if pp.paper and pp.paper.arxiv_id
        }
        papers_data = [p for p in papers_data if base_arxiv_id(p["arxiv_id"]) not in existing_project_ids]
        arxiv_papers_count = sum(1 for p in papers_data if p.get("source") not in ("semantic_scholar", "biorxiv"))
        scholar_papers_count = sum(1 for p in papers_data if p.get("source") == "semantic_scholar")
        biorxiv_papers_count = sum(1 for p in papers_data if p.get("source") == "biorxiv")
        source_counts = _count_sources(papers_data)  # for the email; "added" filled in as papers are added
        crawl_log.papers_found = len(papers_data)
        sources = []
        if arxiv_papers_count > 0 or use_arxiv:
            sources.append("arxiv")
        if use_biorxiv:
            sources.append("biorxiv")
        if use_scholar:
            sources.append("ss")
        if use_citations:
            sources.append("citations")
        crawl_log.sources_used = ",".join(sources) if sources else "arxiv,ss"
        db.session.commit()  # persist papers_found before loop

        _log.info("[crawl] found %d new papers (arXiv: %d, bioRxiv: %d, Scholar: %d) after dedup against existing",
                  len(papers_data), arxiv_papers_count, biorxiv_papers_count, scholar_papers_count)

        if not papers_data:
            _log.info("[crawl] no new papers found for date range — nothing to process")
            _apply_search_outcome(crawl_log, failed_keywords, keywords)
            crawl_log.finished_at = datetime.utcnow()
            db.session.commit()
            crawl_log.failed_keywords = failed_keywords  # transient attr, read by the nightly retry pass
            if defer_notification and failed_keywords:
                crawl_log.notify_carry = _merge_carry(
                    carry, _make_carry(crawl_log, 0, 0, [], source_counts))
            return crawl_log

        papers_to_process = papers_data if paper_limit <= 0 else papers_data[:paper_limit]
        if 0 < paper_limit < len(papers_data):
            _log.info("[crawl] capping at %d of %d found papers (paper_limit)", paper_limit, len(papers_data))

        # Build screening-hash cache: papers already rejected with the same context
        context_hash = compute_screening_hash(
            project.research_interest or "", _collection_pps
        )
        already_rejected = {
            sp.arxiv_id
            for sp in ScreenedPaper.query.filter_by(
                project_id=project_id, screening_hash=context_hash, is_relevant=False
            ).all()
        }
        if already_rejected:
            _log.info("[crawl] %d papers pre-rejected from screening cache (context hash %s)",
                      len(already_rejected), context_hash)

        papers_added = 0
        papers_skipped = 0
        papers_checked = 0
        newly_added_papers = []
        screening_started = time.time()
        cancelled = False

        # Step 1: Sequentially handle papers already in DB (needs DB reads; fast).
        papers_to_check_fresh = []
        for paper_data in papers_to_process:
            db.session.refresh(crawl_log)
            if crawl_log.status == "cancelled":
                cancelled = True
                print(f"[CRAWL] Cancelled during pre-filter at {papers_checked}/{len(papers_to_process)}")
                break

            arxiv_id = paper_data["arxiv_id"]
            existing_paper = Paper.query.filter_by(arxiv_id=arxiv_id).first()
            if existing_paper and ProjectPaper.query.filter_by(
                project_id=project_id, paper_id=existing_paper.id
            ).first():
                papers_checked += 1
                crawl_log.papers_checked = papers_checked
                db.session.commit()
            elif base_arxiv_id(arxiv_id) in already_rejected:
                # Screening cache — skip papers already rejected with the same context
                papers_skipped += 1
                _log.info("[crawl] cache-skip (same context): %s", arxiv_id)
            else:
                # A paper already in the DB (found by another project's crawl) is screened against
                # this project like any new one; if it passes, its stored summary/figure are reused.
                if existing_paper:
                    paper_data = {**paper_data, "existing_paper_id": existing_paper.id}
                papers_to_check_fresh.append(paper_data)

        # Step 2: Process new papers in parallel — workers do only HTTP + LLM calls (no DB).
        if not cancelled and papers_to_check_fresh:
            cancel_event = threading.Event()
            _research_interest = project.research_interest or ""
            _ref_dicts = collection_ref_dicts  # My Collection papers as screening context
            _crawl_log_id = crawl_log.id

            def _process_one(pd):
                if cancel_event.is_set():
                    return None
                arxiv_id = pd["arxiv_id"]
                _crawl_current_papers[_crawl_log_id] = pd.get("title", "")
                _crawl_in_flight[_crawl_log_id] = _crawl_in_flight.get(_crawl_log_id, 0) + 1
                try:
                    quick_relevant, _reason, _, _, rel_tokens = _screen_paper_quick(
                        title=pd["title"],
                        abstract=pd["abstract"],
                        research_interest=_research_interest,
                        collection_papers=_ref_dicts,
                    )
                    if not quick_relevant:
                        return {"relevant": False, "tokens": rel_tokens, "paper_data": pd}

                    if cancel_event.is_set():
                        return None

                    # Only fetch full paper content (arXiv HTML / PDF) for candidates that already passed
                    # the abstract-only quick check — avoids an outbound fetch per candidate for
                    # papers that get rejected immediately.
                    paper_content = get_paper_content(arxiv_id, pd.get("pdf_url"))

                    deep_relevant, _reason, _, _, deep_tokens, why = _screen_paper_deep(
                        title=pd["title"],
                        abstract=pd["abstract"],
                        paper_content=paper_content or pd["abstract"],
                        research_interest=_research_interest,
                        collection_papers=_ref_dicts,
                    )
                    rel_tokens += deep_tokens
                    if not deep_relevant:
                        return {"relevant": False, "tokens": rel_tokens, "paper_data": pd}

                    if cancel_event.is_set():
                        return None

                    screen_source = "full_text" if paper_content else "abstract"
                    if pd.get("existing_paper_id"):
                        return {"relevant": True, "paper_data": pd, "tokens": rel_tokens, "why": why,
                                "screen_source": screen_source}

                    if arxiv_id.startswith("web:"):
                        # bioRxiv: figure 1 from the HTML version if it is there yet (the page the deep check
                        # read, kept for an hour); else the nightly backfill adds it later (Pass 4)
                        figure_url, figure_caption = (fetch_biorxiv_figure(pd["pdf_url"])
                                                      if _is_biorxiv(pd.get("pdf_url") or "") else (None, None))
                        citation_count = _fetch_citations(f"DOI:{pd['doi']}") if pd.get("doi") else None
                    else:
                        figure_url, figure_caption = get_paper_figure(arxiv_id)
                        citation_count = get_citation_count(arxiv_id)
                    summary_data, sum_model, sum_elapsed, sum_tokens = generate_summary_with_full_content(
                        title=pd["title"],
                        abstract=pd["abstract"],
                        paper_content=paper_content,
                    )
                    contributions, _, contrib_time, contrib_tokens = extract_main_contributions(
                        title=pd["title"],
                        paper_content=paper_content,
                        abstract=pd["abstract"],
                    )
                    return {
                        "relevant": True,
                        "paper_data": pd,
                        "paper_content": paper_content,
                        "figure_url": figure_url,
                        "figure_caption": figure_caption,
                        "citation_count": citation_count,
                        "summary_data": summary_data,
                        "summary_source": summary_source(paper_content, pd["abstract"]),
                        "model": sum_model,
                        "total_elapsed": sum_elapsed + contrib_time,
                        "contributions": contributions,
                        "tokens": rel_tokens + sum_tokens + contrib_tokens,
                        "summary_tokens": sum_tokens + contrib_tokens,  # the card shows what the summary cost
                        "why": why,
                        "screen_source": screen_source,
                    }
                finally:
                    _crawl_in_flight[_crawl_log_id] = max(0, _crawl_in_flight.get(_crawl_log_id, 1) - 1)

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(contextvars.copy_context().run, _process_one, pd): pd  # keeps the usage meter
                           for pd in papers_to_check_fresh}
                for future in as_completed(futures):
                    db.session.refresh(crawl_log)
                    if crawl_log.status == "cancelled":
                        cancel_event.set()
                        cancelled = True
                        print(f"[CRAWL] Cancelled mid-parallel at {papers_checked}/{len(papers_to_process)}")
                        break

                    try:
                        result = future.result()
                    except LLMUnavailableError:
                        # Systemic failure, not a per-paper one — abort the whole crawl instead of
                        # silently counting every remaining paper as "checked" with 0 added.
                        cancel_event.set()
                        for f in futures:
                            f.cancel()
                        raise
                    except Exception as e:
                        print(f"[CRAWL] Worker error: {e}")
                        papers_checked += 1
                        crawl_log.papers_checked = papers_checked
                        db.session.commit()
                        continue

                    if result is None:
                        continue  # was cancelled

                    papers_checked += 1
                    tokens_used += result["tokens"]

                    # Persist screening result to cache for future crawls
                    _base_id = base_arxiv_id(result["paper_data"]["arxiv_id"])
                    db.session.add(ScreenedPaper(
                        project_id=project_id,
                        arxiv_id=_base_id,
                        screening_hash=context_hash,
                        is_relevant=result["relevant"],
                    ))

                    if not result["relevant"]:
                        papers_skipped += 1
                        crawl_log.papers_checked = papers_checked
                        crawl_log.total_tokens = tokens_used
                        db.session.commit()
                        continue

                    # Sequential DB writes (main thread only)
                    pd = result["paper_data"]
                    if pd.get("existing_paper_id"):
                        paper = Paper.query.get(pd["existing_paper_id"])
                        paper.doi = paper.doi or pd.get("doi")
                        db.session.add(ProjectPaper(project_id=project_id, paper_id=paper.id, origin="crawl",
                                                    relevance_reason=result["why"] or None,
                                                    screen_source=result["screen_source"],
                                                    found_via=json.dumps(pd.get("found_via") or [])))
                        newly_added_papers.append(paper)
                        _count_sources([pd], source_counts, "added")
                        papers_added += 1
                        crawl_log.papers_added = papers_added
                        crawl_log.papers_checked = papers_checked
                        crawl_log.total_tokens = tokens_used
                        db.session.commit()
                        continue

                    paper = Paper(
                        arxiv_id=pd["arxiv_id"],
                        title=pd["title"],
                        authors=json.dumps(pd["authors"]),
                        abstract=pd["abstract"],
                        pdf_url=pd["pdf_url"],
                        doi=pd.get("doi"),
                        published_date=pd["published_date"],
                        year=pd["year"],
                        institutions=json.dumps(pd["institutions"]) if pd.get("institutions") else None,
                        source=("biorxiv" if pd["arxiv_id"].startswith("web:biorxiv-")
                                else "journal" if pd["arxiv_id"].startswith("web:") else "arxiv"),
                        figure_url=result["figure_url"],
                        figure_caption=result["figure_caption"],
                        enrich_complete=1,
                    )
                    db.session.add(paper)
                    db.session.flush()

                    if result["citation_count"] is not None:
                        paper.citation_count = result["citation_count"]
                        paper.citation_fetched_at = datetime.utcnow()

                    _sd = result["summary_data"] if isinstance(result["summary_data"], dict) else {}
                    summary = PaperSummary(
                        paper_id=paper.id,
                        content_source=result["summary_source"],
                        summary_text=_sd.get("summary", ""),
                        main_contributions=result["contributions"],
                        model_used=result["model"],
                        introduces_dataset=_sd.get("introduces_dataset", False),
                        introduces_architecture=_sd.get("introduces_architecture", False),
                        introduces_method=_sd.get("introduces_method", False),
                        is_survey=_sd.get("is_survey", False),
                        is_benchmark=_sd.get("is_benchmark", False),
                        elapsed_ms=result["total_elapsed"],
                        tokens_used=result["summary_tokens"],
                    )
                    db.session.add(summary)
                    papers_added += 1
                    _count_sources([pd], source_counts, "added")
                    crawl_log.papers_added = papers_added
                    crawl_log.papers_checked = papers_checked
                    crawl_log.total_tokens = tokens_used

                    link = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first()
                    if not link:
                        link = ProjectPaper(project_id=project_id, paper_id=paper.id, origin="crawl",
                                            relevance_reason=result["why"] or None,
                                            screen_source=result["screen_source"],
                                            found_via=json.dumps(pd.get("found_via") or []))
                        db.session.add(link)
                        newly_added_papers.append(paper)
                    db.session.commit()

            _crawl_current_papers.pop(_crawl_log_id, None)
            _crawl_in_flight.pop(_crawl_log_id, None)

        crawl_log.papers_added = papers_added
        crawl_log.papers_checked = papers_checked
        crawl_log.total_tokens = tokens_used
        if not cancelled:
            _apply_search_outcome(crawl_log, failed_keywords, keywords)
        crawl_log.finished_at = datetime.utcnow()
        db.session.commit()

        if not cancelled and papers_added > 0:
            # Batch-enrich institutions and page counts for newly added papers
            all_new_papers = [
                pp.paper for pp in ProjectPaper.query.filter_by(project_id=project_id).all()
                if pp.paper
            ]
            papers_needing_institutions = [p for p in all_new_papers if not p.institutions and p.is_arxiv]
            if papers_needing_institutions:
                enrich_institutions_from_scholar(papers_needing_institutions)
            papers_needing_pages = [p for p in all_new_papers if not p.page_count and p.is_arxiv]
            if papers_needing_pages:
                enrich_page_counts(papers_needing_pages)
            db.session.commit()

        carry = carry or {}
        if defer_notification and failed_keywords and not cancelled:
            crawl_log.notify_carry = _merge_carry(carry, _make_carry(crawl_log, papers_added, papers_checked, newly_added_papers,
                                                                     source_counts))
            should_notify = False
        else:
            should_notify = not cancelled and (papers_added + carry.get("papers_added", 0)) > 0
        if should_notify:
            project = Project.query.get(project_id)
            if project:
                all_added = carry.get("papers", []) + newly_added_papers
                total_added = papers_added + carry.get("papers_added", 0)
                featured, featured_tokens = select_featured_paper_llm(
                    all_added, project.research_interest or ""
                )
                tokens_used += featured_tokens
                crawl_log.total_tokens = tokens_used
                db.session.commit()
                crawl_log.notified = True  # transient, read by the nightly script
                notify_crawl_complete(
                    project.name,
                    project.slug,
                    crawl_log.papers_found + carry.get("papers_found", 0),
                    total_added,
                    [featured] if featured else [],
                    project_id=project_id,
                    keywords_used=list(dict.fromkeys(carry.get("keywords", []) + list(crawl_log.keywords_list or []))),
                    papers_screened=papers_checked + carry.get("papers_checked", 0),
                    paper_limit=paper_limit,
                    source_counts=_merge_sources(carry.get("sources"), source_counts),
                    date_from=date_from,
                    date_to=date_to,
                    keywords_auto_generated=keywords_auto_generated,
                    warning=_search_warning(failed_keywords)
                    if failed_keywords else None,
                )

        _log.info("[crawl] ✓ success: %d added, %d skipped, %d screened in %.1fs",
                  papers_added, papers_skipped, papers_checked, time.time() - screening_started)

    except LLMUnavailableError as e:
        import traceback
        _log.error("[crawl] LLM unavailable: %s\n%s", e, traceback.format_exc())
        if crawl_log:
            _crawl_current_papers.pop(crawl_log.id, None)
            _crawl_in_flight.pop(crawl_log.id, None)
            # A failed flush/commit above leaves the session unusable until rolled
            # back — without this, the commit() below re-raises the same error and
            # the crawl silently dies with the CrawlLog stuck in "running" forever.
            db.session.rollback()
            crawl_log.status = "error"
            crawl_log.error_message = f"LLM unavailable: {str(e)}"
            crawl_log.finished_at = datetime.utcnow()
            db.session.commit()
    except Exception as e:
        import traceback
        _log.error("[crawl] exception: %s\n%s", e, traceback.format_exc())
        if crawl_log:
            _crawl_current_papers.pop(crawl_log.id, None)
            _crawl_in_flight.pop(crawl_log.id, None)
            db.session.rollback()
            crawl_log.status = "error"
            crawl_log.error_message = str(e)
            crawl_log.finished_at = datetime.utcnow()
            db.session.commit()
            print(f"[CRAWL DEBUG] Saved error status to log")

    crawl_log.failed_keywords = failed_keywords  # transient attr, read by the nightly retry pass
    return crawl_log


def generate_relevance_reason(title: str, abstract: str, research_interest: str) -> str:
    """One sentence on why a paper matters for a research interest — for papers that reach a
    project without being screened (copied/moved from another project). Screened papers get
    theirs from the deep relevance check instead. "" if the LLM answer can't be parsed."""
    prompt = prompts.RELEVANCE_REASON.format(
        research_interest=research_interest, title=title, abstract=abstract or ""
    )
    result, _, _, _ = _llm_json([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=4096)
    if not isinstance(result, dict):
        _log.warning("[relevance-reason] unparseable answer for %r: %r", title[:80], result)
        return ""
    return str(result.get("reason") or "").strip()


def start_relevance_reason_async(project_id: int, paper_id: int, app=None) -> threading.Thread:
    """Fill ProjectPaper.relevance_reason for one link in a background thread (fire-and-forget,
    like retry-web-enrich): the card shows it after the next reload."""
    if app is None:
        from flask import current_app

        app = current_app._get_current_object()

    def _run():
        with app.app_context(), counted(project_id, "relevance_reason"):
            project = db.session.get(Project, project_id)
            pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper_id).first()
            if not project or not pp or not (project.research_interest or "").strip():
                return
            try:
                why = generate_relevance_reason(pp.paper.title, pp.paper.abstract, project.research_interest)
            except Exception:
                _log.exception("[relevance-reason] failed for paper %s in project %s", paper_id, project_id)
                return
            pp.relevance_reason = why or None
            db.session.commit()
            _log.info("[relevance-reason] stored for paper %s in project %s", paper_id, project_id)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


# Re-screen of New Papers, per project, in this (single) web process: project_id -> state dict.
# Polled by the UI through /projects/<slug>/rescreen-status.
_rescreen_tasks: dict[int, dict] = {}


def rescreen_state(project_id: int) -> Optional[dict]:
    return _rescreen_tasks.get(project_id)


def start_rescreen_async(project_id: int, app=None) -> bool:
    """Start run_rescreen in a daemon thread. False if one is already running for the project."""
    state = _rescreen_tasks.get(project_id)
    if state and state["status"] == "running":
        return False
    if app is None:
        from flask import current_app

        app = current_app._get_current_object()
    _rescreen_tasks[project_id] = {
        "status": "running", "total": 0, "checked": 0, "kept": 0, "trashed": 0, "skipped": 0,
        "error": None,
    }

    def _run():
        with app.app_context():
            try:
                run_rescreen(project_id)
            except Exception as e:
                _log.exception("[rescreen] project %s failed", project_id)
                _rescreen_tasks[project_id].update(status="error", error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return True


@counts("rescreen")
def run_rescreen(project_id: int) -> dict:
    """Re-check every New Papers paper against the project's current research interest and My
    Collection, with the same quick → deep screen a crawl uses. Papers that no longer pass go
    to Trash (restorable); papers that pass get a fresh relevance_reason. Papers already judged
    relevant under the current screening context (same hash) are skipped, and so is a paper
    the user collected or trashed while the re-screen ran. A parse failure keeps the paper:
    only a clear "not relevant" moves it to Trash."""
    state = _rescreen_tasks.setdefault(project_id, {
        "status": "running", "total": 0, "checked": 0, "kept": 0, "trashed": 0, "skipped": 0,
        "error": None,
    })
    project = db.session.get(Project, project_id)
    research_interest = project.research_interest or ""
    links = ProjectPaper.query.filter_by(project_id=project_id).all()
    collection = sorted(
        [pp for pp in links if pp.manual_tag and not pp.trashed_at and pp.paper], key=_collection_tag_tier
    )
    ref_dicts = [
        {"title": pp.paper.title, "abstract": (pp.paper.abstract or "")[:400]}
        for pp in _select_reference_papers(collection, limit=REFERENCE_PAPERS_LIMIT)
    ]
    context_hash = compute_screening_hash(research_interest, collection)
    already_relevant = {
        sp.arxiv_id for sp in ScreenedPaper.query.filter_by(
            project_id=project_id, screening_hash=context_hash, is_relevant=True
        ).all()
    }

    todo = []
    for pp in links:
        if pp.manual_tag is not None or pp.trashed_at or not pp.paper:
            continue
        if base_arxiv_id(pp.paper.arxiv_id) in already_relevant:
            state["skipped"] += 1
            continue
        todo.append({
            "link_id": pp.id, "arxiv_id": pp.paper.arxiv_id, "pdf_url": pp.paper.pdf_url,
            "title": pp.paper.title, "abstract": pp.paper.abstract or "",
        })
    state["total"] = len(todo)
    _log.info("[rescreen] project %s: %d New Papers to screen, %d already screened against this context",
              project.name, len(todo), state["skipped"])
    started = time.time()

    def _screen(item):
        relevant, reason, _, _, _ = _screen_paper_quick(
            item["title"], item["abstract"], research_interest, ref_dicts
        )
        if not relevant:
            return item, reason, "", None
        content = get_paper_content(item["arxiv_id"], item["pdf_url"])
        _, reason, _, _, _, why = _screen_paper_deep(
            item["title"], item["abstract"], content or item["abstract"], research_interest, ref_dicts
        )
        return item, reason, why, "full_text" if content else "abstract"

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(contextvars.copy_context().run, _screen, item) for item in todo]  # keeps the usage meter
        try:
            for future in as_completed(futures):
                item, reason, why, source = future.result()
                # Fresh read: the user may have collected/trashed it from the UI meanwhile
                pp = db.session.get(ProjectPaper, item["link_id"], populate_existing=True)
                untouched = pp is not None and pp.manual_tag is None and not pp.trashed_at
                if untouched and reason == "not-relevant":
                    pp.trashed_at = datetime.utcnow()
                    pp.trashed_reason = "Re-screen: not relevant" + (" (from the abstract)" if source == "abstract" else "")
                    if source:
                        pp.screen_source = source
                    state["trashed"] += 1
                    verdict = "trashed"
                elif untouched:
                    if why:
                        pp.relevance_reason = why
                    if source and reason in ("relevant", "not-relevant"):
                        pp.screen_source = source
                    state["kept"] += 1
                    verdict = "kept" if reason == "relevant" else f"kept ({reason})"
                else:
                    verdict = "left alone (collected/trashed meanwhile)"
                if reason in ("relevant", "not-relevant"):
                    db.session.add(ScreenedPaper(
                        project_id=project_id, arxiv_id=base_arxiv_id(item["arxiv_id"]),
                        screening_hash=context_hash, is_relevant=(reason == "relevant"),
                    ))
                db.session.commit()
                state["checked"] += 1
                elapsed = time.time() - started
                eta = elapsed / state["checked"] * (state["total"] - state["checked"])
                _log.info("[rescreen] %d/%d %s: %s (%.0fs elapsed, ~%.0fs left)",
                          state["checked"], state["total"], verdict, item["title"][:80], elapsed, eta)
        except BaseException:
            for f in futures:
                f.cancel()
            raise

    state["status"] = "done"
    _log.info("[rescreen] project %s done: %d screened, %d kept, %d moved to Trash, %d skipped",
              project.name, state["checked"], state["kept"], state["trashed"], state["skipped"])
    return state



def _screening_context(project_id: int) -> tuple[str, list[dict], str, list]:
    """(research_interest, reference papers for the prompt, screening hash, all links) as the crawl uses them."""
    project = db.session.get(Project, project_id)
    links = ProjectPaper.query.filter_by(project_id=project_id).all()
    collection = sorted(
        [pp for pp in links if pp.manual_tag and not pp.trashed_at and pp.paper], key=_collection_tag_tier
    )
    ref_dicts = [
        {"title": pp.paper.title, "abstract": (pp.paper.abstract or "")[:400]}
        for pp in _select_reference_papers(collection, limit=REFERENCE_PAPERS_LIMIT)
    ]
    return project.research_interest or "", ref_dicts, compute_screening_hash(project.research_interest or "", collection), links


def _summarize(paper: Paper, content: str) -> None:
    """Write (or rewrite) the paper's summary + main contributions from content."""
    summary_data, model, elapsed, tokens = generate_summary_with_full_content(
        title=paper.title, abstract=paper.abstract or "", paper_content=content,
    )
    contributions, _, contrib_elapsed, contrib_tokens = extract_main_contributions(
        title=paper.title, paper_content=content, abstract=paper.abstract or "",
    )
    summary_data = summary_data if isinstance(summary_data, dict) else {}
    summary = paper.summary or PaperSummary(paper_id=paper.id)
    summary.content_source = summary_source(content, paper.abstract or "")
    summary.summary_text = summary_data.get("summary", "")
    if contributions:
        summary.main_contributions = contributions
    summary.model_used = model
    for flag in ("introduces_dataset", "introduces_architecture", "introduces_method", "is_survey", "is_benchmark"):
        setattr(summary, flag, summary_data.get(flag, False))
    summary.elapsed_ms = elapsed + contrib_elapsed
    summary.tokens_used = tokens + contrib_tokens
    db.session.add(summary)


@counts("refresh")
def refresh_from_full_text(project_id: int, limit: int = FULL_TEXT_REFRESH_PER_NIGHT) -> dict:
    """Nightly: re-read New Papers whose deep relevance check never saw the full text (screen_source
    "abstract", or "unknown" for papers marked from before this was recorded), newest first, at most
    `limit` per night. With the full text, a clear "not relevant" moves the paper to Trash with a
    reason; a relevant one gets a fresh relevance_reason, and its summary is rewritten if that was not
    made from the full text either. My Collection and Trash are never touched. A paper whose full text
    still can't be fetched (or whose check can't be parsed) is retried on later nights, at most
    MAX_FULL_TEXT_ATTEMPTS times."""
    research_interest, ref_dicts, context_hash, links = _screening_context(project_id)
    todo = sorted(
        [pp for pp in links if pp.manual_tag is None and not pp.trashed_at and pp.paper
         and (pp.paper.is_arxiv or pp.paper.source == "biorxiv") and pp.screen_source in ("abstract", "unknown")
         and (pp.paper.full_text_attempts or 0) < MAX_FULL_TEXT_ATTEMPTS],
        key=lambda pp: pp.added_at or datetime.min, reverse=True,
    )[:limit]
    stats = {"total": len(todo), "trashed": 0, "kept": 0, "no_text": 0, "unclear": 0, "resummarized": 0}
    _log.info("[full-text] project %s: re-reading %d New Papers screened without the full text",
              project_id, len(todo))
    if not todo:
        return stats
    items = [{"link_id": pp.id, "paper_id": pp.paper_id, "arxiv_id": pp.paper.arxiv_id, "pdf_url": pp.paper.pdf_url,
              "title": pp.paper.title, "abstract": pp.paper.abstract or ""} for pp in todo]

    def _check(item):
        content = get_paper_content(item["arxiv_id"], item["pdf_url"])
        if not content:
            return item, None, None, ""
        _, reason, _, _, _, why = _screen_paper_deep(
            item["title"], item["abstract"], content, research_interest, ref_dicts
        )
        return item, content, reason, why

    started, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=4) as executor:  # the fetches queue for arXiv anyway
        futures = [executor.submit(contextvars.copy_context().run, _check, item) for item in items]
        for future in as_completed(futures):
            item, content, reason, why = future.result()
            paper = db.session.get(Paper, item["paper_id"])
            # Fresh read: the user may have collected/trashed it meanwhile
            pp = db.session.get(ProjectPaper, item["link_id"], populate_existing=True)
            if pp is None or pp.manual_tag is not None or pp.trashed_at:
                verdict = "left alone (collected/trashed meanwhile)"
            elif content is None or reason not in ("relevant", "not-relevant"):
                paper.full_text_attempts = (paper.full_text_attempts or 0) + 1
                key = "no_text" if content is None else "unclear"
                stats[key] += 1
                verdict = (f"{'no full text yet' if content is None else 'unclear answer'}, try "
                           f"{paper.full_text_attempts}/{MAX_FULL_TEXT_ATTEMPTS}")
                if paper.full_text_attempts >= MAX_FULL_TEXT_ATTEMPTS:
                    _log.warning("[full-text] giving up on %s after %d tries", item["arxiv_id"], paper.full_text_attempts)
            elif reason == "not-relevant":
                pp.trashed_at = datetime.utcnow()
                pp.trashed_reason = "Not relevant after reading the full text"
                pp.screen_source = "full_text"
                stats["trashed"] += 1
                verdict = "trashed"
            else:
                pp.screen_source = "full_text"
                if why:
                    pp.relevance_reason = why
                stats["kept"] += 1
                verdict = "kept"
                if not paper.summary or paper.summary.content_source != "full_text":
                    _summarize(paper, content)
                    stats["resummarized"] += 1
                    verdict = "kept, summary rewritten"
            if reason in ("relevant", "not-relevant"):
                db.session.add(ScreenedPaper(
                    project_id=project_id, arxiv_id=base_arxiv_id(item["arxiv_id"]),
                    screening_hash=context_hash, is_relevant=(reason == "relevant"),
                ))
            db.session.commit()
            done += 1
            elapsed = time.time() - started
            _log.info("[full-text] %d/%d %s: %s (%.0fs elapsed, ~%.0fs left)", done, len(items), verdict,
                      item["title"][:80], elapsed, elapsed / done * (len(items) - done))
    _log.info("[full-text] project %s done: %s", project_id, stats)
    return stats

def start_crawl_async(
    project_id: int,
    date_from: date,
    date_to: date,
    triggered_by: str = "manual",
    app=None,
    paper_limit: int = 500,
    keywords_override: Optional[list[str]] = None,
    cached_papers: Optional[list[dict]] = None,
    use_arxiv: bool = True,
    use_scholar: bool = True,
    use_citations: bool = True,
    use_biorxiv: bool = False,
):
    """Create a "running" CrawlLog synchronously, then hand off run_crawl to a daemon thread.

    The CrawlLog is created in the calling thread (not the background one) specifically so
    callers get a valid crawl_log_id back immediately — the UI needs it right away to start
    polling /crawl-status before the background thread has necessarily even started.

    Returns (thread, crawl_log_id).
    """
    if app is None:
        from flask import current_app

        app = current_app._get_current_object()

    crawl_log_id = None

    def run_with_context():
        with app.app_context():
            run_crawl(
                project_id,
                date_from,
                date_to,
                triggered_by,
                paper_limit=paper_limit,
                crawl_log_id=crawl_log_id,
                keywords_override=keywords_override,
                cached_papers=cached_papers,
                use_arxiv=use_arxiv,
                use_scholar=use_scholar,
                use_citations=use_citations,
                use_biorxiv=use_biorxiv,
            )

    with app.app_context():
        crawl_log = CrawlLog(
            project_id=project_id,
            triggered_by=triggered_by,
            started_at=datetime.utcnow(),
            status="running",
            date_from=date_from,
            date_to=date_to,
        )
        db.session.add(crawl_log)
        db.session.commit()
        crawl_log_id = crawl_log.id
        print(f"[CRAWL] Created log id={crawl_log_id} in main thread")

    thread = threading.Thread(target=run_with_context)
    thread.daemon = True
    thread.start()
    return thread, crawl_log_id


@counts("metadata", per_project=False)
def backfill_missing_paper_metadata() -> int:
    """Ensure all arXiv papers have complete metadata (title, authors, abstract, citations).

    Pass 1 — enrich_complete=0 papers:
    - If title/authors/abstract all present: flip enrich_complete=1 (no API call).
    - If anything missing: fetch from arXiv; fill gaps; mark complete on success.
    - Placeholder title "arXiv:{id}" is treated as missing.
    - On arXiv failure: leaves enrich_complete=0 so next nightly run retries.

    Pass 2 — citation refresh:
    - Papers with citation_fetched_at IS NULL (fetch previously failed) are re-checked.
    - Papers genuinely without a citation record (never fetched) are also filled.

    Runs synchronously — intended for the nightly cronjob after the main crawl loop.
    Returns number of papers updated across both passes.
    """
    from app.models import Paper
    updated = 0

    # --- Pass 1: fill missing title / authors / abstract ---
    incomplete = Paper.query.filter(
        Paper.enrich_complete == 0,
        ~Paper.arxiv_id.like("web:%"),
    ).all()

    if incomplete:
        _log.info("[backfill] %d paper(s) with enrich_complete=0", len(incomplete))

    for paper in incomplete:
        has_title = bool(paper.title and not paper.title.startswith("arXiv:"))
        has_authors = bool(paper.authors and paper.authors != "[]" and paper.authors != "")
        has_abstract = bool(paper.abstract and paper.abstract.strip())

        if has_title and has_authors and has_abstract:
            paper.enrich_complete = 1
            db.session.commit()
            updated += 1
            continue

        try:
            with _ArxivLockCtx():
                results = list(arxiv.Client(delay_seconds=3.1).results(
                    arxiv.Search(id_list=[paper.arxiv_id])
                ))
            if results:
                r = results[0]
                missing = []
                if not has_title:
                    paper.title = r.title
                    missing.append("title")
                if not has_authors:
                    paper.authors = json.dumps([str(a) for a in r.authors])
                    missing.append("authors")
                if not has_abstract:
                    paper.abstract = r.summary
                    missing.append("abstract")
                paper.enrich_complete = 1
                db.session.commit()
                updated += 1
                _log.info("[backfill] filled %s for %s", "+".join(missing), paper.arxiv_id)
            else:
                _log.warning("[backfill] arXiv returned no result for %s", paper.arxiv_id)
        except Exception as e:
            db.session.rollback()
            _log.warning("[backfill] arXiv fetch failed for %s: %s", paper.arxiv_id, e)

    # --- Pass 2: refresh citations ---
    # Covers papers where:
    #   - citation_fetched_at IS NULL: fetch previously failed (429/timeout) — retry
    #   - citation_count = 0: might be a failed fetch stored as 0 before the None fix,
    #     or a new paper that has since accumulated citations
    # Each paper is tried at most once per CITATION_BACKFILL_COOLDOWN_DAYS, least recently tried first.
    from sqlalchemy import or_
    cooldown_cutoff = datetime.utcnow() - timedelta(days=CITATION_BACKFILL_COOLDOWN_DAYS)
    needs_citation = Paper.query.filter(
        ~Paper.arxiv_id.like("web:%"),
        or_(Paper.citation_fetched_at.is_(None), Paper.citation_count == 0),
        or_(Paper.citation_attempted_at.is_(None), Paper.citation_attempted_at < cooldown_cutoff),
    ).order_by(Paper.citation_attempted_at.asc()).limit(SCHOLAR_BATCH_SIZE).all()  # SQLite sorts NULL first

    if needs_citation:
        counts = get_citation_counts([f"arXiv:{base_arxiv_id(p.arxiv_id)}" for p in needs_citation])
        now = datetime.utcnow()
        for paper in needs_citation if counts else []:  # no answer at all: try them all again next run
            paper.citation_attempted_at = now
            count = counts.get(f"arXiv:{base_arxiv_id(paper.arxiv_id)}")
            if count is None:
                continue
            if not paper.citation_manual or count >= paper.citation_count:
                paper.citation_count = count
                paper.citation_manual = False
            paper.citation_fetched_at = now
            updated += 1
        db.session.commit()
        _log.info("[backfill] citation refresh: %d/%d paper(s) answered", len(counts), len(needs_citation))

    # --- Pass 2b: papers added by hand from a bioRxiv link become full bioRxiv papers (upgrade_biorxiv_paper) ---
    # One try a day each (web_enrich_last_attempt_at): one that bioRxiv doesn't know is logged and kept as it is
    hand_added_biorxiv = Paper.query.filter(
        Paper.arxiv_id.like("web:%"),
        or_(Paper.source.is_(None), Paper.source == "web"),
        Paper.pdf_url.like("%biorxiv.org/%"),
        or_(Paper.web_enrich_last_attempt_at.is_(None),
            Paper.web_enrich_last_attempt_at < datetime.utcnow() - timedelta(hours=20)),
    ).limit(20).all()
    for paper in hand_added_biorxiv:
        paper.web_enrich_last_attempt_at = datetime.utcnow()
        updated += upgrade_biorxiv_paper(paper)
        db.session.commit()

    # --- Pass 3: web-paper PDF metadata (direct PDF link, page count, institutions) ---
    # Targets papers whose pdf_url is still just the landing page — either added before
    # this extraction existed, or where it previously failed (no citation_pdf_url found,
    # PDF fetch failed, etc.). Re-running is safe/idempotent: extract_paper_info_from_url
    # only ever gets fed a landing-page-style URL here, never an already-resolved PDF link.
    # Capped at MAX_WEB_ENRICH_ATTEMPTS per paper — without this, a permanently failing fetch
    # (403/bot-block, dead link) got re-tried on every one of the 4 nightly cronjob runs
    # forever, hammering the same publisher URL indefinitely. A paper that hits the cap can
    # still be manually retried from its card in the UI (resets the counter to 0).
    needs_web_enrich = Paper.query.filter(
        Paper.arxiv_id.like("web:%"),
        or_(Paper.source.is_(None), Paper.source == "web"),  # added by hand; not crawl finds (bioRxiv, journal)
        or_(Paper.page_count.is_(None), Paper.institutions.is_(None)),
        Paper.web_enrich_attempts < MAX_WEB_ENRICH_ATTEMPTS,
    ).limit(20).all()

    if needs_web_enrich:
        _log.info("[backfill] %d web paper(s) need PDF metadata (capped at 20/run)", len(needs_web_enrich))

    for paper in needs_web_enrich:
        _enrich_one_web_paper(paper)
        db.session.commit()
        updated += 1
        _log.info("[backfill] web paper %s: page_count=%s institutions=%s (attempt %d/%d)",
                  paper.arxiv_id, paper.page_count, paper.institutions,
                  paper.web_enrich_attempts, MAX_WEB_ENRICH_ATTEMPTS)
        if paper.web_enrich_attempts >= MAX_WEB_ENRICH_ATTEMPTS and (
            paper.page_count is None or paper.institutions is None
        ):
            _log.warning(
                "[backfill] web paper %s giving up after %d failed attempts (page_count=%s institutions=%s)",
                paper.arxiv_id, paper.web_enrich_attempts, paper.page_count, paper.institutions,
            )

    # --- Pass 4: figure 1 of bioRxiv papers, once bioRxiv has added the HTML version ---
    # The crawl usually sees a preprint before its HTML (with the figures) exists, 4 to 6 days after posting.
    # Tried from 4 to BIORXIV_FIGURE_DAYS days after posting, at most once a day (this runs after each of the
    # nightly runs; web_enrich_last_attempt_at is free for bioRxiv papers, Pass 3 never takes them).
    today = date.today()
    needs_figure = Paper.query.filter(
        Paper.source == "biorxiv",
        Paper.figure_url.is_(None),
        Paper.published_date <= today - timedelta(days=4),
        Paper.published_date >= today - timedelta(days=BIORXIV_FIGURE_DAYS),
        or_(Paper.web_enrich_last_attempt_at.is_(None),
            Paper.web_enrich_last_attempt_at < datetime.utcnow() - timedelta(hours=20)),
    ).limit(20).all()
    if needs_figure:
        _log.info("[backfill] %d bioRxiv paper(s) without figure 1 (capped at 20/run)", len(needs_figure))
    for paper in needs_figure:
        paper.web_enrich_last_attempt_at = datetime.utcnow()
        paper.figure_url, paper.figure_caption = fetch_biorxiv_figure(paper.pdf_url or "")
        db.session.commit()
        updated += bool(paper.figure_url)
        _log.info("[backfill] bioRxiv %s: %s", paper.arxiv_id, "figure 1 added" if paper.figure_url else "no HTML yet")

    return updated
