"""A paper's arXiv version, when that version came out, the authors' comment and the journal reference, from the
arXiv API, for the card (redesign 2026-11-12). Kept fresh by a nightly pass; a new version means the paper may now
link its code or name its venue, so app/resources.py reads it again."""
import logging
import re
from datetime import datetime, timedelta

import arxiv

from app import db
from app.models import Paper, ProjectPaper
from app.utils import _ArxivLockCtx, base_arxiv_id

_log = logging.getLogger(__name__)

META_REFRESH_DAYS = 7  # a paper's arXiv data is asked again after this long
META_BATCH = 100  # ids per arXiv API request (one page)
META_BATCHES_PER_RUN = 3  # requests per nightly run (4 runs a night)


def from_result(r) -> dict:
    """The fields we keep from an arxiv.Result."""
    m = re.search(r"v(\d+)$", r.entry_id or "")
    updated = getattr(r, "updated", None)
    return {
        "arxiv_version": int(m.group(1)) if m else None,
        "arxiv_updated": updated.date() if updated and updated.year > 1 else None,
        "arxiv_comment": (getattr(r, "comment", None) or "").strip() or None,
        "journal_ref": (getattr(r, "journal_ref", None) or "").strip() or None,
    }


def apply(paper, meta: dict) -> None:
    """Store meta on the paper (caller commits). A version newer than the one stored clears resources_checked_at,
    so the nightly backfill reads the paper's links and venue again."""
    old = paper.arxiv_version
    if old and meta["arxiv_version"] and meta["arxiv_version"] > old:
        paper.resources_checked_at = None
        paper.references_fetched_at = None  # its references may have changed too
        _log.info("[arxiv-meta] %s: new version v%d (was v%d)", paper.arxiv_id, meta["arxiv_version"], old)
    for k, v in meta.items():
        setattr(paper, k, v)
    paper.arxiv_meta_at = datetime.utcnow()


def _due(limit: int) -> list:
    """arXiv papers someone can see (in a project, not trashed there) whose arXiv data is missing or older than
    META_REFRESH_DAYS, never asked first, then the longest ago."""
    in_use = db.session.query(ProjectPaper.paper_id).filter(ProjectPaper.trashed_at.is_(None))
    stale = datetime.utcnow() - timedelta(days=META_REFRESH_DAYS)
    return (Paper.query.filter(~Paper.arxiv_id.like("web:%"), Paper.id.in_(in_use),
                               (Paper.arxiv_meta_at.is_(None)) | (Paper.arxiv_meta_at < stale))
            .order_by(Paper.arxiv_meta_at.isnot(None), Paper.arxiv_meta_at, Paper.id.desc())
            .limit(limit).all())


def _fetch(ids: list[str]) -> dict:
    """base arXiv id -> meta for one batch (one request)."""
    with _ArxivLockCtx():
        client = arxiv.Client(page_size=len(ids), num_retries=1, delay_seconds=3.1)
        results = list(client.results(arxiv.Search(id_list=ids, max_results=len(ids))))
    return {base_arxiv_id(r.entry_id.split("/abs/")[-1]): from_result(r) for r in results}


def refresh(batches: int = META_BATCHES_PER_RUN) -> int:
    """Nightly: arXiv data for papers that have none or old data, META_BATCH per request. Stops at the first failed
    request (arXiv answers 429 at times; the next run goes on). Returns how many papers were updated."""
    updated = 0
    for _ in range(batches):
        papers = _due(META_BATCH)
        if not papers:
            break
        try:
            metas = _fetch([p.arxiv_id for p in papers])
        except Exception as e:
            _log.warning("[arxiv-meta] request failed, next run goes on: %s", e)
            break
        for p in papers:
            meta = metas.get(base_arxiv_id(p.arxiv_id))
            if meta:
                apply(p, meta)
                updated += 1
            else:  # no answer for this id (withdrawn, or a bad id): ask again in META_REFRESH_DAYS
                p.arxiv_meta_at = datetime.utcnow()
        db.session.commit()
        _log.info("[arxiv-meta] %d/%d papers updated in this batch", len(metas), len(papers))
        if len(papers) < META_BATCH:
            break
    return updated
