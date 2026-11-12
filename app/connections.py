"""How a paper connects to the rest of the project, for the card's sidebar (redesign 2026-11-12): the My Collection
papers it was found through (it cites them), other My Collection papers it cites, authors who also wrote a My
Collection paper, and the other projects it is in. The paper's references come from Semantic Scholar, fetched by
a nightly pass (fetch_references) into PaperReference."""
import logging
import re
from datetime import datetime, timedelta

from flask import has_request_context, request

from app import db
from app.models import Paper, PaperReference, Project, ProjectPaper
from app.utils import base_arxiv_id

_log = logging.getLogger(__name__)

REFERENCES_BATCH = 100  # papers per Semantic Scholar batch request
REFERENCES_BATCHES_PER_RUN = 2
REFERENCES_RETRY_DAYS = 3  # Semantic Scholar often has a new paper's references only after a few days
MAX_SHOWN = 3  # papers listed per line on the card; the rest as "+N"


def _doi_key(doi: str) -> str:
    doi = doi.strip().lower()
    m = re.fullmatch(r"10\.48550/arxiv\.(.+)", doi)  # arXiv's own DOI is the arXiv paper
    return f"arxiv:{base_arxiv_id(m.group(1))}" if m else f"doi:{doi}"


def paper_keys(paper) -> set[str]:
    """The keys a reference to this paper can carry (see PaperReference)."""
    keys = set()
    if paper.is_arxiv:
        keys.add(f"arxiv:{base_arxiv_id(paper.arxiv_id)}")
    if paper.doi:
        keys.add(_doi_key(paper.doi))
    if paper.semantic_scholar_id:
        keys.add(f"s2:{paper.semantic_scholar_id}")
    return keys


def reference_keys(item: dict) -> set[str]:
    """The keys of one reference from Semantic Scholar ({"paperId", "externalIds"})."""
    ext = item.get("externalIds") or {}
    keys = set()
    if ext.get("ArXiv"):
        keys.add(f"arxiv:{base_arxiv_id(ext['ArXiv'])}")
    if ext.get("DOI"):
        keys.add(_doi_key(ext["DOI"]))
    if item.get("paperId"):
        keys.add(f"s2:{item['paperId']}")
    return keys


def _references_due(limit: int) -> list:
    """Papers someone can see whose references we don't have, not tried in the last REFERENCES_RETRY_DAYS, newest
    first, that Semantic Scholar can know (arXiv id, DOI or Semantic Scholar id)."""
    in_use = db.session.query(ProjectPaper.paper_id).filter(ProjectPaper.trashed_at.is_(None))
    retry = datetime.utcnow() - timedelta(days=REFERENCES_RETRY_DAYS)
    return (Paper.query.filter(Paper.id.in_(in_use), Paper.references_fetched_at.is_(None),
                               (Paper.references_attempted_at.is_(None)) | (Paper.references_attempted_at < retry),
                               (~Paper.arxiv_id.like("web:%")) | Paper.doi.isnot(None)
                               | Paper.semantic_scholar_id.isnot(None))
            .order_by(Paper.id.desc()).limit(limit).all())


def store_references(paper, refs: list[dict]) -> None:
    """Replace the paper's stored references (caller commits)."""
    PaperReference.query.filter_by(paper_id=paper.id).delete()
    for key in sorted(set().union(*(reference_keys(r) for r in refs if r))):
        db.session.add(PaperReference(paper_id=paper.id, ref_key=key))
    paper.references_fetched_at = datetime.utcnow()


def fetch_references(batches: int = REFERENCES_BATCHES_PER_RUN) -> int:
    """Nightly: references from Semantic Scholar for papers without them. Returns how many papers got them."""
    from app.crawl import _scholar_call, _scholar_ref, scholar_seed
    done = 0
    for _ in range(batches):
        papers = _references_due(REFERENCES_BATCH)
        if not papers:
            break
        ids = [_scholar_ref(scholar_seed(p)) for p in papers]
        try:
            r = _scholar_call("POST", "https://api.semanticscholar.org/graph/v1/paper/batch",
                              params={"fields": "references.externalIds"}, json={"ids": ids}, timeout=60.0)
        except Exception as e:
            _log.warning("[references] Semantic Scholar request failed, next run goes on: %s", e)
            break
        if r.status_code != 200:
            _log.warning("[references] Semantic Scholar answered %d, next run goes on: %s", r.status_code, r.text[:200])
            break
        now = datetime.utcnow()
        for paper, item in zip(papers, r.json()):
            paper.references_attempted_at = now
            refs = (item or {}).get("references") or []
            if refs:
                store_references(paper, refs)
                done += 1
        db.session.commit()
        _log.info("[references] %d/%d papers got their references", done, len(papers))
        if len(papers) < REFERENCES_BATCH:
            break
    return done


def _norm_author(name: str) -> str:
    return re.sub(r"[^a-z ]", "", re.sub(r"\s+", " ", (name or "").lower())).strip()


def _collection_index(project_id: int) -> dict:
    """The project's My Collection, indexed by reference key and by author, once per request (kept on the request,
    not on flask.g, which can outlive it and would keep stale rows alive)."""
    cache = request.environ.setdefault("rw.connections", {}) if has_request_context() else {}
    if project_id not in cache:
        pps = (ProjectPaper.query.filter_by(project_id=project_id, manual_tag="related")
               .filter(ProjectPaper.trashed_at.is_(None)).all())
        by_key, by_author = {}, {}
        for cp in pps:
            if not cp.paper:
                continue
            for k in paper_keys(cp.paper):
                by_key[k] = cp.paper
            for a in cp.paper.authors_list:
                if _norm_author(a):
                    by_author.setdefault(_norm_author(a), []).append(cp.paper)
        cache[project_id] = {"by_key": by_key, "by_author": by_author, "ids": {cp.paper_id for cp in pps}}
    return cache[project_id]


def card_connections(pp) -> dict:
    """For the card: found_via [papers or {"title"}], also_cites [papers], shared_authors [([names], [papers])]
    (authors who share the same papers in one entry),
    other_projects [(project, "My Collection" | "New Papers")]. Papers are Paper rows (link to their card)."""
    idx = _collection_index(pp.project_id)
    paper = pp.paper
    found, found_ids = [], set()
    for v in pp.found_via_list:
        if v.get("via") != "cites":
            continue
        key = (f"arxiv:{base_arxiv_id(v['arxiv_id'])}" if v.get("arxiv_id")
               else _doi_key(v["seed"][4:]) if (v.get("seed") or "").startswith("DOI:")
               else f"s2:{v['seed']}" if v.get("seed") else None)
        hit = idx["by_key"].get(key) if key else None
        found.append(hit or {"title": v.get("title") or "a paper in My Collection"})
        if hit:
            found_ids.add(hit.id)
    refs = {r.ref_key for r in PaperReference.query.filter_by(paper_id=paper.id).all()}
    also = {}
    for k in refs:
        hit = idx["by_key"].get(k)
        if hit and hit.id != paper.id and hit.id not in found_ids:
            also[hit.id] = hit
    # Authors who wrote the same My Collection papers go in one line ("A, B and C also wrote X"), in author order
    groups = {}
    for a in paper.authors_list:
        others = list({p.id: p for p in idx["by_author"].get(_norm_author(a), []) if p.id != paper.id}.values())
        if others:
            groups.setdefault(tuple(sorted(p.id for p in others)), ([], others))[0].append(a)
    shared = list(groups.values())
    other_projects = []
    for op in (ProjectPaper.query.filter(ProjectPaper.paper_id == paper.id, ProjectPaper.project_id != pp.project_id,
                                         ProjectPaper.trashed_at.is_(None)).all()):
        proj = db.session.get(Project, op.project_id)
        if proj and not proj.trashed_at:
            other_projects.append((proj, "My Collection" if op.manual_tag else "New Papers"))
    return {"found_via": found, "also_cites": sorted(also.values(), key=lambda p: p.title or ""),
            "shared_authors": shared, "other_projects": other_projects}
