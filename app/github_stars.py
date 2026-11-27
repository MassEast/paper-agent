"""GitHub stars of a paper's code repository (Paper.code_url), for the card's Code button and the "Stars" sort.

Nightly (backfill Pass 10): papers someone can see whose code_url is a github.com repo, never checked or checked
more than STARS_MAX_AGE_DAYS ago, the oldest check first. Without GITHUB_TOKEN GitHub allows 60 requests an hour
per IP, so a run stops when the answer says none are left (X-RateLimit-Remaining) and the next run goes on; a token
(no permissions needed) raises it to 5000.
"""
import logging
import os
import re
from datetime import datetime, timedelta

import httpx

from app import db
from app.models import Paper, ProjectPaper
from app.utils import USER_AGENT

_log = logging.getLogger(__name__)

STARS_MAX_AGE_DAYS = 7
STARS_PER_RUN = int(os.environ.get("GITHUB_STARS_PER_RUN", "50"))  # 4 runs a night
_REPO = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)", re.I)
_NOT_OWNERS = {"orgs", "sponsors", "topics", "features", "marketplace", "settings", "apps"}


def repo_of(code_url: str | None) -> str | None:
    """"owner/repo" of a github.com repository link, else None (a user page, github.io, gitlab, ...)."""
    m = _REPO.search(code_url or "")
    if not m or m.group(1).lower() in _NOT_OWNERS:
        return None
    return m.group(1) + "/" + re.sub(r"\.git$", "", m.group(2))


def _headers() -> dict:
    h = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    return h


def fetch_one(paper) -> None:
    """Stars for one paper right away (a code link added by hand). On a rate limit or network error nothing is stored,
    so the nightly refresh picks it up; the caller commits."""
    repo = repo_of(paper.code_url)
    if not repo:
        return
    try:
        r = httpx.get(f"https://api.github.com/repos/{repo}", headers=_headers(), timeout=10.0, follow_redirects=True)
    except httpx.HTTPError as e:
        _log.warning("[github] %s: %s, left for the nightly refresh", repo, e)
        return
    if r.status_code in (200, 404):
        paper.github_checked_at = datetime.utcnow()
        paper.github_stars = r.json()["stargazers_count"] if r.status_code == 200 else None
    else:
        _log.warning("[github] %s answered %d, left for the nightly refresh", repo, r.status_code)


def refresh(limit: int = STARS_PER_RUN) -> int:
    """Fetch stars for up to `limit` due papers. Returns how many got a count."""
    in_use = db.session.query(ProjectPaper.paper_id).filter(ProjectPaper.trashed_at.is_(None))
    cutoff = datetime.utcnow() - timedelta(days=STARS_MAX_AGE_DAYS)
    due = [p for p in Paper.query.filter(
        Paper.id.in_(in_use), Paper.code_url.ilike("%github.com/%"),
        db.or_(Paper.github_checked_at.is_(None), Paper.github_checked_at < cutoff),
    ).order_by(Paper.github_checked_at.asc()).all() if repo_of(p.code_url)][:limit]  # SQLite sorts NULL first
    if not due:
        return 0
    _log.info("[github] stars for %d repo(s) (capped at %d/run)", len(due), limit)
    got = 0
    with httpx.Client(headers=_headers(), timeout=20.0, follow_redirects=True) as client:  # renamed repos redirect
        for i, paper in enumerate(due, 1):
            repo = repo_of(paper.code_url)
            try:
                r = client.get(f"https://api.github.com/repos/{repo}")
            except httpx.HTTPError as e:
                _log.warning("[github] %s: %s, stopping for this run", repo, e)
                break
            if r.status_code in (403, 429):
                _log.info("[github] rate limit reached after %d/%d (resets %s), the next run goes on",
                          i - 1, len(due), r.headers.get("X-RateLimit-Reset"))
                break
            paper.github_checked_at = datetime.utcnow()
            if r.status_code == 200:
                paper.github_stars = r.json()["stargazers_count"]
                got += 1
            elif r.status_code == 404:  # deleted, private or a wrong link: no stars to show
                paper.github_stars = None
            else:
                _log.warning("[github] %s answered %d", repo, r.status_code)
            db.session.commit()
            if r.headers.get("X-RateLimit-Remaining") == "0":
                _log.info("[github] rate limit used up after %d/%d, the next run goes on", i, len(due))
                break
    _log.info("[github] stars: %d/%d repo(s) answered", got, len(due))
    return got
