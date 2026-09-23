#!/usr/bin/env python3
"""
Nightly crawl entrypoint for CronJob.
Loops through all projects and crawls for new papers.
Uses project.saved_keywords if set, otherwise generates keywords fresh via LLM.
"""

import json
import logging
import os
import sys
from datetime import timedelta, datetime, timezone
from logging.handlers import RotatingFileHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env for local dev (no-op if python-dotenv not installed or .env absent)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app import create_app, db
from app.models import Project, Paper, ProjectPaper
from app.crawl import run_crawl, backfill_missing_paper_metadata, get_citation_count, get_citation_count_by_scholar_id, send_notification

_log = logging.getLogger(__name__)


def _setup_crawl_log():
    logs_dir = os.environ.get("LOGS_DIR", "/data/logs")
    os.makedirs(logs_dir, exist_ok=True)
    handler = RotatingFileHandler(
        f"{logs_dir}/crawl.log", maxBytes=10 * 1024 * 1024, backupCount=10
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
    ))
    logging.getLogger().addHandler(handler)
    # Suppress noisy third-party library logs
    for noisy in ("arxiv", "httpx", "httpcore", "urllib3", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _refresh_collection_citations():
    """Update citation counts for papers in any My Collection that are >30 days stale."""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)

    collection_paper_ids = {
        row[0] for row in
        db.session.query(ProjectPaper.paper_id).filter(
            ProjectPaper.manual_tag.isnot(None),
            ProjectPaper.trashed_at.is_(None),
        ).distinct().all()
    }

    if not collection_paper_ids:
        _log.info("[nightly] citation refresh: no collection papers found")
        return 0

    papers = Paper.query.filter(
        Paper.id.in_(collection_paper_ids),
        db.or_(
            Paper.source == "arxiv",
            Paper.semantic_scholar_id.isnot(None),
        ),
        db.or_(
            Paper.citation_fetched_at.is_(None),
            Paper.citation_fetched_at < cutoff,
        ),
    ).all()

    _log.info("[nightly] citation refresh: %d paper(s) due (stale > 30 days)", len(papers))
    updated = 0
    for paper in papers:
        try:
            if paper.source == "arxiv" and paper.arxiv_id:
                count = get_citation_count(paper.arxiv_id)
            elif paper.semantic_scholar_id:
                count = get_citation_count_by_scholar_id(paper.semantic_scholar_id)
            else:
                continue
            if count is not None:
                if paper.citation_manual and count < paper.citation_count:
                    # Scholar returned fewer than the manually set value — keep it; just refresh timestamp
                    _log.info("[nightly] citation refresh: keeping manual override for %s (%d > %d from Scholar)",
                              paper.arxiv_id or paper.semantic_scholar_id, paper.citation_count, count)
                else:
                    paper.citation_count = count
                    paper.citation_manual = False
                paper.citation_fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
                updated += 1
        except Exception as e:
            _log.warning("[nightly] citation refresh failed for %s: %s", paper.arxiv_id or paper.semantic_scholar_id, e)

    db.session.commit()
    _log.info("[nightly] citation refresh: %d updated", updated)
    return updated


def nightly_crawl():
    _setup_crawl_log()
    app = create_app()
    current_hour = datetime.now(timezone.utc).hour

    with app.app_context():
        all_projects = Project.query.filter_by(trashed_at=None).all()

        hour_override = os.environ.get("CRAWL_HOUR_OVERRIDE", "").strip().lower()
        if hour_override == "all":
            projects = [p for p in all_projects if p.crawl_hour is not None]
            _log.info("[nightly] CRAWL_HOUR_OVERRIDE=all — crawling all %d enabled project(s)", len(projects))
        else:
            projects = [p for p in all_projects if p.crawl_hour == current_hour]
            _log.info("[nightly] Hour %02d:00 UTC — %d/%d project(s) scheduled now", current_hour, len(projects), len(all_projects))

        for idx, project in enumerate(projects, 1):
            keywords_override = project.saved_keywords_list if project.saved_keywords else None
            _log.info("[nightly] ── project %d/%d: %s ──────────────────────", idx, len(projects), project.name)
            if keywords_override:
                _log.info("[nightly] using saved keywords (%d): %s", len(keywords_override), keywords_override)
            else:
                _log.info("[nightly] no saved keywords — will generate via LLM")

            # Skip and notify if the project has no research interest and no keywords —
            # run_crawl would raise ValueError, and generating keywords without any input is meaningless.
            if not project.research_interest and not keywords_override:
                _log.warning("[nightly] SKIP %s — no research interest and no keywords", project.name)
                if project.notification_emails:
                    server_url = os.environ.get("SERVER_URL", "http://localhost:5001")
                    project_url = f"{server_url}/projects/{project.slug}"
                    send_notification(
                        subject=f"[{project.name}] Nightly crawl skipped — setup required",
                        message="",
                        project_emails=project.notification_emails,
                        content_html=(
                            f'<p style="margin:0 0 10px 0;">The nightly crawl for project '
                            f'<strong>{project.name}</strong> was skipped because no research interest '
                            f'and no search keywords are set.</p>'
                            f'<p style="margin:0 0 10px 0;">Please visit your project to complete setup:</p>'
                            f'<p style="margin:0;"><a href="{project_url}" '
                            f'style="color:#7c3aed;">{project_url}</a></p>'
                        ),
                    )
                continue

            days_back = int(os.environ.get("CRAWL_DAYS_BACK", "1"))
            today_utc = datetime.now(timezone.utc).date()
            date_from = today_utc - timedelta(days=days_back)
            date_to = today_utc

            crawl_log = run_crawl(
                project.id,
                date_from,
                date_to,
                triggered_by="cron",
                keywords_override=keywords_override,
            )
            if crawl_log and crawl_log.status == "error":
                _log.error("[nightly] ✗ FAILED: %s — %s", project.name, crawl_log.error_message)
                if project.notification_emails:
                    server_url = os.environ.get("SERVER_URL", "http://localhost:5001")
                    project_url = f"{server_url}/projects/{project.slug}"
                    send_notification(
                        subject=f"[{project.name}] Nightly crawl failed",
                        message="",
                        project_emails=project.notification_emails,
                        content_html=(
                            f'<p style="margin:0 0 10px 0;">The nightly crawl for project '
                            f'<strong>{project.name}</strong> failed:</p>'
                            f'<p style="margin:0 0 10px 0; font-family:monospace; font-size:0.85em;">'
                            f'{crawl_log.error_message or "unknown error"}</p>'
                            f'<p style="margin:0;"><a href="{project_url}" '
                            f'style="color:#7c3aed;">{project_url}</a></p>'
                        ),
                    )
            else:
                _log.info("[nightly] ✓ done: %s", project.name)

        _log.info("[nightly] ── backfill: enriching incomplete papers ──────────────────────")
        filled = backfill_missing_paper_metadata()
        _log.info("[nightly] backfill complete: %d paper(s) updated", filled)

        _log.info("[nightly] ── citation refresh: updating stale collection papers ──────────────────────")
        _refresh_collection_citations()


if __name__ == "__main__":
    nightly_crawl()
