"""Backfill Pass 2 (citation refresh): one batch request, per-paper cooldown, nothing marked when Semantic Scholar
gave no answer at all."""
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import patch

import app.crawl as crawl


@contextmanager
def _offline_backfill(citation_side_effect):
    # Pass 1/3 may pick up unrelated papers from the session-scoped DB — keep them offline and instant.
    with patch("app.crawl._ArxivLockCtx"), patch("app.crawl.time.sleep"), \
            patch("app.crawl.arxiv.Client") as mock_arxiv_client, \
            patch("app.crawl.extract_paper_info_from_url", return_value={"pdf_url": None}), \
            patch("app.crawl.get_citation_counts", side_effect=citation_side_effect) as mock_count:
        mock_arxiv_client.return_value.results.return_value = []
        yield mock_count


def _add(arxiv_id, attempted_at):
    from app import db
    from app.models import Paper
    p = Paper(arxiv_id=arxiv_id, title=arxiv_id, authors='["A"]', abstract="x", enrich_complete=1,
              citation_count=0, citation_fetched_at=None, citation_attempted_at=attempted_at)
    db.session.add(p)
    db.session.commit()
    return p.id


def test_citation_backfill_respects_cooldown(app):
    from app import db
    from app.models import Paper
    with app.app_context():
        recent = _add("2699.00001", datetime.utcnow() - timedelta(days=1))
        stale = _add("2699.00002", datetime.utcnow() - timedelta(days=crawl.CITATION_BACKFILL_COOLDOWN_DAYS + 1))
        never = _add("2699.00003", None)
        try:
            with _offline_backfill(lambda refs: {"arXiv:2699.00002": 4}) as mock_count:
                crawl.backfill_missing_paper_metadata()
            mock_count.assert_called_once()
            fetched = set(mock_count.call_args.args[0])
            assert "arXiv:2699.00001" not in fetched
            assert {"arXiv:2699.00002", "arXiv:2699.00003"} <= fetched
            assert Paper.query.get(stale).citation_count == 4
            # unknown to Semantic Scholar still counts as an attempt, so it waits out the cooldown too
            assert Paper.query.get(never).citation_attempted_at > datetime.utcnow() - timedelta(minutes=1)
            assert Paper.query.get(never).citation_fetched_at is None
        finally:
            for pid in (recent, stale, never):
                db.session.delete(Paper.query.get(pid))
            db.session.commit()


def test_citation_backfill_marks_nothing_without_any_answer(app):
    from app import db
    from app.models import Paper
    with app.app_context():
        pid = _add("2699.00004", None)
        try:
            with _offline_backfill(lambda refs: {}):
                crawl.backfill_missing_paper_metadata()
            assert Paper.query.get(pid).citation_attempted_at is None  # not marked: tried again next run
        finally:
            db.session.delete(Paper.query.get(pid))
            db.session.commit()
