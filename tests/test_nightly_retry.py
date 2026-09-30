import importlib.util
import pathlib
from types import SimpleNamespace
from unittest.mock import patch

from app import db
from app.models import Project

_spec = importlib.util.spec_from_file_location(
    "nightly_crawl_script", pathlib.Path(__file__).parent.parent / "scripts" / "nightly_crawl.py"
)
nightly = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nightly)


def _run(app, results_by_call):
    """Run nightly_crawl() with one project and run_crawl returning `results_by_call` in order."""
    with app.app_context():
        for p in Project.query.all():
            p.crawl_hour = None
        proj = Project(name="Nightly Retry", slug="nightly-retry", research_interest="x",
                       saved_keywords='["a", "b"]', crawl_hour=1, notification_emails='["x@y.z"]')
        db.session.add(proj)
        db.session.commit()
    calls = []

    def fake_run_crawl(project_id, d_from, d_to, triggered_by=None, keywords_override=None, **kw):
        calls.append(keywords_override)
        return results_by_call[len(calls) - 1]

    try:
        with patch.object(nightly, "create_app", return_value=app), \
             patch.object(nightly, "_setup_crawl_log"), \
             patch.object(nightly, "run_crawl", side_effect=fake_run_crawl), \
             patch.object(nightly, "backfill_missing_paper_metadata", return_value=0), \
             patch.object(nightly, "_refresh_collection_citations"), \
             patch.object(nightly, "send_notification") as notify, \
             patch.object(nightly.time, "sleep") as sleep, \
             patch.dict("os.environ", {"CRAWL_HOUR_OVERRIDE": "all"}):
            nightly.nightly_crawl()
    finally:
        with app.app_context():
            Project.query.filter_by(slug="nightly-retry").delete()
            db.session.commit()
    return calls, notify, sleep


def _log(status, failed=(), msg=None, added=0):
    return SimpleNamespace(status=status, failed_keywords=list(failed), error_message=msg, papers_added=added)


def test_failed_keywords_are_retried_once_at_end_and_no_email_if_retry_works(app):
    calls, notify, sleep = _run(app, [_log("error", ["a", "b"], "all failed"), _log("success", added=3)])
    assert calls == [["a", "b"], ["a", "b"]]
    sleep.assert_called_once()
    notify.assert_not_called()


def test_retry_that_fails_again_emails(app):
    calls, notify, _ = _run(app, [_log("success", ["b"], "1/2 failed"), _log("success", ["b"], "1/2 failed")])
    assert calls == [["a", "b"], ["b"]]
    notify.assert_called_once()


def test_quiet_day_has_no_retry(app):
    calls, notify, sleep = _run(app, [_log("success")])
    assert len(calls) == 1
    sleep.assert_not_called()
    notify.assert_not_called()
