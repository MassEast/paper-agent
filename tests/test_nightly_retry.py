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
    kwargs_seen = []

    def fake_run_crawl(project_id, d_from, d_to, triggered_by=None, keywords_override=None, **kw):
        calls.append(keywords_override)
        kwargs_seen.append(kw)
        return results_by_call[len(calls) - 1]

    try:
        with patch.object(nightly, "create_app", return_value=app), \
             patch.object(nightly, "_setup_crawl_log"), \
             patch.object(nightly, "run_crawl", side_effect=fake_run_crawl), \
             patch.object(nightly, "backfill_missing_paper_metadata", return_value=0), \
             patch.object(nightly, "_refresh_collection_citations"), \
             patch.object(nightly, "send_notification") as notify, \
             patch.object(nightly, "send_carry_notification") as carry_notify, \
             patch.object(nightly.time, "sleep") as sleep, \
             patch.dict("os.environ", {"CRAWL_HOUR_OVERRIDE": "all"}):
            nightly.nightly_crawl()
    finally:
        with app.app_context():
            Project.query.filter_by(slug="nightly-retry").delete()
            db.session.commit()
    notify.kwargs_seen = kwargs_seen
    notify.carry_notify = carry_notify
    return calls, notify, sleep


def _log(status, failed=(), msg=None, added=0):
    return SimpleNamespace(status=status, failed_keywords=list(failed), error_message=msg, papers_added=added)


def test_failed_keywords_are_retried_once_at_end_and_no_email_if_retry_works(app):
    calls, notify, sleep = _run(app, [_log("error", ["a", "b"], "all failed"), _log("success", added=3)])
    assert calls == [["a", "b"], ["a", "b"]]
    # the retry is arXiv only: the first pass already screened the recommendations and citing papers
    assert notify.kwargs_seen[1]["use_scholar"] is False and notify.kwargs_seen[1]["use_citations"] is False
    sleep.assert_called_once()
    notify.assert_not_called()


def test_search_failing_all_night_is_retried_twice_then_says_so_honestly(app):
    fail = lambda: _log("error", ["a", "b"], "all failed")
    calls, notify, sleep = _run(app, [fail(), fail(), fail()])
    assert calls == [["a", "b"]] * 3
    assert sleep.call_count == 2
    notify.assert_called_once()
    assert "arXiv search could not run tonight" in notify.call_args.kwargs["subject"]
    assert "next nights cover these days again" in notify.call_args.kwargs["content_html"]


def test_second_retry_round_succeeding_sends_no_failure_mail(app):
    fail = lambda: _log("error", ["a", "b"], "all failed")
    calls, notify, sleep = _run(app, [fail(), fail(), _log("success", added=1)])
    assert len(calls) == 3
    notify.assert_not_called()


def test_only_the_last_round_may_email_and_carries_add_up(app):
    first_carry, round1_carry = {"papers_added": 1, "keywords": ["a", "b"]}, {"papers_added": 2, "keywords": ["a", "b"]}
    first = SimpleNamespace(status="error", failed_keywords=["a", "b"], error_message="x", papers_added=1, notify_carry=first_carry)
    round1 = SimpleNamespace(status="error", failed_keywords=["a", "b"], error_message="x", papers_added=1, notify_carry=round1_carry)
    _, notify, _ = _run(app, [first, round1, _log("success")])
    _, r1_kw, r2_kw = notify.kwargs_seen
    assert (r1_kw["defer_notification"], r1_kw["carry"]) == (True, first_carry)
    assert (r2_kw["defer_notification"], r2_kw["carry"]) == (False, round1_carry)


def test_retry_failing_for_another_reason_is_a_failure_mail_without_another_round(app):
    calls, notify, _ = _run(app, [_log("error", ["a", "b"], "all failed"), _log("error", [], "LLM unavailable: down")])
    assert len(calls) == 2
    notify.assert_called_once()
    assert "Nightly crawl failed" in notify.call_args.kwargs["subject"]


def test_quiet_day_has_no_retry(app):
    calls, notify, sleep = _run(app, [_log("success")])
    assert len(calls) == 1
    sleep.assert_not_called()
    notify.assert_not_called()


def test_first_pass_defers_email_and_hands_its_numbers_to_the_retry(app):
    carry = {"papers_added": 3}
    first = SimpleNamespace(status="success", failed_keywords=["b"], error_message="1/2 failed", papers_added=3, notify_carry=carry)
    _, notify, _ = _run(app, [first, _log("success", added=2)])
    first_kw, retry_kw = notify.kwargs_seen
    assert first_kw["defer_notification"] is True
    assert retry_kw["carry"] is carry


def _carry_log(added=3):
    return SimpleNamespace(status="success", failed_keywords=["b"], error_message="1/2 failed", papers_added=added,
                           notify_carry={"papers_added": added, "keywords": ["a", "b"]})


def test_retry_error_with_first_pass_papers_sends_one_combined_email_not_a_failure_mail(app):
    fail = lambda: _log("error", ["a", "b"], "all failed")
    _, notify, _ = _run(app, [_carry_log(), fail(), fail()])
    notify.assert_not_called()  # no separate failure mail
    notify.carry_notify.assert_called_once()
    warning = notify.carry_notify.call_args.args[-1]
    assert "(a, b)" in warning and "Semantic Scholar" in warning


def test_retry_that_notified_with_warning_sends_no_failure_mail(app):
    retry = SimpleNamespace(status="success", failed_keywords=["b"], error_message="1/2 failed", papers_added=2, notified=True)
    _, notify, _ = _run(app, [_carry_log(), _log("error", ["b"], "all failed"), retry])
    notify.assert_not_called()
    notify.carry_notify.assert_not_called()


def test_retry_finding_nothing_new_still_announces_first_pass(app):
    _, notify, _ = _run(app, [_carry_log(), _log("success")])
    notify.carry_notify.assert_called_once()
    assert notify.carry_notify.call_args.args[-1] is None
