"""
Nightly crawl reminders when nobody uses a project: any request in the project counts as use; after 30 and 60
idle days an IMPORTANT email asks whether the crawl is still needed, after 90 days (and both emails) it turns off.
"""

import importlib.util
import pathlib
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from app import db
from app.models import Project

_spec = importlib.util.spec_from_file_location(
    "nightly_crawl_script", pathlib.Path(__file__).parent.parent / "scripts" / "nightly_crawl.py")
nightly = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nightly)

NOW = datetime(2027, 3, 1, 3, 0)


@pytest.fixture
def proj(app):
    with app.app_context():
        p = Project(name="Idle", slug="idle-proj", research_interest="x", crawl_hour=2,
                    notification_emails="a@b.c", last_activity_at=NOW, inactivity_reminders=0)
        db.session.add(p)
        db.session.commit()
        yield p
        db.session.delete(p)
        db.session.commit()


def _check(p, idle_days, reminders):
    p.last_activity_at = NOW - timedelta(days=idle_days)
    p.inactivity_reminders = reminders
    p.crawl_hour = 2
    db.session.commit()
    with patch.object(nightly, "send_notification", return_value=True) as send:
        go = nightly._check_inactivity(p, NOW)
    return go, send


def test_no_email_before_30_days(proj):
    go, send = _check(proj, 29, 0)
    assert go and not send.called and proj.inactivity_reminders == 0


def test_first_reminder_at_30_days_once(proj):
    go, send = _check(proj, 30, 0)
    assert go and proj.inactivity_reminders == 1
    kw = send.call_args.kwargs
    assert kw["important"] and kw["subject"].startswith("IMPORTANT [Idle]") and kw["project_emails"] == "a@b.c"
    assert "30 days" in kw["content_html"] and "April 30, 2027" in kw["content_html"]  # 90 days after last use
    go, send = _check(proj, 31, 1)
    assert go and not send.called  # not every night


def test_second_reminder_at_60_days_names_the_turn_off_date(proj):
    go, send = _check(proj, 60, 1)
    assert go and proj.inactivity_reminders == 2
    assert "60 days" in send.call_args.kwargs["content_html"]
    assert "March 31, 2027" in send.call_args.kwargs["content_html"]


def test_turned_off_after_90_days_and_both_reminders(proj):
    go, send = _check(proj, 90, 2)
    assert not go and proj.crawl_hour is None
    assert send.call_args.kwargs["subject"] == "[Idle] Nightly crawl turned off"


def test_not_turned_off_without_the_second_reminder(proj):
    go, send = _check(proj, 95, 1)  # e.g. the 60-day night was missed: warn first
    assert go and proj.crawl_hour == 2 and proj.inactivity_reminders == 2


def _login(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True


def test_opening_the_project_counts_as_use_and_resets_reminders(client, proj):
    proj.last_activity_at, proj.inactivity_reminders = NOW - timedelta(days=45), 1
    db.session.commit()
    client.get("/projects/idle-proj")  # not logged in: redirect, no use
    db.session.refresh(proj)
    assert proj.inactivity_reminders == 1
    _login(client)
    assert client.get("/projects/idle-proj").status_code == 200
    db.session.refresh(proj)
    assert proj.inactivity_reminders == 0
    assert datetime.utcnow() - proj.last_activity_at < timedelta(minutes=1)


def test_a_card_action_with_project_id_counts(client, proj):
    proj.last_activity_at = NOW - timedelta(days=10)
    db.session.commit()
    _login(client)
    client.get(f"/projects/x-missing?project_id={proj.id}")  # 404: no use
    db.session.refresh(proj)
    assert proj.last_activity_at == NOW - timedelta(days=10)
    from app import activity
    with client.application.test_request_context(f"/paper/abc/tag?project_id={proj.id}"):
        assert activity._project_id() == proj.id
