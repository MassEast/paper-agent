"""The nightly script picks projects by the run's scheduled hour, not by when a delayed pod starts."""
from datetime import datetime, timezone

from scripts.nightly_crawl import _scheduled_hour


def test_scheduled_hour_from_cronjob_pod_name(monkeypatch):
    # 29855640 minutes since epoch = 2026-10-07 02:00 UTC (real pod that started at 02:27)
    monkeypatch.setenv("HOSTNAME", "related-work-nightly-crawl-29855640-x7k2p")
    assert _scheduled_hour() == 2


def test_scheduled_hour_falls_back_to_now_for_manual_runs(monkeypatch):
    monkeypatch.setenv("HOSTNAME", "nightly-manual-test-abcde")
    assert _scheduled_hour() == datetime.now(timezone.utc).hour


def test_app_restart_keeps_nightly_crawl_off(app):
    """crawl_hour NULL means "nightly off"; a startup must not switch it back on
    (a leftover backfill in create_app did, on every redeploy, until 2026-10-09)."""
    from app import create_app, db
    from app.models import Project

    with app.app_context():
        p = Project(name="Nightly Off", slug="nightly-off-restart", crawl_hour=None)
        db.session.add(p)
        db.session.commit()
        pid = p.id

    restarted = create_app()
    with restarted.app_context():
        p = db.session.get(Project, pid)
        assert p.crawl_hour is None
        db.session.delete(p)
        db.session.commit()
