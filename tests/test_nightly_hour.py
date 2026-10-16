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


def test_a_run_catches_up_projects_whose_hour_passed_without_a_nightly_crawl(app):
    """2026-10-16: the 02:00 run lasted until 04:05, so the 03:00 run never started and its projects
    were not crawled. The 04:00 run now takes them too; projects already crawled tonight are left alone."""
    from app import db
    from app.models import CrawlLog, Project
    from scripts.nightly_crawl import _projects_for_run

    run_at = datetime(2026, 10, 16, 4, 0, tzinfo=timezone.utc)
    with app.app_context():
        ps = {h: Project(name=f"Hour {h}", slug=f"catchup-{h}", crawl_hour=h) for h in (1, 2, 3, 4)}
        off = Project(name="Off", slug="catchup-off", crawl_hour=None)
        db.session.add_all([*ps.values(), off])
        db.session.flush()
        db.session.add_all([
            CrawlLog(project_id=ps[1].id, triggered_by="cron", started_at=datetime(2026, 10, 16, 1, 0)),
            CrawlLog(project_id=ps[2].id, triggered_by="cron", started_at=datetime(2026, 10, 15, 2, 0)),  # yesterday
            CrawlLog(project_id=ps[3].id, triggered_by="manual", started_at=datetime(2026, 10, 16, 3, 30)),
        ])
        db.session.commit()
        try:
            mine = [*ps.values(), off]
            projects, caught_up = _projects_for_run(mine, run_at)
            assert [p.name for p in caught_up] == ["Hour 2", "Hour 3"]
            assert [p.name for p in projects] == ["Hour 2", "Hour 3", "Hour 4"]
            projects, caught_up = _projects_for_run(mine, run_at.replace(hour=1))
            assert caught_up == [] and [p.name for p in projects] == ["Hour 1"]
        finally:
            CrawlLog.query.filter(CrawlLog.project_id.in_([p.id for p in ps.values()])).delete()
            for p in mine:
                db.session.delete(p)
            db.session.commit()
