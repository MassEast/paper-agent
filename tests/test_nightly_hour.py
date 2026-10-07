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
