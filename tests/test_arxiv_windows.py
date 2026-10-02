"""Adaptive date-window arXiv search + version-insensitive dedup (no network)."""
from datetime import date
from unittest.mock import patch

from app.crawl import _collect_arxiv_keyword, search_arxiv_with_full_papers
from app.utils import base_arxiv_id


def _p(aid):
    return {"arxiv_id": aid, "title": aid, "authors": [], "abstract": "", "pdf_url": "", "published_date": date(2026, 9, 1), "year": 2026, "categories": []}


def test_base_arxiv_id():
    assert base_arxiv_id("2605.23872v2") == "2605.23872"
    assert base_arxiv_id("2605.23872") == "2605.23872"
    assert base_arxiv_id("solv-int/9901001v1") == "solv-int/9901001"
    assert base_arxiv_id("solv-int/9901001") == "solv-int/9901001"
    assert base_arxiv_id("web:abc") == "web:abc"


def _pd(aid, d):
    return {**_p(aid), "published_date": d}


def test_cap_hit_continues_from_oldest_date():
    """Newest-first results: after a capped query, next window ends at the oldest date seen."""
    calls = []
    pages = [
        ([_pd("a", date(2026, 9, 10)), _pd("b", date(2026, 9, 8))], True),   # capped, oldest = 9/8
        ([_pd("b", date(2026, 9, 8)), _pd("c", date(2026, 9, 5))], True),    # boundary day re-queried; oldest = 9/5
        ([_pd("d", date(2026, 9, 2))], False),                              # fits -> done
    ]

    def fake_fetch(kw, d_from, d_to, max_results):
        calls.append((d_from, d_to))
        batch, cap = pages[len(calls) - 1]
        return batch, cap, False, False

    warnings = []
    with patch("app.crawl._fetch_arxiv_window", side_effect=fake_fetch):
        papers = _collect_arxiv_keyword("kw", date(2026, 9, 1), date(2026, 9, 10), 2, warnings)
    assert calls == [(date(2026, 9, 1), date(2026, 9, 10)), (date(2026, 9, 1), date(2026, 9, 8)), (date(2026, 9, 1), date(2026, 9, 5))]
    assert [p["arxiv_id"] for p in papers] == ["a", "b", "c", "d"]  # "b" not duplicated
    assert not warnings


def test_single_day_over_cap_warns_and_moves_on():
    warnings = []
    day = date(2026, 9, 3)
    with patch("app.crawl._fetch_arxiv_window", side_effect=[([_pd("a", day)] * 5, True, False, False), ([], False, False, False)]) as m:
        _collect_arxiv_keyword("kw", date(2026, 9, 1), day, 5, warnings)
    assert len(warnings) == 1 and "kw" in warnings[0] and "cut off" in warnings[0]
    assert m.call_args_list[1].args[1:3] == (date(2026, 9, 1), date(2026, 9, 2))


def test_timeout_warns():
    warnings = []
    with patch("app.crawl._fetch_arxiv_window", return_value=([_p("a")], False, True, False)):
        _collect_arxiv_keyword("kw", date(2026, 9, 1), date(2026, 9, 3), 200, warnings)
    assert len(warnings) == 1 and "timed out" in warnings[0]


def test_search_dedups_across_keywords_and_reports_warnings():
    def fake_collect(kw, d_from, d_to, max_results, warnings, failed=None):
        warnings.append(f"w-{kw}")
        return [_p("1"), _p(f"2-{kw}")]

    w = []
    with patch("app.crawl._collect_arxiv_keyword", side_effect=fake_collect):
        papers = search_arxiv_with_full_papers(["a", "b"], date(2026, 9, 1), date(2026, 9, 2), warnings=w)
    assert [p["arxiv_id"] for p in papers] == ["1", "2-a", "2-b"]
    assert w == ["w-a", "w-b"]


def test_errored_query_is_reported_as_failed_keyword():
    warnings, failed = [], []
    with patch("app.crawl._fetch_arxiv_window", return_value=([], False, False, True)):
        papers = _collect_arxiv_keyword("kw", date(2026, 9, 1), date(2026, 9, 3), 200, warnings, failed)
    assert papers == [] and failed == ["kw"]
    assert len(warnings) == 1 and "failed" in warnings[0]


def test_apply_search_outcome_all_failed_is_error_partial_is_success_with_message():
    from types import SimpleNamespace
    from app.crawl import _apply_search_outcome
    log = SimpleNamespace(status=None, error_message=None)
    _apply_search_outcome(log, ["a", "b"], ["a", "b"])
    assert log.status == "error" and "all 2" in log.error_message

    log = SimpleNamespace(status=None, error_message=None)
    _apply_search_outcome(log, ["a"], ["a", "b"])
    assert log.status == "success" and "1/2" in log.error_message

    log = SimpleNamespace(status=None, error_message=None)
    _apply_search_outcome(log, [], ["a", "b"])  # quiet day: nothing failed
    assert log.status == "success" and log.error_message is None


def test_citation_fetch_strips_arxiv_version():
    from unittest.mock import MagicMock
    from app import crawl
    crawl._citation_cache.clear()
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"citationCount": 7}
    with patch("app.crawl.httpx.Client") as client:
        client.return_value.__enter__.return_value.get.return_value = resp
        assert crawl._fetch_citations("arXiv:2510.01706v2") == 7
        url = client.return_value.__enter__.return_value.get.call_args.args[0]
    assert url.endswith("/paper/arXiv:2510.01706?fields=citationCount")


import pytest


@pytest.mark.slow
def test_live_generic_keyword_pages_back_through_window():
    """Real arXiv: a very generic keyword with a tiny per-query cap must page back over several days,
    covering the range with no duplicates and nothing outside it."""
    from datetime import timedelta
    d_to = date.today() - timedelta(days=2)
    d_from = d_to - timedelta(days=4)
    warnings = []
    with patch("app.crawl._fetch_arxiv_window", wraps=__import__("app.crawl", fromlist=["x"])._fetch_arxiv_window) as spy:
        papers = _collect_arxiv_keyword("model", d_from, d_to, 40, warnings)
    ids = [p["arxiv_id"] for p in papers]
    assert spy.call_count > 1, "cap was never hit — keyword not generic enough for this test"
    assert len(ids) == len(set(ids))
    assert all(d_from <= p["published_date"] <= d_to for p in papers)
    assert min(p["published_date"] for p in papers) <= d_from + timedelta(days=1), "did not page back to the start of the window"


class TestCountStepDedup:
    """/arxiv-count must treat '2605.23872v1' from arXiv as the same paper as a stored '2605.23872'."""

    def test_versioned_result_matches_unversioned_collection_paper(self, app, client, project):
        import threading
        from app import db
        from app.models import Paper, ProjectPaper, ScreenedPaper, compute_screening_hash

        class _SyncThread:
            def __init__(self, target=None, daemon=None, **kw):
                self._t = target

            def start(self):
                self._t()

        with app.app_context():
            coll = Paper(arxiv_id="2605.23872", title="In collection", abstract="x")
            rejected = Paper(arxiv_id="2609.00002", title="Was rejected", abstract="x")
            db.session.add_all([coll, rejected])
            db.session.flush()
            pp = ProjectPaper(project_id=project.id, paper_id=coll.id, manual_tag="related")
            db.session.add(pp)
            db.session.commit()
            h = compute_screening_hash(project.research_interest, [pp])
            db.session.add(ScreenedPaper(project_id=project.id, arxiv_id="2609.00002", screening_hash=h, is_relevant=False))
            db.session.commit()

        found = [_p("2605.23872v1"), _p("2609.00001v1"), _p("2609.00002v2")]

        def fake_search(keywords, d_from, d_to, progress_callback=None, warnings=None):
            warnings.append("kw: cut off")
            return found

        with client.session_transaction() as sess:
            sess["logged_in"] = True
        with patch("threading.Thread", _SyncThread), \
                patch("app.routes.projects.search_arxiv_with_full_papers", side_effect=fake_search):
            r = client.post(f"/projects/{project.slug}/arxiv-count", data={
                "keywords_json": '["kw"]', "date_from": "2026-09-01", "date_to": "2026-09-02", "use_scholar": "0",
            })
            task_id = r.get_json()["task_id"]
        res = client.get(f"/projects/{project.slug}/arxiv-count-status/{task_id}").get_json()
        assert res["status"] == "done", res
        assert res["already_known"] == 1
        assert res["count"] == 2
        assert res["previously_rejected"] == 1
        assert res["warnings"] == ["kw: cut off"]

        with app.app_context():
            ScreenedPaper.query.filter_by(project_id=project.id).delete()
            ProjectPaper.query.filter_by(project_id=project.id).delete()
            Paper.query.filter(Paper.arxiv_id.in_(["2605.23872", "2609.00002"])).delete()
            db.session.commit()



def test_search_warning_names_failed_and_ok_keywords():
    from app.crawl import _search_warning
    w = _search_warning(["bar"], ["foo", "moe", "bar"])
    assert "1 of 3 keywords (bar)" in w and "Searched fine: foo, moe." in w
    assert "No keyword could be searched" in _search_warning(["foo"], ["foo"])
