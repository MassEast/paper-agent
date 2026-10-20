"""
Token usage per project (app/usage.py, LlmUsage): every action that calls the model writes one row with
its kind, project totals and the "LLM Usage" panel under the crawl calendar sum those rows, and the card's
summary tokens cover only the summary + contributions of the latest run.

LLM calls are mocked; llm._count stands in for a completed call where only the counting matters.
"""

import json
from unittest.mock import patch

import httpx
import pytest

from app import db, llm, USAGE_BACKFILL_SQL
from app import crawl
from app.models import CrawlLog, LlmUsage, Paper, PaperSummary, Project, ProjectPaper
from app.usage import counted
from tests.test_paper_chat import _mock_http, _sse


@pytest.fixture
def proj(app):
    with app.app_context():
        p = Project(name="Usage Proj", slug="usage-proj", research_interest="sparse attention")
        db.session.add(p)
        db.session.commit()
        pid = p.id
    yield pid
    with app.app_context():
        LlmUsage.query.filter_by(project_id=pid).delete()
        CrawlLog.query.filter_by(project_id=pid).delete()
        for pp in ProjectPaper.query.filter_by(project_id=pid):
            PaperSummary.query.filter_by(paper_id=pp.paper_id).delete()
            Paper.query.filter_by(id=pp.paper_id).delete()
        ProjectPaper.query.filter_by(project_id=pid).delete()
        Project.query.filter_by(id=pid).delete()
        db.session.commit()


@pytest.fixture
def logged_in(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    return client


def _rows(pid):
    return [(r.kind, r.tokens, r.calls) for r in LlmUsage.query.filter_by(project_id=pid).order_by(LlmUsage.id)]


def test_one_row_per_action_with_all_its_calls(app, proj):
    with app.app_context():
        with counted(proj, "chat"):
            llm._count({"total_tokens": 100})
            llm._count({"prompt_tokens": 30, "completion_tokens": 20})  # no total from the server
        with counted(proj, "regenerate"):
            pass  # no model call, no row
        assert _rows(proj) == [("chat", 150, 2)]


def test_tokens_of_a_failed_action_still_count(app, proj):
    with app.app_context():
        with pytest.raises(RuntimeError):
            with counted(proj, "add"):
                llm._count({"total_tokens": 40})
                raise RuntimeError("summary failed")
        assert _rows(proj) == [("add", 40, 1)]


def test_rescreen_counts_the_calls_made_in_its_worker_threads(app, proj):
    with app.app_context():
        paper = Paper(arxiv_id="2601.07001", title="Usage Paper", abstract="About attention.", source="arxiv")
        db.session.add(paper)
        db.session.flush()
        db.session.add(ProjectPaper(project_id=proj, paper_id=paper.id))
        db.session.commit()

        def quick(*a, **kw):
            llm._count({"total_tokens": 7})
            return False, "not-relevant", "m", 1, 7

        with patch("app.crawl._screen_paper_quick", side_effect=quick):
            crawl.run_rescreen(proj)
        assert _rows(proj) == [("rescreen", 7, 1)]


def test_streamed_chat_answer_counts_the_server_usage(app, proj, monkeypatch):
    _mock_http(monkeypatch, lambda r: httpx.Response(200, text=_sse(
        ["It ", "works."], usage={"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19})))
    with app.app_context():
        with counted(proj, "chat"):
            list(llm._llm_stream([{"role": "user", "content": "hi"}]))
        assert _rows(proj) == [("chat", 19, 1)]


def test_a_stopped_chat_answer_still_counts_an_estimate(app, proj, monkeypatch):
    _mock_http(monkeypatch, lambda r: httpx.Response(200, text=_sse(["word "] * 50)))
    with app.app_context():
        with counted(proj, "chat"):
            stream = llm._llm_stream([{"role": "user", "content": "x" * 4000}])
            next(stream)
            stream.close()  # what the chat's Stop does
        [(kind, tokens, calls)] = _rows(proj)
        assert kind == "chat" and calls == 1 and tokens >= 1000  # the 4000-char prompt was read


def test_old_crawls_are_backfilled_once(app, proj):
    with app.app_context():
        saved = [(r.project_id, r.kind, r.tokens, r.calls, r.created_at) for r in LlmUsage.query]
        LlmUsage.query.delete()
        db.session.add(CrawlLog(project_id=proj, status="success", total_tokens=500))
        db.session.add(CrawlLog(project_id=proj, status="success", total_tokens=0))
        db.session.commit()
        for _ in range(2):  # runs on every startup
            db.session.execute(db.text(USAGE_BACKFILL_SQL))
            db.session.commit()
        assert _rows(proj) == [("crawl", 500, None)]
        for row in saved:
            db.session.add(LlmUsage(project_id=row[0], kind=row[1], tokens=row[2], calls=row[3], created_at=row[4]))
        db.session.commit()


def test_project_page_shows_the_usage_by_kind_and_the_total(app, proj, logged_in):
    with app.app_context():
        db.session.add_all([LlmUsage(project_id=proj, kind="crawl", tokens=3_000_000),
                            LlmUsage(project_id=proj, kind="chat", tokens=1_200)])
        db.session.commit()
    html = logged_in.get("/projects/usage-proj").get_data(as_text=True)
    panel = html[html.index('id="usage-panel"'):]
    assert panel.index("Crawls") < panel.index("Paper chat")
    assert "3.00M" in panel and "1.2k" in panel and "Everything is counted since Oct 15, 2026" in panel
    assert "LLM usage:" in html and "~3.00M tokens" in html and "(all crawls)" not in html

    listing = logged_in.get("/projects").get_data(as_text=True)
    assert "3.00M tok" in listing


def test_regenerate_replaces_the_cards_tokens_instead_of_adding(app, proj, logged_in):
    with app.app_context():
        paper = Paper(arxiv_id="2601.07002", title="Regen Paper", abstract="Abstract.", source="arxiv")
        db.session.add(paper)
        db.session.flush()
        db.session.add(ProjectPaper(project_id=proj, paper_id=paper.id, manual_tag="related"))
        db.session.add(PaperSummary(paper_id=paper.id, summary_text="old", model_used="m", elapsed_ms=1,
                                    tokens_used=999))
        db.session.commit()

    def summary(**kw):
        llm._count({"total_tokens": 100})
        return {"summary": "new"}, "m", 10, 100

    def contributions(**kw):
        llm._count({"total_tokens": 50})
        return "c", "", 5, 50

    with patch("app.routes.papers.get_paper_figure", return_value=(None, None)), \
         patch("app.routes.papers.get_citation_count", return_value=None), \
         patch("app.routes.papers.get_paper_content", return_value="full text"), \
         patch("app.crawl.generate_summary_with_full_content", side_effect=summary), \
         patch("app.crawl.extract_main_contributions", side_effect=contributions), \
         patch("app.routes.papers.generate_relevance_reason", return_value="why"):
        resp = logged_in.post("/paper/2601.07002/regenerate", data={"project_id": proj})
    assert resp.status_code in (200, 204)
    with app.app_context():
        assert Paper.query.filter_by(arxiv_id="2601.07002").one().summary.tokens_used == 150
        assert _rows(proj) == [("regenerate", 150, 2)]
