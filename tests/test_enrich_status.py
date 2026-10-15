"""
A freshly added paper's card says what is still being fetched in the background and fills in by itself
(crawl._enriching, /project-paper/<id>/card), no reload needed.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from app import crawl, db
from app.models import Paper, PaperSummary, Project, ProjectPaper


class _SyncThread:
    """Runs the background task inline, so the test sees its whole run."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self._target = target

    def start(self):
        self._target()


@pytest.fixture
def added(app):
    """A project with a paper that was just added by URL (placeholder title, nothing fetched yet)."""
    with app.app_context():
        project = Project(name="Enrich Proj", slug="enrich-proj", research_interest="attention")
        paper = Paper(arxiv_id="2610.00001", title="arXiv:2610.00001", source="manual", authors="[]")
        db.session.add_all([project, paper])
        db.session.flush()
        pp = ProjectPaper(project_id=project.id, paper_id=paper.id, manual_tag="related", paper_tags='["important"]')
        db.session.add(pp)
        db.session.commit()
        ids = {"project": project.id, "paper": paper.id, "pp": pp.id}
    yield ids
    crawl._enriching.pop(ids["paper"], None)
    with app.app_context():
        PaperSummary.query.filter_by(paper_id=ids["paper"]).delete()
        ProjectPaper.query.filter_by(project_id=ids["project"]).delete()
        Paper.query.filter_by(id=ids["paper"]).delete()
        Project.query.filter_by(id=ids["project"]).delete()
        db.session.commit()


@pytest.fixture
def c(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    return client


@pytest.fixture
def steps(monkeypatch):
    """Records every step a background task reports."""
    seen = []
    real = crawl._set_step
    monkeypatch.setattr(crawl, "_set_step", lambda pid, by, step: (seen.append((by, step)), real(pid, by, step)))
    return seen


def test_only_the_task_that_set_the_status_clears_it():
    crawl._set_step(-1, "arxiv", "Fetching the figure")
    crawl._end_steps(-1, "web")  # a web add that handed this paper over to the arXiv task
    assert crawl.enrich_step(-1) == "Fetching the figure"
    crawl._end_steps(-1, "arxiv")
    assert crawl.enrich_step(-1) is None


def test_arxiv_add_reports_each_step_and_clears_at_the_end(app, added, steps):
    with app.app_context(), \
            patch("app.crawl.threading.Thread", _SyncThread), \
            patch("app.crawl._ArxivLockCtx"), \
            patch("app.crawl.arxiv.Client") as client, \
            patch("app.crawl.get_paper_figure", return_value=(None, None)), \
            patch("app.crawl.get_citation_count", return_value=None), \
            patch("app.crawl.enrich_page_counts"), \
            patch("app.crawl.get_paper_content", return_value=None), \
            patch("app.crawl.generate_summary_with_full_content", return_value=({"summary": "S"}, "m", 1, 1)), \
            patch("app.crawl.extract_main_contributions", return_value=("- C", "", 1, 1)), \
            patch("app.crawl.enrich_institutions_from_scholar"):
        client.return_value.results.return_value = []
        crawl.start_collection_add_async(added["project"], "2610.00001", added["paper"], app)
        assert ProjectPaper.query.get(added["pp"]).relevance_reason is None  # added by hand: no why-line
    assert [s for _, s in steps] == [
        "Waiting to start", "Fetching title, authors and abstract from arXiv", "Fetching the figure",
        "Fetching the citation count", "Counting the pages", "Writing the AI summary and main contributions",
        "Finding the authors' institutions"]
    assert crawl.enrich_step(added["paper"]) is None


def test_arxiv_add_that_crashes_still_clears_the_status(app, added):
    with app.app_context(), \
            patch("app.crawl.threading.Thread", _SyncThread), \
            patch("app.crawl._ArxivLockCtx", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            crawl.start_collection_add_async(added["project"], "2610.00001", added["paper"], app)
    assert crawl.enrich_step(added["paper"]) is None  # the card stops saying "fetching"


def test_web_add_handing_over_to_arxiv_keeps_the_arxiv_status(app, added, steps):
    info = {"title": "Sparse Attention", "abstract": "a", "authors": [], "page_arxiv_links": [],
            "doi": None, "pdf_url": None, "page_count": None, "full_text": "a"}
    candidate = MagicMock(title="Sparse Attention", entry_id="http://arxiv.org/abs/2610.00002v1")

    def handover(project_id, arxiv_id, paper_id, app):
        crawl._set_step(paper_id, "arxiv", "Waiting to start")

    with app.app_context():
        Paper.query.get(added["paper"]).arxiv_id = "web:handover"
        db.session.commit()
        with patch("app.crawl.threading.Thread", _SyncThread), \
                patch("app.crawl.extract_paper_info_from_url", return_value=info), \
                patch("app.crawl._ArxivLockCtx"), \
                patch("app.crawl.arxiv.Client") as client, \
                patch("app.crawl.start_collection_add_async", side_effect=handover):
            client.return_value.results.return_value = [candidate]
            crawl.start_web_paper_add_async(added["project"], "https://example.org/p", added["paper"], app)
    assert ("web", "Reading the web page") in steps and ("web", "Looking for the paper on arXiv") in steps
    assert crawl.enrich_step(added["paper"]) == "Waiting to start"  # the arXiv task's, not cleared by the web task


def test_card_shows_the_step_and_polls_only_while_fetching(c, added):
    crawl._set_step(added["paper"], "arxiv", "Fetching the figure")
    html = c.get("/projects/enrich-proj").data.decode()
    assert "Fetching the figure" in html and f'hx-get="/project-paper/{added["pp"]}/card?step=Fetching%20the%20figure"' in html
    crawl._end_steps(added["paper"], "arxiv")
    html = c.get("/projects/enrich-proj").data.decode()
    assert 'class="enrich-status"' not in html


def test_card_refresh_answers_only_when_something_changed(c, app, added):
    crawl._set_step(added["paper"], "arxiv", "Fetching the figure")
    assert c.get(f"/project-paper/{added['pp']}/card?step=Fetching the figure").status_code == 204

    crawl._set_step(added["paper"], "arxiv", "Counting the pages")
    resp = c.get(f"/project-paper/{added['pp']}/card?step=Fetching the figure")
    assert resp.status_code == 200 and resp.headers["HX-Retarget"] == f'[data-pp-card="{added["pp"]}"]'
    assert "Counting the pages" in resp.data.decode()

    with app.app_context():
        p = Paper.query.get(added["paper"])
        p.title, p.abstract, p.authors = "Real Title", "Real abstract.", json.dumps(["A. Author"])
        db.session.commit()
    crawl._end_steps(added["paper"], "arxiv")
    html = c.get(f"/project-paper/{added['pp']}/card?step=Counting the pages").data.decode()
    assert "Real Title" in html and 'class="enrich-status"' not in html  # done: filled in, polling ends


def test_card_refresh_of_a_removed_paper_stops_polling(c):
    assert c.get("/project-paper/987654/card?step=x").status_code == 286


# ── Browser (slow): the card fills in without a reload ───────────────────────

@pytest.mark.slow
def test_added_paper_fills_in_without_a_reload(live_server, browser_session, app, added):
    browser, state = browser_session
    crawl._set_step(added["paper"], "arxiv", "Fetching title, authors and abstract from arXiv")
    context = browser.new_context(storage_state=state)
    pg = context.new_page()
    pg.goto(f"{live_server}/projects/enrich-proj")
    pg.wait_for_selector("text=Fetching title, authors and abstract from arXiv")
    pg.evaluate("window.__notReloaded = true")

    crawl._set_step(added["paper"], "arxiv", "Writing the AI summary and main contributions")
    pg.wait_for_selector("text=Writing the AI summary", timeout=6000)

    # someone is typing a note: the card waits instead of redrawing under them
    card = pg.locator(f'[data-pp-card="{added["pp"]}"]')
    card.locator('textarea[name="notes"]').click()
    with app.app_context():
        p = Paper.query.get(added["paper"])
        p.title, p.abstract = "The Real Title", "The real abstract."
        db.session.commit()
    crawl._end_steps(added["paper"], "arxiv")
    pg.wait_for_timeout(4500)
    assert pg.locator("text=The Real Title").count() == 0
    pg.locator("body").click(position={"x": 5, "y": 5})  # focus leaves the card
    pg.wait_for_selector("text=The Real Title", timeout=6000)
    assert card.locator(".enrich-status").count() == 0
    assert pg.evaluate("window.__notReloaded") is True
    context.close()
