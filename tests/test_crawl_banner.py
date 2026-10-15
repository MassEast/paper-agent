"""
"Nightly crawl is off" banner (partials/crawl_banner.html): says what the nightly crawl does, lists what
the project still needs, and updates itself when that changes, without a reload.
"""

import json
import re

import pytest

from app import db
from app.models import Paper, Project, ProjectPaper


@pytest.fixture
def proj(app):
    """Research interest and keywords set; no collection papers, no email, nightly crawl off."""
    with app.app_context():
        project = Project(name="Banner Proj", slug="banner-proj", research_interest="sparse attention",
                          saved_keywords=json.dumps(["sparse attention"]))
        paper = Paper(arxiv_id="2611.00001", title="Some Paper", source="arxiv", abstract="A.")
        db.session.add_all([project, paper])
        db.session.flush()
        db.session.add(ProjectPaper(project_id=project.id, paper_id=paper.id))  # in New Papers, not collected
        db.session.commit()
        ids = {"project": project.id, "paper": paper.id}
    yield ids
    with app.app_context():
        ProjectPaper.query.filter_by(project_id=ids["project"]).delete()
        Paper.query.filter_by(id=ids["paper"]).delete()
        Project.query.filter_by(id=ids["project"]).delete()
        db.session.commit()


@pytest.fixture
def c(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    return client


def _ticks(html):
    banner = re.sub(r"\s+", " ", html)
    return {label: done == "done" for done, label in re.findall(r'<li class="(done|)"><span class="crawl-step-mark">.</span> <span><b>([^<]+)</b>', banner)}


def test_banner_ticks_what_is_set_and_says_now_only_when_everything_is(c, app, proj):
    html = c.get("/projects/banner-proj/crawl-banner").data.decode()
    assert _ticks(html) == {"Research interest": True, "Search keywords": True,
                            "A few papers in My Collection": False, "An email address": False}
    assert "Now switch on" not in html
    with app.app_context():
        Project.query.get(proj["project"]).notification_emails = "justus.westerhoff@bht-berlin.de"
        ProjectPaper.query.filter_by(project_id=proj["project"]).first().manual_tag = "related"
        db.session.commit()
    html = c.get("/projects/banner-proj/crawl-banner").data.decode()
    assert all(_ticks(html).values()) and "Now switch on" in html


def test_banner_is_empty_once_the_nightly_crawl_is_on(c, app, proj):
    with app.app_context():
        Project.query.get(proj["project"]).crawl_hour = 2
        db.session.commit()
    html = c.get("/projects/banner-proj/crawl-banner").data.decode()
    assert 'id="crawl-banner"' in html and "crawl-off-banner" not in html  # the wrapper stays, to come back


def test_setup_changes_tell_the_banner_to_redraw(c, proj):
    resp = c.post("/projects/banner-proj/notification-emails", data={"notification_emails": "justus.westerhoff@bht-berlin.de"},
                  headers={"HX-Request": "true"})
    assert resp.headers["HX-Trigger"] == "crawlSetupChanged"
    resp = c.post("/projects/banner-proj/toggle-nightly-crawl", headers={"HX-Request": "true"})
    assert resp.headers["HX-Trigger"] == "crawlSetupChanged"
    resp = c.post("/paper/2611.00001/tag", data={"project_id": proj["project"], "tag": "related"},
                  headers={"HX-Request": "true"})
    assert resp.headers["HX-Trigger"] == "crawlSetupChanged"


def test_fetch_saves_on_the_page_fire_the_redraw_too(c, proj):
    """Research interest, keywords and emails are saved with fetch(), which ignores HX-Trigger."""
    html = c.get("/projects/banner-proj").data.decode()
    assert html.count("htmx.trigger(document.body, 'crawlSetupChanged')") == 3


def test_saving_no_keywords_clears_them(c, app, proj):
    c.post("/projects/banner-proj/save-keywords", data={"keywords_json": "[]"})
    with app.app_context():
        assert Project.query.get(proj["project"]).saved_keywords is None  # not "[]", which would count as set
    assert _ticks(c.get("/projects/banner-proj/crawl-banner").data.decode())["Search keywords"] is False


def test_no_static_enable_hint_under_the_switch(c, proj):
    """The banner lists what is missing and updates itself; the old hint under the switch did neither."""
    html = c.get("/projects/banner-proj").data.decode()
    assert "To enable:" not in html and "crawl-pulse" not in html


@pytest.mark.slow
def test_keywords_can_be_removed_and_the_checklist_follows(live_server, browser_session, app, proj):
    browser, state = browser_session
    with app.app_context():
        Project.query.get(proj["project"]).saved_keywords = json.dumps(["sparse attention", "pruning"])
        db.session.commit()
    context = browser.new_context(storage_state=state)
    pg = context.new_page()
    pg.goto(f"{live_server}/projects/banner-proj")
    pg.get_by_role("button", name="Remove keyword pruning").click()
    assert pg.locator('input[name="keyword_item"]').count() == 1
    pg.locator("#save-kw-btn").click()
    pg.wait_for_selector("#save-kw-btn:has-text('Saved')")
    with app.app_context():
        assert json.loads(Project.query.get(proj["project"]).saved_keywords) == ["sparse attention"]

    pg.get_by_role("button", name="Remove keyword sparse attention").click()
    assert "No keywords yet" in pg.locator("#kw-panel-label").inner_text()
    pg.locator("#save-kw-btn").click()
    pg.wait_for_function("() => !document.querySelector('#crawl-off-banner li.done b') || "
                         "![...document.querySelectorAll('#crawl-off-banner li.done b')].some(b => b.textContent === 'Search keywords')")
    with app.app_context():
        assert Project.query.get(proj["project"]).saved_keywords is None
    context.close()


@pytest.mark.slow
def test_checklist_updates_live_and_banner_goes_when_switched_on(live_server, browser_session, app, proj):
    browser, state = browser_session
    with app.app_context():
        ProjectPaper.query.filter_by(project_id=proj["project"]).first().manual_tag = "related"
        db.session.commit()
    context = browser.new_context(storage_state=state)
    pg = context.new_page()
    pg.goto(f"{live_server}/projects/banner-proj")
    pg.wait_for_selector("#crawl-off-banner")
    assert pg.locator("#crawl-off-banner li.done").count() == 3 and pg.locator("text=Now switch on").count() == 0
    pg.evaluate("window.__notReloaded = true")

    pg.fill(f"#notification-emails-{proj['project']}", "justus.westerhoff@bht-berlin.de")
    pg.locator(f"#email-save-btn-{proj['project']}").click()
    pg.wait_for_selector("text=Now switch on")
    assert pg.locator("#crawl-off-banner li.done").count() == 4

    pg.locator(f"#crawl-toggle-btn-{proj['project']}").click()
    pg.wait_for_selector("#crawl-off-banner", state="detached")
    assert pg.evaluate("window.__notReloaded") is True
    context.close()
