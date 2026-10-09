"""
What happens when a click or edit can't reach the server (no connection, server error, logged out).

The fast tests cover the server side (logged-out htmx requests). The slow ones drive a real headless
Chromium against a live server with the connection cut via Playwright, and check both what the page
shows and that the DB is untouched. They are `slow` because the page loads htmx/Tailwind from CDNs.
Playwright comes from requirements-dev.txt; its browser needs `python -m playwright install chromium`.
"""

import threading

import pytest

from app import db
from app.models import Paper, Project, ProjectPaper


def test_logged_out_htmx_request_gets_hx_redirect(client):
    resp = client.post("/paper/2501.00001/trash", headers={"HX-Request": "true"})
    assert resp.status_code == 401
    assert resp.headers["HX-Redirect"] == "/login"


def test_logged_out_plain_request_still_redirects(client):
    resp = client.get("/projects")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/login")


# ── Browser tests ─────────────────────────────────────────────────────────────

SLUG = "offline-ui-test"
NEW_ID = "2501.11111"
COLL_ID = "2501.22222"


@pytest.fixture(scope="module")
def live_server(app):
    from werkzeug.serving import make_server

    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def seeded(app):
    """A project with one New Papers paper and one My Collection paper."""
    with app.app_context():
        project = Project(name="Offline UI Test", slug=SLUG, research_interest="original interest")
        db.session.add(project)
        db.session.flush()
        for arxiv_id, tag in ((NEW_ID, None), (COLL_ID, "related")):
            paper = Paper(arxiv_id=arxiv_id, title=f"Paper {arxiv_id}", abstract="An abstract.", source="arxiv")
            db.session.add(paper)
            db.session.flush()
            db.session.add(ProjectPaper(project_id=project.id, paper_id=paper.id, manual_tag=tag))
        db.session.commit()
        pid = project.id
    yield pid
    with app.app_context():
        ProjectPaper.query.filter_by(project_id=pid).delete()
        Paper.query.filter(Paper.arxiv_id.in_([NEW_ID, COLL_ID])).delete()
        db.session.delete(db.session.get(Project, pid))
        db.session.commit()


def _link(app, pid, arxiv_id):
    with app.app_context():
        paper = Paper.query.filter_by(arxiv_id=arxiv_id).one()
        return ProjectPaper.query.filter_by(project_id=pid, paper_id=paper.id).one()


@pytest.fixture(scope="module")
def browser_session(live_server):
    """One browser + one login per module: /login is rate-limited to 5 per minute."""
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        pg = context.new_page()
        pg.goto(f"{live_server}/login")
        pg.fill("input[name=password]", "testpass")
        pg.click("button[type=submit]")
        state = context.storage_state()
        context.close()
        yield browser, state
        browser.close()


@pytest.fixture
def page(live_server, browser_session, seeded):
    browser, state = browser_session
    context = browser.new_context(storage_state=state)
    pg = context.new_page()
    pg.goto(f"{live_server}/projects/{SLUG}")
    pg.wait_for_function("window.htmx !== undefined")
    yield pg
    context.close()


def _notes(page, pid, arxiv_id):
    return page.locator(f'[id="notes-{arxiv_id}-{pid}"]')


def _notes_status(page, pid, arxiv_id):
    return page.locator(f'[id="notes-saved-{arxiv_id}-{pid}"]')


@pytest.mark.slow
def test_offline_click_shows_toast_and_changes_nothing(app, page, seeded):
    page.context.set_offline(True)
    assert page.locator("#offline-banner").is_visible()

    page.get_by_role("button", name="Move to My Collection").first.click()

    toast = page.locator(".rw-toast")
    toast.wait_for()
    assert "NOT saved" in toast.inner_text()
    assert _link(app, seeded, NEW_ID).manual_tag is None
    # Button is usable again for a retry
    assert page.get_by_role("button", name="Move to My Collection").first.is_enabled()


@pytest.mark.slow
def test_offline_notes_kept_and_saved_when_back_online(app, page, seeded):
    page.context.set_offline(True)
    notes = _notes(page, seeded, COLL_ID)
    notes.click()
    notes.type("written on the train")
    notes.blur()

    status = _notes_status(page, seeded, COLL_ID)
    page.wait_for_function("el => el.textContent.includes('not saved')", arg=status.element_handle())
    assert "will retry when online" in status.inner_text()
    assert _link(app, seeded, COLL_ID).notes is None

    page.context.set_offline(False)
    page.evaluate("window.dispatchEvent(new Event('online'))")
    page.wait_for_function(
        "key => localStorage.getItem(key) === null", arg=f"notes-draft:{COLL_ID}:{seeded}", timeout=10_000
    )
    assert _link(app, seeded, COLL_ID).notes == "written on the train"


@pytest.mark.slow
def test_unsaved_notes_survive_reload(app, page, seeded):
    # Server unreachable for notes only, so the page itself can still be reloaded afterwards
    page.route("**/notes", lambda route: route.abort())
    notes = _notes(page, seeded, COLL_ID)
    notes.click()
    notes.type("draft before reload")
    notes.blur()
    page.wait_for_function(
        "el => el.textContent.includes('not saved')", arg=_notes_status(page, seeded, COLL_ID).element_handle()
    )
    assert _link(app, seeded, COLL_ID).notes is None

    page.unroute("**/notes")
    page.on("dialog", lambda d: d.accept())  # beforeunload prompt for the pending note
    page.reload()
    page.wait_for_function(
        "key => localStorage.getItem(key) === null", arg=f"notes-draft:{COLL_ID}:{seeded}", timeout=10_000
    )
    assert _notes(page, seeded, COLL_ID).input_value() == "draft before reload"
    assert _link(app, seeded, COLL_ID).notes == "draft before reload"


@pytest.mark.slow
def test_offline_crawl_toggle_does_not_flip(app, page, seeded):
    toggle = page.locator(f"#crawl-toggle-btn-{seeded}")
    assert toggle.get_attribute("data-on") == "false"
    page.context.set_offline(True)
    toggle.click()
    page.locator(".rw-toast").wait_for()
    assert toggle.get_attribute("data-on") == "false"
    with app.app_context():
        assert db.session.get(Project, seeded).crawl_hour is None


@pytest.mark.slow
def test_offline_research_interest_save_reports_failure(app, page, seeded):
    page.fill("#research-interest-field", "changed interest")
    page.context.set_offline(True)
    page.click("#save-btn")
    page.locator(".rw-toast").wait_for()
    assert "Not saved" in page.locator("#save-btn").inner_text()
    assert page.locator("#save-btn").is_enabled()
    assert page.evaluate("unsavedChecks.some(f => f())")
    with app.app_context():
        assert db.session.get(Project, seeded).research_interest == "original interest"


@pytest.mark.slow
def test_logged_out_click_goes_to_login_page(app, page, seeded):
    page.context.clear_cookies()
    page.get_by_role("button", name="Move to My Collection").first.click()
    page.wait_for_url("**/login")
    assert _link(app, seeded, NEW_ID).manual_tag is None
