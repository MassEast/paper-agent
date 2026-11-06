"""Notes when several people or a bad connection are involved, and the Notify email.

Fast tests: a save based on an older note is refused, the lock on a note, what the email shows.
Slow tests: real Chromium, two people in two browsers, with the connection cut (see test_offline_ui.py)."""

import json
from unittest.mock import patch

import pytest

from app import db, members, note_locks
from app.models import Paper, Project, ProjectMember, ProjectPaper

SLUG = "note-safety"
ARXIV = "2611.09999"


@pytest.fixture
def team(app):
    note_locks._locks.clear()
    with app.app_context():
        project = Project(name="Notes", slug=SLUG, per_person=True,
                          notification_emails="ann.a@example.org,bea.b@example.org")
        db.session.add(project)
        db.session.flush()
        ProjectMember.query.filter_by(project_id=project.id).delete()
        members.sync_from_emails(project)
        paper = Paper(arxiv_id=ARXIV, title="A Paper", abstract="Abstract.", source="arxiv")
        db.session.add(paper)
        db.session.flush()
        pp = ProjectPaper(project_id=project.id, paper_id=paper.id, manual_tag="related")
        db.session.add(pp)
        db.session.commit()
        info = {"id": project.id, "pp": pp.id, "people": {m.name: m.id for m in members.members(project)}}
    yield info
    with app.app_context():
        ProjectPaper.query.filter_by(project_id=info["id"]).delete()
        ProjectMember.query.filter_by(project_id=info["id"]).delete()
        Paper.query.filter_by(arxiv_id=ARXIV).delete()
        db.session.delete(db.session.get(Project, info["id"]))
        db.session.commit()


@pytest.fixture
def c(client, team):
    with client.session_transaction() as s:
        s["logged_in"] = True
    return client


def _note(app, team):
    with app.app_context():
        return db.session.get(ProjectPaper, team["pp"]).notes


def _save(c, team, notes, base):
    return c.post(f"/paper/{ARXIV}/notes", data={"project_id": team["id"], "notes": notes, "base": base})


def test_a_save_from_an_older_note_is_refused_and_gets_the_current_one(app, c, team):
    assert _save(c, team, "mine", "").status_code == 204
    r = _save(c, team, "theirs", "")
    assert r.status_code == 409 and r.get_json()["notes"] == "mine"
    assert _note(app, team) == "mine"


def test_resending_what_is_already_saved_is_no_conflict(app, c, team):
    """The answer to the first save got lost on the way back, so the browser sends it again."""
    assert _save(c, team, "mine", "").status_code == 204
    assert _save(c, team, "mine", "").status_code == 204


def test_line_endings_alone_are_no_change(app, c, team):
    assert _save(c, team, "one\r\ntwo", "").status_code == 204
    assert _save(c, team, "one\ntwo\nthree", "one\ntwo").status_code == 204


def test_a_resize_is_never_refused(app, c, team):
    _save(c, team, "mine", "")
    assert c.post(f"/paper/{ARXIV}/notes", data={"project_id": team["id"], "notes_height": 90}).status_code == 204


def test_colored_notes_refuse_an_older_save_too(app, c, team):
    me = team["people"]["Ann"]
    post = lambda text, base: c.post(f"/paper/{ARXIV}/notes-runs", data={
        "project_id": team["id"], "runs": json.dumps([[me, text]]), "base": base})
    assert post("mine", "").status_code == 204
    r = post("theirs", "")
    assert r.status_code == 409 and r.get_json()["runs"][0][:2] == [me, "mine"]


def test_a_note_has_one_writer(c, team):
    lock = lambda client, **kw: c.post(f"/paper/{ARXIV}/note-lock", data={
        "project_id": team["id"], "client": client, **kw}).get_json()["by"]
    state = lambda client: c.get(f"/paper/{ARXIV}/note-state?project_id={team['id']}&client={client}").get_json()["by"]
    c.set_cookie(f"rw_member_{team['id']}", str(team["people"]["Ann"]))
    assert lock("ann") is None
    assert state("ann") is None  # your own lock doesn't lock you out
    assert state("bea") == "Ann"
    assert lock("bea") == "Ann"
    lock("ann", release=1)
    assert state("bea") is None
    assert lock("bea") is None


def test_a_lock_ends_by_itself_when_its_writer_disappears(team):
    with patch("app.note_locks.time.monotonic", return_value=100):
        assert note_locks.claim(1, 1, "gone", "Ann") is None
        assert note_locks.holder(1, 1, "other") == "Ann"
    with patch("app.note_locks.time.monotonic", return_value=100 + note_locks.TTL_S + 1):
        assert note_locks.holder(1, 1, "other") is None


def _email(c, team, **form):
    with patch("app.crawl.send_notification", return_value=True) as send:
        r = c.post(f"/paper/{ARXIV}/notify", data={"project_id": team["id"], **form})
    assert r.get_json()["ok"]
    return send.call_args.kwargs["content_html"]


def test_the_email_shows_each_persons_part_under_their_name_with_its_line_breaks(app, c, team):
    ann, bea = team["people"]["Ann"], team["people"]["Bea"]
    c.post(f"/paper/{ARXIV}/notes-runs", data={"project_id": team["id"], "runs": json.dumps(
        [[ann, "first\nsecond"], [bea, " <b>agreed</b>"], [None, "old note"]])})
    html = _email(c, team)
    assert "Personal Notes" not in html and ">Notes</p>" in html
    assert html.index(">Ann</p>") < html.index("first<br>second") < html.index(">Bea</p>") < html.index("&lt;b&gt;agreed")
    assert "old note" in html


def test_outside_per_person_projects_a_note_has_no_names(app, c, team):
    with app.app_context():
        db.session.get(Project, team["id"]).per_person = False
        db.session.commit()
    c.post(f"/paper/{ARXIV}/notes", data={"project_id": team["id"], "notes": "one\ntwo"})
    html = _email(c, team)
    assert "one<br>two" in html and ">Ann</p>" not in html


def test_the_message_comes_first_with_the_senders_name(c, team):
    c.set_cookie(f"rw_member_{team['id']}", str(team["people"]["Ann"]))
    html = _email(c, team, message="  read this\nbecause <reasons>  ")
    assert "Ann writes" in html and "read this<br>because &lt;reasons&gt;" in html
    assert html.index("Ann writes") < html.index("Flagged Paper")


def test_no_message_no_message_block(c, team):
    assert "writes" not in _email(c, team, message="   ")


# ── Browser tests ─────────────────────────────────────────────────────────────

@pytest.fixture
def open_as(live_server, browser_session, team):
    browser, state = browser_session
    contexts = []

    def _open(name):
        context = browser.new_context(storage_state=state)
        contexts.append(context)
        page = context.new_page()
        page.goto(f"{live_server}/projects/{SLUG}?as={team['people'][name]}")
        page.wait_for_function("window.htmx !== undefined")
        page.evaluate("showPaperTab('col')")
        page.wait_for_selector(".rich-notes:visible")
        return page

    yield _open
    for context in contexts:
        context.close()


def _status(page, text, timeout=10_000):
    page.wait_for_function("t => document.querySelector('.rich-notes-status').textContent.includes(t)",
                           arg=text, timeout=timeout)


def _saved(page):
    page.wait_for_function("document.querySelector('.rich-notes-status').textContent === 'saved'", timeout=10_000)


@pytest.mark.slow
def test_colored_note_written_offline_saves_itself_when_back(app, open_as, team):
    page = open_as("Ann")
    page.context.set_offline(True)
    page.locator(".rich-notes:visible").click()
    page.keyboard.type("on the train")
    _status(page, "kept in this browser")
    assert "Keep this page open" in page.locator("#offline-banner").inner_text()
    assert _note(app, team) is None
    page.context.set_offline(False)
    page.evaluate("window.dispatchEvent(new Event('online'))")
    _saved(page)
    assert _note(app, team) == "on the train"
    assert "Back online" in page.locator("#offline-banner").inner_text()


@pytest.mark.slow
def test_colored_note_survives_a_reload_before_it_could_save(app, open_as, team):
    page = open_as("Ann")
    page.route("**/notes-runs", lambda route: route.abort())
    page.locator(".rich-notes:visible").click()
    page.keyboard.type("draft")
    _status(page, "not saved")
    page.unroute("**/notes-runs")
    page.on("dialog", lambda d: d.accept())
    page.reload()
    page.evaluate("showPaperTab('col')")
    page.wait_for_function(f"localStorage.getItem('rich-notes-draft:{ARXIV}:{team['id']}') === null")
    assert page.locator(".rich-notes:visible").inner_text() == "draft"
    assert _note(app, team) == "draft"


@pytest.mark.slow
def test_a_note_changed_while_you_were_offline_keeps_both(app, open_as, team):
    page = open_as("Ann")
    page.context.set_offline(True)
    page.locator(".rich-notes:visible").click()
    page.keyboard.type("mine")
    _status(page, "kept in this browser")
    with app.app_context():
        pp = db.session.get(ProjectPaper, team["pp"])
        pp.notes = "theirs"
        db.session.commit()
    page.context.set_offline(False)
    page.evaluate("window.dispatchEvent(new Event('online'))")
    _status(page, "your text is added below")
    assert _note(app, team) == "theirs\n\nmine"


@pytest.mark.slow
def test_wifi_that_passes_nothing_counts_as_offline(open_as):
    """The browser still says it's online here; only a failed request gives it away."""
    page = open_as("Ann")
    page.route("**/paper/**", lambda route: route.abort())
    page.locator(".rich-notes:visible").click()
    page.keyboard.type("x")
    _status(page, "kept in this browser")
    assert page.evaluate("navigator.onLine")
    assert page.locator("#offline-banner").is_visible()


@pytest.mark.slow
def test_someone_elses_open_note_is_locked_and_then_updates(app, open_as, team):
    ann, bea = open_as("Ann"), open_as("Bea")
    ann.locator(".rich-notes:visible").click()
    bea.wait_for_function("document.querySelector('.rich-notes-status').textContent === 'Ann is writing here'",
                          timeout=8_000)
    assert bea.locator(".rich-notes:visible").get_attribute("contenteditable") == "false"

    ann.keyboard.type("hello")
    _saved(ann)
    ann.locator(".rich-notes:visible").blur()
    bea.wait_for_function("document.querySelector('.rich-notes:not([contenteditable=false])') !== null", timeout=8_000)
    bea.wait_for_function("document.querySelector('.rich-notes').textContent === 'hello'", timeout=8_000)


@pytest.mark.slow
def test_notify_saves_the_note_first_and_sends_the_message(app, open_as, team):
    page = open_as("Ann")
    page.locator(".rich-notes:visible").click()
    page.keyboard.type("just typed")
    card = page.locator(".paper-card:visible")
    card.locator("details.card-menu summary").click()
    with patch("app.crawl.send_notification", return_value=True) as send:
        card.get_by_role("button", name="Notify").click()
        card.locator(".notify-box textarea").fill("you should read this")
        card.locator(".notify-send").click()
        card.get_by_text("✓ Sent").wait_for()
    html = send.call_args.kwargs["content_html"]
    assert "you should read this" in html and "just typed" in html and "Ann writes" in html
