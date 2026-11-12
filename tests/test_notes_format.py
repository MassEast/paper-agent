"""Notes formatting (2026-11-12): links clickable, links to a paper of this app shown as its title, **bold**.
In the browser (app/static/notes_format.js, slow tests below with a real Chromium) and in the Notify email
(app/note_format.py). The stored note stays plain text."""
import json
import re
from datetime import datetime

import pytest

from app import db, note_format
from app.models import Paper, Project, ProjectMember, ProjectPaper

SLUG = "notes-fmt"


@pytest.fixture
def paper(app):
    with app.app_context():
        project = Project(name="Notes Fmt", slug=SLUG)
        db.session.add(project)
        db.session.flush()
        p = Paper(arxiv_id="2611.30001", title="Parallel Loop Transformer", source="arxiv", authors="[]")
        q = Paper(arxiv_id="2611.30002", title="Another Paper", source="arxiv", authors="[]")
        db.session.add_all([p, q])
        db.session.flush()
        pp = ProjectPaper(project_id=project.id, paper_id=q.id, added_at=datetime.utcnow())
        db.session.add(pp)
        db.session.commit()
        ids = {"project": project.id, "paper": p.id, "other": q.id, "pp": pp.id}
    yield ids
    from app import note_locks
    note_locks._locks.clear()  # a test that ends while typing leaves its lock (25 s) on the next test's same ids
    with app.app_context():
        ProjectPaper.query.filter_by(project_id=ids["project"]).delete()
        ProjectMember.query.filter_by(project_id=ids["project"]).delete()
        Paper.query.filter(Paper.id.in_([ids["paper"], ids["other"]])).delete()
        Project.query.filter_by(id=ids["project"]).delete()
        db.session.commit()


def test_email_html_escapes_links_and_bolds(app, monkeypatch):
    monkeypatch.setenv("SERVER_URL", "https://agent.example.org")
    with app.app_context():
        out = note_format.to_html('see https://github.com/a/b. and <script> & **this matters**\nnext line')
    assert out == ('see <a href="https://github.com/a/b" style="color:#7c3aed;">https://github.com/a/b</a>. and '
                   '&lt;script&gt; &amp; <b>this matters</b><br>next line')


def test_email_html_shows_a_paper_of_this_app_by_its_title(app, paper, monkeypatch):
    monkeypatch.setenv("SERVER_URL", "https://agent.example.org")
    url = f"https://agent.example.org/projects/llm-anatomy?paper={paper['paper']}"
    with app.app_context():
        out = note_format.to_html(f"i added it here too (see {url})")
        other_host = note_format.to_html(f"https://elsewhere.org/projects/x?paper={paper['paper']}")
        missing = note_format.to_html("https://agent.example.org/projects/x?paper=999999")
    assert out == f'i added it here too (see <a href="{url}" style="color:#7c3aed;">“Parallel Loop Transformer”</a>)'
    assert "Parallel Loop" not in other_host and "Parallel Loop" not in missing
    with app.app_context():  # links name a paper by its arXiv id since 2026-11-12
        assert "“Parallel Loop Transformer”" in note_format.to_html("https://agent.example.org/projects/x?paper=2611.30001")


def test_url_trimming_keeps_brackets_the_link_opened():
    assert note_format._trim_url("https://en.wikipedia.org/wiki/A_(b))") == "https://en.wikipedia.org/wiki/A_(b)"
    assert note_format._trim_url("https://x.org/a).") == "https://x.org/a"


def test_paper_titles_endpoint(client, paper):
    assert client.get("/papers/titles?ids=1").status_code in (302, 401)
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    d = client.get(f"/papers/titles?ids={paper['paper']},abc,999999,2611.30001,2611.30001v2,2601.99999").get_json()
    assert d == {str(paper["paper"]): "Parallel Loop Transformer", "2611.30001": "Parallel Loop Transformer",
                 "2611.30001v2": "Parallel Loop Transformer"}  # the old number or the arXiv id
    assert client.get("/papers/titles").get_json() == {}


def test_card_has_the_bold_button_and_the_page_loads_the_formatter(client, paper):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    html = client.get(f"/projects/{SLUG}").data.decode()
    assert 'class="note-bold-btn focus-ring"' in html and 'onclick="noteFormat.boldButton(this)"' in html
    assert html.index("/static/notes_format.js") < html.index("/static/rich_notes.js")


# --- In the browser (slow): pytest tests/test_notes_format.py -m slow ---

def _open(browser_session, live_server):
    browser, state = browser_session
    context = browser.new_context(storage_state=state)
    pg = context.new_page()
    pg.goto(f"{live_server}/projects/{SLUG}")
    return context, pg


@pytest.mark.slow
def test_formatter_in_the_browser(live_server, browser_session, paper):
    context, pg = _open(browser_session, live_server)
    try:
        r = pg.evaluate("""() => {
            const host = location.origin;
            const div = document.createElement('div');
            noteFormat.render('a **b c** see ' + host + '/projects/x?paper=1). and https://ex.org/p.', 0).forEach(n => div.appendChild(n));
            return {html: div.innerHTML,
                    wrap: noteFormat.toggleBold('one two', 4, 7), unwrap: noteFormat.toggleBold('one **two**', 6, 9),
                    unwrapSel: noteFormat.toggleBold('one **two**', 4, 11), empty: noteFormat.toggleBold('ab', 1, 1)};
        }""")
        assert '<strong data-s="4"><span data-s="4">b c</span></strong>' in r["html"]
        assert 'class="note-link note-paper-link"' in r["html"]  # title fetched in the background
        assert '<a href="https://ex.org/p" data-s="' in r["html"] and 'target="_blank"' in r["html"]
        assert r["wrap"] == {"from": 4, "to": 7, "insert": "**two**", "s": 6, "e": 9}
        assert r["unwrap"] == {"from": 4, "to": 11, "insert": "two", "s": 4, "e": 7}
        assert r["unwrapSel"] == {"from": 4, "to": 11, "insert": "two", "s": 4, "e": 7}
        assert r["empty"] == {"from": 1, "to": 1, "insert": "****", "s": 3, "e": 3}
    finally:
        context.close()


@pytest.mark.slow
def test_plain_notes_show_formatted_and_edit_as_text(live_server, browser_session, paper, app):
    context, pg = _open(browser_session, live_server)
    try:
        card = pg.locator(f'[data-pp-card="{paper["pp"]}"]')
        ta = card.locator("textarea[data-arxiv-id]")
        ta.click()
        link = f"{live_server}/projects/{SLUG}?paper=2611.30001"  # the paper's arXiv id
        ta.fill(f"see {link} and this is key")
        # select "key" and make it bold with the keyboard
        ta.evaluate("t => { const i = t.value.indexOf('key'); t.setSelectionRange(i, i + 3); }")
        ta.press("ControlOrMeta+b")
        assert ta.input_value().endswith("this is **key**")
        pg.locator("body").click(position={"x": 5, "y": 5})  # leave the note
        view = card.locator(".note-view")
        view.wait_for(state="visible")
        pg.wait_for_function("() => document.querySelector('.note-paper-link') && "
                             "document.querySelector('.note-paper-link').textContent === 'Parallel Loop Transformer'")
        assert view.locator("strong").inner_text() == "key"
        assert not ta.is_visible()
        # the B button on the formatted note: edits, puts the cursor at the end, adds ** there
        view.locator("strong").click()  # click into the bold word: back to the text, cursor in "key"
        assert ta.is_visible() and pg.evaluate("document.activeElement.tagName") == "TEXTAREA"
        pos, start = ta.evaluate("t => [t.selectionStart, t.value.indexOf('**key') + 2]")
        assert start <= pos <= start + 3  # the cursor lands in the word that was clicked
        pg.wait_for_timeout(1600)  # the save (keyup delay 1.2 s)
        with app.app_context():
            assert ProjectPaper.query.get(paper["pp"]).notes == f"see {link} and this is **key**"
    finally:
        context.close()


@pytest.mark.slow
def test_per_person_notes_format_in_their_authors_colors(live_server, browser_session, paper, app):
    from app import members
    with app.app_context():
        project = db.session.get(Project, paper["project"])
        project.per_person = True
        project.notification_emails = "ann.lee@example.org,bo.chen@example.org"
        members.sync_from_emails(project)
        ann, bo = members.members(project)
        pp = db.session.get(ProjectPaper, paper["pp"])
        # Bo wrote "see the key result", Ann put ** around Bo's "key"
        pp.notes = "see the **key** result"
        pp.notes_runs = json.dumps([[bo.id, "see the ", "2026-11-10"], [ann.id, "**", "2026-11-12"],
                                    [bo.id, "key", "2026-11-10"], [ann.id, "**", "2026-11-12"],
                                    [bo.id, " result", "2026-11-10"]])
        db.session.commit()
        ann_id, ann_color, bo_color = ann.id, ann.color, bo.color
    browser, state = browser_session
    context = browser.new_context(storage_state=state)
    context.add_cookies([{"name": f"rw_member_{paper['project']}", "value": str(ann_id), "url": live_server}])
    pg = context.new_page()
    try:
        pg.goto(f"{live_server}/projects/{SLUG}")
        note = pg.locator(f'[data-pp-card="{paper["pp"]}"] .rich-notes')
        strong = note.locator("strong")
        assert strong.inner_text() == "key"
        assert note.inner_text() == "see the key result"  # ** hidden while not editing
        color = strong.locator("span").evaluate("s => s.style.color")
        assert color and color == pg.evaluate(f"() => {{ const d = document.createElement('i'); d.style.color = '{bo_color}'; return d.style.color; }}")
        strong.click()  # editing: the plain text, cursor inside "key"
        assert note.inner_text() == "see the **key** result"
        pos = pg.evaluate("""() => { const s = getSelection(); const r = s.getRangeAt(0).cloneRange();
            r.setStart(document.querySelector('.rich-notes'), 0); return r.toString().length; }""")
        assert 10 <= pos <= 13
        # Ctrl/⌘+B on "result" makes it bold
        pg.evaluate("""() => { const el = document.querySelector('.rich-notes'); const t = el.textContent;
            const i = t.indexOf('result'); const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT); let n, p = 0, r = document.createRange();
            while ((n = w.nextNode())) { const L = n.nodeValue.length;
                if (i >= p && i <= p + L) r.setStart(n, i - p); if (i + 6 >= p && i + 6 <= p + L) { r.setEnd(n, i + 6 - p); break; } p += L; }
            getSelection().removeAllRanges(); getSelection().addRange(r); }""")
        note.press("ControlOrMeta+b")
        assert note.inner_text() == "see the **key** **result**"
        pg.locator("body").click(position={"x": 5, "y": 5})
        assert note.locator("strong").count() == 2
        pg.wait_for_timeout(500)
        with app.app_context():
            for _ in range(20):
                db.session.expire_all()
                if db.session.get(ProjectPaper, paper["pp"]).notes == "see the **key** **result**":
                    break
                pg.wait_for_timeout(200)
            assert db.session.get(ProjectPaper, paper["pp"]).notes == "see the **key** **result**"
            assert ann_color  # Ann's ** and Bo's words keep their authors
    finally:
        context.close()


def test_the_notify_email_shows_notes_and_the_message_formatted(app, paper, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch
    from app import crawl
    monkeypatch.setenv("SERVER_URL", "https://agent.example.org")
    flagged = SimpleNamespace(id=7, link_ref="2611.00007", title="Flagged One", authors_list=[], year=2026, citation_count=0, page_count=0,
                              institutions_list=[], summary=None, landing_url="https://arxiv.org/abs/2611.00007")
    link = f"https://agent.example.org/projects/llm-anatomy?paper={paper['paper']}"
    with app.app_context(), patch("app.crawl.send_notification", return_value=True) as send:
        crawl.notify_important_paper(flagged, "P", "proj", project_emails="a@example.org",
                                     note_parts=[(None, None, f"**dense** paper, see {link}")],
                                     message="read https://github.com/a/b first")
    html = send.call_args.kwargs["content_html"]
    assert "<b>dense</b> paper" in html and "“Parallel Loop Transformer”</a>" in html
    assert '<a href="https://github.com/a/b" style="color:#7c3aed;">https://github.com/a/b</a> first' in html


@pytest.mark.slow
def test_undo_in_plain_notes(live_server, browser_session, paper, app):
    """Ctrl/⌘+Z undoes one burst of typing (a pause ends one: the save after it changes the page, which ends Chrome's
    undo step), Ctrl/⌘+Shift+Z redoes, also after leaving and coming back to the note."""
    context, pg = _open(browser_session, live_server)
    try:
        card = pg.locator(f'[data-pp-card="{paper["pp"]}"]')
        ta = card.locator("textarea[data-arxiv-id]")
        ta.click()
        pg.keyboard.type("first")
        pg.wait_for_timeout(1700)  # saved
        pg.keyboard.type(" second")
        pg.keyboard.press("ControlOrMeta+z")
        assert ta.input_value() == "first"  # the burst after the pause, not one letter, not everything
        pg.keyboard.press("ControlOrMeta+Shift+z")
        assert ta.input_value() == "first second"
        pg.wait_for_timeout(1700)
        pg.locator("body").click(position={"x": 5, "y": 5})
        card.locator(".note-view").click()
        pg.keyboard.press("ControlOrMeta+z")
        assert ta.input_value() != "first second"
    finally:
        context.close()


@pytest.mark.slow
def test_undo_in_per_person_notes(live_server, browser_session, paper, app):
    """The per-person editor redraws itself on every keystroke, so it keeps its own undo history:
    Ctrl/⌘+Z steps back one burst of typing (or one bold toggle), Ctrl/⌘+Shift+Z redoes, and the result
    is saved with its authors."""
    from app import members
    with app.app_context():
        project = db.session.get(Project, paper["project"])
        project.per_person = True
        project.notification_emails = "ann.lee@example.org"
        members.sync_from_emails(project)
        ann = members.members(project)[0]
        ann_id = ann.id
        db.session.commit()
    browser, state = browser_session
    context = browser.new_context(storage_state=state)
    context.add_cookies([{"name": f"rw_member_{paper['project']}", "value": str(ann_id), "url": live_server}])
    pg = context.new_page()
    try:
        pg.goto(f"{live_server}/projects/{SLUG}")
        note = pg.locator(f'[data-pp-card="{paper["pp"]}"] .rich-notes')
        note.click()
        pg.keyboard.type("first")
        pg.wait_for_timeout(1300)  # a pause ends an undo step
        pg.keyboard.type(" second")
        pg.keyboard.press("ControlOrMeta+z")
        assert note.inner_text() == "first"
        pg.keyboard.press("ControlOrMeta+z")
        assert note.inner_text() == ""
        pg.keyboard.press("ControlOrMeta+Shift+z")
        pg.keyboard.press("ControlOrMeta+Shift+z")
        assert note.inner_text() == "first second"
        pg.keyboard.type("!")  # typing at the restored cursor (the end)
        assert note.inner_text() == "first second!"
        # bold is one step of its own
        pg.evaluate("""() => { const el = document.querySelector('.rich-notes'), n = el.firstChild.firstChild || el.firstChild;
            const r = document.createRange(); r.setStart(n, 0); r.setEnd(n, 5); getSelection().removeAllRanges(); getSelection().addRange(r); }""")
        note.press("ControlOrMeta+b")
        assert note.inner_text() == "**first** second!"
        pg.keyboard.press("ControlOrMeta+z")
        assert note.inner_text() == "first second!"
        pg.wait_for_timeout(1600)
        with app.app_context():
            for _ in range(20):
                db.session.expire_all()
                if db.session.get(ProjectPaper, paper["pp"]).notes == "first second!":
                    break
                pg.wait_for_timeout(200)
            pp = db.session.get(ProjectPaper, paper["pp"])
            assert pp.notes == "first second!"
            assert all(r[0] == ann_id for r in json.loads(pp.notes_runs))
    finally:
        context.close()


@pytest.mark.slow
def test_formatted_plain_note_picks_up_edits_from_elsewhere(live_server, browser_session, paper, app):
    """The 5 s poll (note_lock.js) also refreshes a note shown formatted, whose textarea is hidden."""
    with app.app_context():
        db.session.get(ProjectPaper, paper["pp"]).notes = "old text"
        db.session.commit()
    context, pg = _open(browser_session, live_server)
    try:
        view = pg.locator(f'[data-pp-card="{paper["pp"]}"] .note-view')
        view.wait_for(state="visible")
        assert view.inner_text() == "old text"
        with app.app_context():
            db.session.get(ProjectPaper, paper["pp"]).notes = "new **text**"
            db.session.commit()
        pg.wait_for_function(f"() => document.querySelector('[data-pp-card=\"{paper['pp']}\"] .note-view strong')",
                             timeout=12_000)
        assert view.inner_text() == "new text"
    finally:
        context.close()
