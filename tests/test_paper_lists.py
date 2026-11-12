"""
Project page lists (app/paper_lists.py): sorted/filtered on the server, sent a page at a time, and card
actions that update only what changed instead of re-sending every list.
"""

import re
from datetime import date, datetime, timedelta

import pytest

from app import db
from app.models import Paper, Project, ProjectPaper
from app.paper_lists import PAGE_SIZE

N_NEW = 25


@pytest.fixture
def big_project(app):
    """25 New Papers (paper i added i days ago, released i days after 2026-01-01) and 3 collected ones."""
    now = datetime.utcnow()
    with app.app_context():
        project = Project(name="Lists Proj", slug="pl-proj")
        db.session.add(project)
        db.session.flush()
        ids = []
        for i in range(N_NEW):
            p = Paper(arxiv_id=f"2602.{i:05d}", title=f"New {i}", source="arxiv",
                      published_date=date(2026, 1, 1) + timedelta(days=i))
            db.session.add(p)
            db.session.flush()
            ids.append(p.id)
            db.session.add(ProjectPaper(project_id=project.id, paper_id=p.id, added_at=now - timedelta(days=i)))
        for i, tags in enumerate(['["important"]', '["to_read"]', '[]']):
            p = Paper(arxiv_id=f"2603.{i:05d}", title=f"Col {i}", source="arxiv", published_date=date(2025, 1, 1 + i))
            db.session.add(p)
            db.session.flush()
            ids.append(p.id)
            db.session.add(ProjectPaper(project_id=project.id, paper_id=p.id, manual_tag="related", paper_tags=tags,
                                        added_at=now - timedelta(days=100), collected_at=now - timedelta(days=10 + i)))
        db.session.commit()
        pid = project.id
    yield pid
    with app.app_context():
        ProjectPaper.query.filter_by(project_id=pid).delete()
        Paper.query.filter(Paper.id.in_(ids)).delete()
        Project.query.filter_by(id=pid).delete()
        db.session.commit()


@pytest.fixture
def c(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    return client


def _cards(html):
    return re.findall(r'id="paper-card-(\d{4}-\d{5})-\d+"', html)


def _section(html, sec):
    """The HTML of one section (from its opening tag to the next section)."""
    start = html.index(f'id="{sec}-section"')
    nxt = [html.find(f'id="{o}-section"', start + 1) for o in ("new", "col", "trash")]
    nxt = [n for n in nxt if n > start]
    return html[start:min(nxt) if nxt else len(html)]


def test_page_shows_first_page_of_each_list_and_a_sentinel(c, big_project):
    html = c.get("/projects/pl-proj").data.decode()
    new = _cards(_section(html, "new"))
    assert new == [f"2602-{i:05d}" for i in range(PAGE_SIZE)]  # newest added first
    # New Papers is a one-card carousel: its next page loads when the carousel nears the end, not on scroll
    assert f'section=new&offset={PAGE_SIZE}' in html and 'hx-trigger="loadmore once"' in html
    assert f'data-left="{N_NEW - PAGE_SIZE}"' in html and 'data-carousel="next"' in html
    assert '<span id="new-count" class="paper-tab-count">25</span>' in html  # in the New Papers tab


def test_more_pages_until_the_end(c, big_project):
    seen = []
    offset = 0
    while True:
        html = c.get(f"/projects/pl-proj/papers/more?section=new&offset={offset}").data.decode()
        seen += _cards(html)
        m = re.search(r"section=new&offset=(\d+)", html)
        if not m:
            break
        offset = int(m.group(1))
    assert seen == [f"2602-{i:05d}" for i in range(N_NEW)]


def test_sort_state_comes_from_the_request(c, big_project):
    html = c.get("/projects/pl-proj/papers/section?section=new&new_by=pub&new_dir=desc").data.decode()
    assert _cards(html)[0] == f"2602-{N_NEW - 1:05d}"  # latest release first
    html = c.get("/projects/pl-proj/papers/section?section=new&new_by=added&new_dir=asc").data.decode()
    assert _cards(html)[0] == f"2602-{N_NEW - 1:05d}"  # added longest ago first


def test_collection_tag_filter_and_grouping(c, big_project):
    html = c.get("/projects/pl-proj/papers/section?section=col").data.decode()
    assert _cards(html) == ["2603-00000", "2603-00001", "2603-00002"]  # collected most recently first
    html = c.get("/projects/pl-proj/papers/section?section=col&col_tags=to_read").data.decode()
    assert _cards(html) == ["2603-00001"]
    assert "1 to read" in html and "1 important" in html  # counts ignore the filter
    html = c.get("/projects/pl-proj/papers/section?section=col&col_group=1").data.decode()
    assert _cards(html) == ["2603-00001", "2603-00000", "2603-00002"]  # to read → important → untagged


def test_collect_removes_card_and_redraws_only_my_collection(c, big_project):
    """Collecting paper 15 (not on the first page): it goes to the top of My Collection right away,
    New Papers is not re-sent (the reader keeps their place), and the counts update."""
    resp = c.post("/paper/2602.00015/tag", data={"project_id": big_project, "tag": "related"},
                  headers={"HX-Request": "true"})
    html = resp.data.decode()
    assert 'id="col-section" hx-swap-oob="true"' in html
    assert _cards(_section(html, "col"))[0] == "2602-00015"
    assert 'id="new-section"' not in html
    assert '<span id="new-count" class="paper-tab-count" hx-swap-oob="true">24</span>' in html
    assert '<span id="col-count" class="paper-tab-count" hx-swap-oob="true">4</span>' in html


def test_trash_and_restore_update_the_trash_and_new_sections(c, big_project):
    html = c.post("/paper/2602.00003/trash", data={"project_id": big_project}, headers={"HX-Request": "true"}).data.decode()
    assert 'id="trash-section" hx-swap-oob="true"' in html and "New 3" in _section(html, "trash")
    html = c.post("/paper/2602.00003/restore", data={"project_id": big_project}, headers={"HX-Request": "true"}).data.decode()
    assert 'id="new-section" hx-swap-oob="true"' in html and "2602-00003" in _cards(_section(html, "new"))
    assert '<span id="trash-count" class="paper-tab-count" hx-swap-oob="true">0</span>' in html


def test_tag_toggle_returns_the_card_and_fresh_tag_counts(c, big_project):
    html = c.post("/paper/2603.00002/tag", data={"project_id": big_project, "tag": "to_discuss"},
                  headers={"HX-Request": "true"}).data.decode()
    assert html.lstrip().startswith('<div class="paper-card"')
    assert 'id="col-tag-pills" hx-swap-oob="true"' in html and "1 to discuss" in html


def test_discuss_only_tags_and_never_emails(c, big_project, monkeypatch):
    """'Discuss' is a plain tag toggle: it used to also build a flagged-paper email (figure download inside
    the arXiv lock, ~5 s per click), which only the Notify button should send."""
    import app.crawl as crawl
    sent = []
    monkeypatch.setattr(crawl, "send_notification", lambda *a, **k: sent.append(a) or True)
    with c.application.app_context():
        Project.query.get(big_project).notification_emails = "someone@example.org"
        db.session.commit()
    html = c.get("/projects/pl-proj").data.decode()
    assert "/tag-and-notify" not in html
    assert re.search(r'hx-post="/paper/2603\.00000/tag"\s+hx-vals=\'\{"project_id": \d+, "tag": "to_discuss"\}\'', html)
    resp = c.post("/paper/2603.00000/tag", data={"project_id": big_project, "tag": "to_discuss"},
                  headers={"HX-Request": "true"})
    assert resp.status_code == 200 and "1 to discuss" in resp.data.decode()
    assert sent == []


def test_bad_section_or_offset_is_rejected(c, big_project):
    assert c.get("/projects/pl-proj/papers/more?section=nope&offset=0").status_code == 400
    assert c.get("/projects/pl-proj/papers/more?section=new&offset=-1").status_code == 400
    assert c.get("/projects/pl-proj/papers/section?section=nope").status_code == 400


# ── Browser (slow): one New Papers card at a time, collecting keeps your place ─────────

@pytest.mark.slow
def test_new_papers_carousel_steps_loads_more_and_collect_shows_next_on_a_phone(live_server, browser_session, big_project):
    browser, state = browser_session
    from playwright.sync_api import sync_playwright  # noqa: F401  (browser_session skips without playwright)
    context = browser.new_context(storage_state=state, viewport={"width": 390, "height": 844}, is_mobile=True)
    pg = context.new_page()
    pg.goto(f"{live_server}/projects/pl-proj")
    pos = lambda: pg.locator("#new-section .new-carousel-pos").inner_text()
    shown = lambda: pg.locator("#new-papers-list > .paper-card:visible")
    count = lambda: pg.locator("#new-papers-list > .paper-card").count()
    assert count() == PAGE_SIZE and shown().count() == 1 and pos() == f"1 / {N_NEW}"
    assert pg.locator('#new-section [data-carousel="prev"]').is_disabled()

    for _ in range(PAGE_SIZE - 1):  # nearing the last loaded card fetches the next page
        pg.locator('#new-section [data-carousel="next"]').click()
    pg.wait_for_function(f"() => document.querySelectorAll('#new-papers-list > .paper-card').length > {PAGE_SIZE}")
    assert pos() == f"{PAGE_SIZE} / {N_NEW}" and shown().count() == 1
    pg.keyboard.press("ArrowRight")
    pg.keyboard.press("ArrowRight")
    pg.keyboard.press("ArrowLeft")
    assert pos() == f"{PAGE_SIZE + 1} / {N_NEW}"
    assert shown().first.get_attribute("id").startswith(f"paper-card-2602-{PAGE_SIZE:05d}-")

    shown().first.get_by_role("button", name="Move to My Collection").click()
    pg.wait_for_function(f"() => !document.querySelector('#new-papers-list [id^=\"paper-card-2602-{PAGE_SIZE:05d}-\"]')")
    assert pg.locator("#curated-list > .paper-card").first.get_attribute("id").startswith(f"paper-card-2602-{PAGE_SIZE:05d}-")
    assert shown().count() == 1 and pos() == f"{PAGE_SIZE + 1} / {N_NEW - 1}"  # the next paper takes its place
    assert shown().first.get_attribute("id").startswith(f"paper-card-2602-{PAGE_SIZE + 1:05d}-")
    assert pg.locator("#new-count").inner_text() == f"{N_NEW - 1}"

    # My Collection is a one-card carousel too, in its own tab
    pg.locator(".paper-tab[data-tab='col']").click()
    assert pg.locator("#curated-list > .paper-card:visible").count() == 1
    assert pg.locator("#col-section .new-carousel-pos").inner_text() == "1 / 4"
    pg.keyboard.press("ArrowRight")
    assert pg.locator("#col-section .new-carousel-pos").inner_text() == "2 / 4"
    pg.locator(".paper-tab[data-tab='new']").click()

    # sorting is done by the server and starts again at the first card
    pg.get_by_role("button", name="Release date").first.click()
    pg.wait_for_function(f"() => document.querySelector('#new-papers-list > .paper-card:not([style*=\"none\"])').id.startsWith('paper-card-2602-{N_NEW - 1:05d}')")
    assert pos() == f"1 / {N_NEW - 1}"

    # The phone's bottom bar acts on the card shown
    first = shown().first.get_attribute("id")
    pg.locator(".mobile-triage").get_by_role("button", name="Trash").click()
    pg.wait_for_function(f"() => !document.querySelector('#new-papers-list > #{first}')")
    assert pg.locator(f"#trash-papers-list > #{first}").count() == 1  # Trash shows the full card
    assert pg.locator("#new-count").inner_text() == f"{N_NEW - 2}"
    context.close()


@pytest.mark.slow
def test_inbox_title_list_and_keys_on_a_laptop(live_server, browser_session, big_project):
    """The title list on the left shows the loaded cards; a click or J / K picks the card, C collects it,
    X trashes it, and in Trash R restores it."""
    browser, state = browser_session
    context = browser.new_context(storage_state=state, viewport={"width": 1440, "height": 900})
    pg = context.new_page()
    pg.goto(f"{live_server}/projects/pl-proj")
    items = pg.locator("#inbox-list .inbox-item")
    shown = lambda: pg.locator("#new-papers-list > .paper-card:visible").first.get_attribute("id")
    # the short test titles don't fill the list, so it loads pages until it can scroll (see the test below)
    pg.wait_for_function("() => { const s = document.getElementById('inbox-list');"
                         " return s.scrollHeight > s.clientHeight || !s.querySelector('.inbox-more'); }")
    assert items.count() >= PAGE_SIZE and "is-current" in items.first.get_attribute("class")
    # Sorting and Re-screen sit in the left column above the list; the ‹ › arrows are for phones only
    assert pg.locator("#inbox-tools #new-sort-pub").is_visible() and not pg.locator("#new-section .new-carousel-nav").is_visible()
    more = pg.locator("#inbox-list .inbox-more")
    assert not more.count() or more.inner_text() == f"{N_NEW - items.count()} more, loading as you scroll"

    items.nth(3).click()
    assert shown().startswith("paper-card-2602-00003-") and "is-current" in items.nth(3).get_attribute("class")
    pg.keyboard.press("j")
    assert shown().startswith("paper-card-2602-00004-")
    pg.keyboard.press("k")
    pg.keyboard.press("k")
    assert shown().startswith("paper-card-2602-00002-")

    pg.keyboard.press("c")  # to My Collection, which opens with it; in New Papers the next paper takes its place
    pg.wait_for_function("() => document.getElementById('paper-list').dataset.tab === 'col'")
    pg.locator(".paper-tab[data-tab='new']").click()
    assert shown().startswith("paper-card-2602-00003-") and pg.locator("#new-count").inner_text() == f"{N_NEW - 1}"
    pg.keyboard.press("x")  # trash
    pg.wait_for_function("() => !document.querySelector('#new-papers-list [id^=\"paper-card-2602-00003-\"]')")
    assert pg.locator("#trash-count").inner_text() == "1"

    pg.locator(".paper-tab[data-tab='trash']").click()
    assert pg.locator("#inbox-list .inbox-item").count() == 1
    assert pg.locator("#inbox-list .inbox-why").first.inner_text().startswith("Trashed ")
    pg.keyboard.press("r")  # restore
    pg.wait_for_function("() => document.getElementById('trash-count').textContent.trim() === '0'")

    # Scrolling the title list to its end loads the next page of cards
    pg.locator(".paper-tab[data-tab='new']").click()
    pg.locator("#inbox-list").evaluate("el => { el.scrollTop = el.scrollHeight; el.dispatchEvent(new Event('scroll')); }")
    pg.wait_for_function(f"() => document.querySelectorAll('#inbox-list .inbox-item').length > {PAGE_SIZE - 1}")

    # Find papers opens its side panel; keys don't act on cards while it is open
    pg.get_by_role("button", name="Find papers").click()
    assert pg.locator("#drawer-find").is_visible() and pg.locator("#arxiv-search-btn").is_visible()
    before = shown()
    pg.keyboard.press("j")
    assert shown() == before
    pg.keyboard.press("Escape")
    assert not pg.locator("#drawer-find").is_visible()
    # A side panel opens at the top, or at the top of the section a chip is about
    pg.locator(".pd-actions button[aria-label=Settings]").click()
    assert pg.locator("#drawer-settings .drawer-body").evaluate("el => el.scrollTop") == 0
    pg.keyboard.press("Escape")
    pg.locator(".pd-chip", has_text="Nightly").click()
    top = pg.locator("#drawer-settings .drawer-body").evaluate(
        "el => [...el.children].find(c => c.querySelector('[id^=crawl-section-]')).getBoundingClientRect().top - el.getBoundingClientRect().top")
    assert 0 <= top <= 20
    context.close()


def test_nightly_crawl_off_banner_says_what_it_means(c, big_project):
    html = c.get("/projects/pl-proj").data.decode()
    banner = re.sub(r"\s+", " ", html.split('id="crawl-off-banner"')[1].split("</ol>")[0])
    assert "gets no new papers on its own" in banner and "early every morning" in banner
    assert "add them manually" in banner and "To turn it on:" in banner
    steps = re.findall(r'<li class="(done|)"><span class="crawl-step-mark">(.)</span>\s*<span><b>([^<]+)</b>( \(required\))?', banner)
    assert [(label, done == "done", bool(req)) for done, _, label, req in steps] == [
        ("Research interest", False, True), ("Search keywords", False, True),
        ("A few papers in My Collection", True, False), ("An email address", False, False)]


@pytest.mark.slow
def test_a_title_list_that_does_not_fill_its_box_loads_more_by_itself(live_server, browser_session, big_project):
    """On a tall screen the first page of titles doesn't fill the list, so it can't be scrolled to load more:
    it loads the next pages until it can (or none are left); the "N more" row loads on a click too."""
    browser, state = browser_session
    context = browser.new_context(storage_state=state, viewport={"width": 1440, "height": 2600})
    pg = context.new_page()
    try:
        pg.goto(f"{live_server}/projects/pl-proj")
        pg.wait_for_function(f"() => document.querySelectorAll('#inbox-list .inbox-item').length > {PAGE_SIZE}")
        side = pg.locator("#inbox-list")
        pg.wait_for_function("() => { const s = document.getElementById('inbox-list');"
                             " return s.scrollHeight > s.clientHeight || !s.querySelector('.inbox-more'); }")
        n = pg.locator("#inbox-list .inbox-item").count()
        more = side.locator(".inbox-more")
        if more.count():
            more.click()
            pg.wait_for_function(f"() => document.querySelectorAll('#inbox-list .inbox-item').length > {n}")
    finally:
        context.close()


@pytest.mark.slow
def test_a_collected_paper_is_shown_in_my_collection_and_new_papers_goes_on_after_it(live_server, browser_session,
                                                                                     big_project):
    """C (or the button) on a New Papers card: My Collection opens with that paper shown (its tags can be set
    right away; none is set by itself); back in New Papers the paper that came after it is shown."""
    browser, state = browser_session
    context = browser.new_context(storage_state=state, viewport={"width": 1440, "height": 900})
    pg = context.new_page()
    try:
        pg.goto(f"{live_server}/projects/pl-proj")
        visible = lambda lst: pg.locator(f"#{lst} > .paper-card:visible").first.get_attribute("id")
        pg.locator("#inbox-list .inbox-item").nth(2).click()
        assert visible("new-papers-list").startswith("paper-card-2602-00002-")
        pg.keyboard.press("c")
        pg.wait_for_function("() => document.getElementById('paper-list').dataset.tab === 'col'")
        assert visible("curated-list").startswith("paper-card-2602-00002-")
        card = pg.locator("#curated-list > .paper-card:visible").first
        assert "to read" in card.locator(".card-actions").inner_text().lower()  # its tag buttons are there to set
        pg.locator(".paper-tab[data-tab='new']").click()
        assert visible("new-papers-list").startswith("paper-card-2602-00003-")
    finally:
        context.close()
