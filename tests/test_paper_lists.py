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
    assert f'section=new&offset={PAGE_SIZE}' in html and 'hx-trigger="intersect once"' in html
    assert "New Papers <span id=\"new-count\">(25)</span>" in html


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
    assert '<span id="new-count" hx-swap-oob="true">(24)</span>' in html
    assert '<span id="col-count" hx-swap-oob="true">(4)</span>' in html


def test_trash_and_restore_update_the_trash_and_new_sections(c, big_project):
    html = c.post("/paper/2602.00003/trash", data={"project_id": big_project}, headers={"HX-Request": "true"}).data.decode()
    assert 'id="trash-section" hx-swap-oob="true"' in html and "New 3" in _section(html, "trash")
    html = c.post("/paper/2602.00003/restore", data={"project_id": big_project}, headers={"HX-Request": "true"}).data.decode()
    assert 'id="new-section" hx-swap-oob="true"' in html and "2602-00003" in _cards(_section(html, "new"))
    assert '<span id="trash-count" hx-swap-oob="true">(0)</span>' in html


def test_tag_toggle_returns_the_card_and_fresh_tag_counts(c, big_project):
    html = c.post("/paper/2603.00002/tag", data={"project_id": big_project, "tag": "to_discuss"},
                  headers={"HX-Request": "true"}).data.decode()
    assert html.lstrip().startswith('<div class="paper-card"')
    assert 'id="col-tag-pills" hx-swap-oob="true"' in html and "1 to discuss" in html


def test_bad_section_or_offset_is_rejected(c, big_project):
    assert c.get("/projects/pl-proj/papers/more?section=nope&offset=0").status_code == 400
    assert c.get("/projects/pl-proj/papers/more?section=new&offset=-1").status_code == 400
    assert c.get("/projects/pl-proj/papers/section?section=nope").status_code == 400


# ── Browser (slow): scrolling loads more, collecting keeps your place ─────────

@pytest.mark.slow
def test_scroll_loads_more_and_collect_keeps_place_on_a_phone(live_server, browser_session, big_project):
    browser, state = browser_session
    from playwright.sync_api import sync_playwright  # noqa: F401  (browser_session skips without playwright)
    context = browser.new_context(storage_state=state, viewport={"width": 390, "height": 844}, is_mobile=True)
    pg = context.new_page()
    pg.goto(f"{live_server}/projects/pl-proj")
    count = lambda: pg.locator("#new-papers-list > .paper-card").count()
    assert count() == PAGE_SIZE
    pg.locator("#new-papers-list").scroll_into_view_if_needed()  # the list itself on screen, as when reading it
    pg.locator("#new-papers-list").evaluate("el => el.scrollTop = el.scrollHeight")
    pg.wait_for_function(f"() => document.querySelectorAll('#new-papers-list > .paper-card').length > {PAGE_SIZE}")

    card = pg.locator('[id^="paper-card-2602-00012-"]')
    card.scroll_into_view_if_needed()
    scroll_before = pg.locator("#new-papers-list").evaluate("el => el.scrollTop")
    loaded_before = count()
    card.get_by_role("button", name="Move to My Collection").click()
    pg.wait_for_function("() => !document.querySelector('#new-papers-list [id^=\"paper-card-2602-00012-\"]')")
    assert pg.locator("#curated-list > .paper-card").first.get_attribute("id").startswith("paper-card-2602-00012-")
    assert count() == loaded_before - 1  # the loaded pages stay; only that card went
    assert abs(pg.locator("#new-papers-list").evaluate("el => el.scrollTop") - scroll_before) < 400
    assert pg.locator("#new-count").inner_text() == f"({N_NEW - 1})"

    # sorting is done by the server and still pages
    pg.get_by_role("button", name="Release date").first.click()
    pg.wait_for_function(f"() => document.querySelector('#new-papers-list > .paper-card').id.startsWith('paper-card-2602-{N_NEW - 1:05d}')")
    context.close()
