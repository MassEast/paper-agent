"""Paper titles in the crawl email open the paper's card on the project page (?paper=<Paper.id>), wherever
the paper is by then: still in New Papers, collected by someone, or trashed. The email also lists every
other paper the crawl added (since 2026-11-02)."""

import re
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app import crawl, db
from app.models import Paper, Project, ProjectPaper
from app.paper_lists import PAGE_SIZE


def _paper(i, **kw):
    return SimpleNamespace(id=i, title=f"Paper {i}", institutions_list=kw.get("insts", []), year=2026,
                           published_date=kw.get("pub"), citation_count=kw.get("cites", 0))


def _send(top, added):
    with patch("app.crawl.send_notification", return_value=True) as send, \
            patch.object(crawl, "_fetch_figure_html", return_value=""):
        crawl.notify_crawl_complete("P", "proj", 9, len(added), [top] if top else [], added_papers=added)
    return send.call_args.kwargs["content_html"], send.call_args.args[1]


def test_featured_title_and_the_other_added_papers_link_to_their_cards():
    featured = _paper(1)
    other = _paper(2, insts=["EPFL", "MIT", "CMU"], pub=date(2026, 10, 30), cites=3)
    html, text = _send(featured, [featured, other])
    assert html.count("/projects/proj?paper=1") == 1  # the featured title, not repeated in the table
    assert "Also added (1)" in html
    assert "/projects/proj?paper=2" in html and "EPFL, MIT et al." in html
    assert "Oct 30, 2026" in html and "3 citations" in html
    assert "arxiv.org/abs/" in html  # the arXiv link stays
    assert "- Paper 2: http" in text and "?paper=2" in text


def test_a_single_added_paper_has_no_table_and_a_long_list_is_cut():
    only = _paper(1)
    assert "Also added" not in _send(only, [only])[0]
    many = [_paper(i) for i in range(1, crawl.EMAIL_OTHERS_MAX + 6)]
    html, _ = _send(many[0], many)
    assert f"Also added ({len(many) - 1})" in html
    assert html.count("?paper=") == 1 + crawl.EMAIL_OTHERS_MAX
    assert "and 4 more" in html


@pytest.fixture
def focus_project(app):
    """15 New Papers, one collected, one trashed. Returns (project id, slug, {name: Paper.id})."""
    now = datetime.utcnow()
    with app.app_context():
        project = Project(name="Focus Proj", slug="focus-proj")
        db.session.add(project)
        db.session.flush()
        ids = {}
        for i in range(15):
            p = Paper(arxiv_id=f"2610.9{i:04d}", title=f"New {i}", source="arxiv", published_date=date(2026, 10, 1))
            db.session.add(p)
            db.session.flush()
            ids[f"new{i}"] = p.id
            db.session.add(ProjectPaper(project_id=project.id, paper_id=p.id, added_at=now - timedelta(days=i)))
        for name, extra in (("col", {"manual_tag": "related", "collected_at": datetime(2026, 11, 1, 12)}),
                            ("trash", {"trashed_at": datetime(2026, 11, 1, 12), "trashed_reason": "not relevant"})):
            p = Paper(arxiv_id=f"2610.8{len(ids):04d}", title=f"The {name} paper", source="arxiv")
            db.session.add(p)
            db.session.flush()
            ids[name] = p.id
            db.session.add(ProjectPaper(project_id=project.id, paper_id=p.id, added_at=now, **extra))
        db.session.commit()
        pid = project.id
    yield pid, "focus-proj", ids
    with app.app_context():
        ProjectPaper.query.filter_by(project_id=pid).delete()
        Paper.query.filter(Paper.id.in_(ids.values())).delete()
        Project.query.filter_by(id=pid).delete()
        db.session.commit()


@pytest.fixture
def c(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    return client


def _new_titles(html):
    section = html.split('id="new-papers-list"', 1)[1].split('id="col-section"', 1)[0]
    return re.findall(r">\s*(New \d+)\s*</a>", section)


def test_a_new_paper_opened_from_the_email_comes_first_and_pages_neither_skip_nor_repeat_it(c, focus_project):
    _, slug, ids = focus_project
    html = c.get(f"/projects/{slug}?paper={ids['new12']}").get_data(as_text=True)
    assert "still in New Papers, shown first below" in html
    first_page = _new_titles(html)
    assert first_page[0] == "New 12" and len(first_page) == PAGE_SIZE
    rest = c.get(f"/projects/{slug}/papers/more?section=new&offset={PAGE_SIZE}&paper={ids['new12']}").get_data(as_text=True)
    shown = first_page + re.findall(r">\s*(New \d+)\s*</a>", rest)
    assert sorted(shown) == sorted(f"New {i}" for i in range(15))


def test_a_collected_or_trashed_paper_says_where_it_went(c, focus_project):
    _, slug, ids = focus_project
    html = c.get(f"/projects/{slug}?paper={ids['col']}").get_data(as_text=True)
    assert "already in My Collection (since Nov 01), shown first there" in html
    # The tab switch is valid JS: the quotes around the tab name were HTML-escaped (a JS error) until 2026-11-03
    assert """showPaperTab("col" || 'new');""" in html
    html = c.get(f"/projects/{slug}?paper={ids['trash']}").get_data(as_text=True)
    assert "moved to Trash on Nov 01 (not relevant)" in html
    assert re.search(r'data-pp-card="\d+"', html)  # Trash shows full cards (inbox layout)


def test_a_paper_not_in_the_project_and_a_page_without_one(c, focus_project):
    _, slug, _ = focus_project
    assert "no longer in this project" in c.get(f"/projects/{slug}?paper=999999").get_data(as_text=True)
    assert "paper-focus-note" not in c.get(f"/projects/{slug}").get_data(as_text=True)


def _flag(project=None, flagged_by=None, emails="a@example.org"):
    paper = SimpleNamespace(id=7, title="Flagged One", authors_list=[], year=2026, citation_count=0, page_count=0,
                            institutions_list=[], summary=None, landing_url="https://arxiv.org/abs/2611.00007")
    with patch("app.crawl.send_notification", return_value=True) as send:
        assert crawl.notify_important_paper(paper, "P", "proj", project_emails=emails, project=project,
                                            flagged_by=flagged_by)
    return send


def test_the_subject_says_who_flagged_the_paper():
    assert _flag(flagged_by="Justus").call_args.args[0] == "📄 Justus flagged paper: Flagged One"
    assert _flag().call_args.args[0] == "📄 Paper flagged: Flagged One"


def test_the_flagged_paper_email_links_to_the_card():
    html = _flag().call_args.kwargs["content_html"]
    assert html.count("/projects/proj?paper=7") == 2  # the title and "Open in project"
    assert "arxiv.org/abs/2611.00007" in html  # the arXiv link stays
    assert "Flagged at" in html


def test_a_flagged_paper_in_a_per_person_project_goes_to_everyone_with_their_own_links(app):
    with app.app_context():
        from app import members
        from app.models import ProjectMember
        project = Project(name="Flag Team", slug="flag-team", per_person=True,
                          notification_emails="a@example.org,b@example.org")
        db.session.add(project)
        db.session.flush()
        ProjectMember.query.filter_by(project_id=project.id).delete()
        members.sync_from_emails(project)
        ids = {m.email: m.id for m in members.members(project)}
        send = _flag(project, "A", project.notification_emails)
        db.session.rollback()
    sent = {c.kwargs["project_emails"]: c.kwargs["content_html"] for c in send.call_args_list}
    assert set(sent) == {"a@example.org", "b@example.org"}
    for email, html in sent.items():
        assert f"/projects/proj?as={ids[email]}&paper=7" in html
    assert "Flagged by A at" in sent["b@example.org"]


def test_every_card_carries_its_paper_id_for_the_address_bar(c, focus_project):
    _, slug, ids = focus_project
    html = c.get(f"/projects/{slug}").get_data(as_text=True)
    assert f'data-paper-id="{ids["new0"]}"' in html and f'data-paper-id="{ids["col"]}"' in html


@pytest.mark.slow
def test_the_address_bar_names_the_shown_card_so_reload_and_shared_links_open_it(live_server, browser_session,
                                                                                focus_project):
    """Since 2026-11-04: picking a card writes ?paper=<Paper.id> (the emails' link) into the address bar."""
    _, slug, ids = focus_project
    browser, state = browser_session
    context = browser.new_context(storage_state=state)
    page = context.new_page()
    page.goto(f"{live_server}/projects/{slug}?as=1")
    page.wait_for_function("window.htmx !== undefined")
    assert f"paper={ids['new0']}" in page.url and "as=" not in page.url  # the first card; never someone's as=
    page.locator("#inbox-list .inbox-item").nth(2).click()
    assert page.url.endswith(f"?paper={ids['new2']}")
    page.reload()
    page.wait_for_function("window.htmx !== undefined")
    shown = page.locator("#new-papers-list > .paper-card:visible")
    assert shown.get_attribute("data-paper-id") == str(ids["new2"])
    assert page.locator("#paper-focus-note").count() == 0  # a reload is not "opened from a link"
    other = context.new_page()  # someone opening the shared link gets the note
    other.goto(page.url)
    assert other.locator("#paper-focus-note").count() == 1
    other.close()
    page.evaluate("showPaperTab('col')")
    assert page.url.endswith(f"?paper={ids['col']}")
    context.close()
