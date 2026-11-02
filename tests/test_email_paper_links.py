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
    html = c.get(f"/projects/{slug}?paper={ids['trash']}").get_data(as_text=True)
    assert "moved to Trash on Nov 01 (not relevant)" in html
    assert re.search(r'data-pp-row="\d+"', html)


def test_a_paper_not_in_the_project_and_a_page_without_one(c, focus_project):
    _, slug, _ = focus_project
    assert "no longer in this project" in c.get(f"/projects/{slug}?paper=999999").get_data(as_text=True)
    assert "paper-focus-note" not in c.get(f"/projects/{slug}").get_data(as_text=True)
