"""The paper card's layout since 2026-11-12 (mockup G2): header with title, authors (corresponding ones as mailto
links), badges, link buttons (a button only for links we found) and the vote; reading column; fact sidebar with
the paper's data, Figure 1 and connections; "Ask about this paper" in the action bar (key A)."""
import json
import re
from datetime import date, datetime
from pathlib import Path

import pytest

from app import db, members
from app.models import Paper, PaperReference, PaperSummary, Project, ProjectMember, ProjectPaper

SLUG = "layout-proj"


@pytest.fixture
def card(app):
    with app.app_context():
        project = Project(name="Layout Proj", slug=SLUG)
        db.session.add(project)
        db.session.flush()
        made = []

        def add(arxiv_id, tag=None, **kw):
            kw.setdefault("title", f"Paper {arxiv_id}")
            kw.setdefault("source", "arxiv")
            p = Paper(arxiv_id=arxiv_id, **kw)
            db.session.add(p)
            db.session.flush()
            pp = ProjectPaper(project_id=project.id, paper_id=p.id, manual_tag=tag, added_at=datetime.utcnow())
            db.session.add(pp)
            db.session.flush()
            made.append(p.id)
            return p, pp
        yield {"project": project, "add": add}
        PaperReference.query.filter(PaperReference.paper_id.in_(made)).delete()
        PaperSummary.query.filter(PaperSummary.paper_id.in_(made)).delete()
        ProjectPaper.query.filter_by(project_id=project.id).delete()
        ProjectMember.query.filter_by(project_id=project.id).delete()
        Paper.query.filter(Paper.id.in_(made)).delete()
        db.session.delete(project)
        db.session.commit()


@pytest.fixture
def c(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    return client


def _card(html, arxiv_id):
    start = html.index(f'id="paper-card-{arxiv_id.replace(".", "-").replace(":", "-")}-')
    nxt = html.find('<div class="paper-card"', start)
    return html[start:nxt if nxt > 0 else len(html)]


def _page(c, card):
    db.session.commit()
    return c.get(f"/projects/{SLUG}").data.decode()


def _buttons(card_html):
    head = card_html[card_html.index('class="card-links"'):]
    head = head[:head.index("</div>")]
    return [re.sub(r"<[^>]+>", "", t).strip() for t in re.findall(r'<a class="card-link"[^>]*>(.*?)</a>', head, re.S)]


def test_link_buttons_only_for_links_we_found_in_the_agreed_order(c, card):
    card["add"]("2611.10001", project_url="https://lens.github.io", code_url="https://github.com/a/lens")
    card["add"]("2611.10002")
    html = _page(c, card)
    assert _buttons(_card(html, "2611.10001")) == ["arXiv", "PDF", "Project page", "Code"]
    assert 'href="https://github.com/a/lens" target="_blank" rel="noopener"' in _card(html, "2611.10001")
    assert _buttons(_card(html, "2611.10002")) == ["arXiv", "PDF"]
    assert "BibTeX" not in html and "Demo" not in html


def test_web_papers_keep_source_and_pdf(c, card):
    card["add"]("web:layout-x", source="journal", doi="10.1/x", pdf_url="https://example.org/x.pdf",
                code_url="https://github.com/a/x")
    assert _buttons(_card(_page(c, card), "web:layout-x")) == ["Source ↗", "PDF", "Code"]


def test_corresponding_authors_are_mailto_links_even_when_the_list_is_shortened(c, card):
    authors = [f"Author {i}" for i in range(8)]
    card["add"]("2611.10003", authors=json.dumps(authors),
                corresponding_authors=json.dumps([{"name": "Author 5", "email": "a5@uni.edu"}]))
    html = _card(_page(c, card), "2611.10003")
    line = html[html.index('class="card-authors"'):]
    line = line[:line.index("</div>")]
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", line)).strip()
    assert "Author 0, Author 1, Author 2 … (+2) … Author 5 … (+1) … Author 7" in text
    assert '<a class="card-author-mail" href="mailto:a5@uni.edu"' in line
    assert 'data-tip="Author 3, Author 4"' in line and 'data-tip="Author 6"' in line


def test_short_author_lists_are_shown_whole_with_commas(c, card):
    card["add"]("2611.10004", authors=json.dumps(["Ann Lee", "Bo Chen"]),
                corresponding_authors=json.dumps([{"name": "Bo Chen", "email": "bo@x.org"}]))
    line = _card(_page(c, card), "2611.10004")
    line = line[line.index('class="card-authors"'):]
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", line[:line.index("</div>")]))
    assert "Ann Lee, Bo Chen" in text and 'href="mailto:bo@x.org"' in line


def _facts(card_html):
    side = card_html[card_html.index('<dl class="card-facts">'):card_html.index("</dl>")]
    return {re.sub(r"\s+", " ", dt).strip(): re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", dd)).strip()
            for dt, dd in re.findall(r"<dt>(.*?)</dt><dd[^>]*>(.*?)</dd>", side, re.S)}


def test_the_sidebar_lists_what_we_know_about_the_paper(c, card):
    card["add"]("2611.10005", published_date=date(2026, 11, 3), arxiv_version=2, arxiv_updated=date(2026, 11, 10),
                venue="ICLR 2026", page_count=9, citation_count=4, citation_fetched_at=datetime(2026, 11, 11),
                institutions=json.dumps(["Stanford University", "MIT"]))
    card["add"]("2611.10006", arxiv_version=1, arxiv_updated=date(2026, 11, 1))
    html = _page(c, card)
    facts = _facts(_card(html, "2611.10005"))
    assert list(facts) == ["Published", "Version", "Venue", "Length", "Citations", "From", "Added"]
    assert facts["Published"] == "Nov 3, 2026" and facts["Version"] == "v2, Nov 10, 2026"
    assert facts["Venue"] == "ICLR 2026" and facts["Length"] == "9 pages" and facts["From"] == "Stanford University, MIT"
    assert facts["Citations"].startswith("4 citations (as of Nov 2026)")
    assert facts["Added"].startswith("Added ")
    few = _facts(_card(html, "2611.10006"))
    assert few["Version"] == "v1" and "Venue" not in few and "Length" not in few and "From" not in few


def test_figure_1_is_shown_in_the_sidebar_and_opens_the_lightbox(c, card):
    card["add"]("2611.10007", figure_url="/figures/2611.10007.png", figure_caption="Figure 1: Overview.")
    card["add"]("2611.10008")
    html = _page(c, card)
    fig = _card(html, "2611.10007")
    assert '<img src="/figures/2611.10007.png" alt="Figure 1 of the paper" loading="lazy">' in fig
    assert 'class="card-fig' in fig and "openFigureLightbox" in fig and 'data-figure-caption="Figure 1: Overview."' in fig
    assert 'class="card-fig' not in _card(html, "2611.10008")


def test_contributions_are_bullets_and_badges_sit_in_the_header(c, card):
    p, _ = card["add"]("2611.10009")
    db.session.add(PaperSummary(paper_id=p.id, summary_text="S.", main_contributions="- First thing\n- Second thing",
                                introduces_method=True, content_source="full_text"))
    html = _card(_page(c, card), "2611.10009")
    assert re.search(r"<li>First thing</li>\s*<li>Second thing</li>", html)
    head = html[html.index('class="card-head"'):html.index('class="card-grid"')]
    assert "Method" in head and "Main Contributions (" not in html


def test_ask_button_is_in_the_action_bar_and_the_chat_slot_starts_empty(c, card):
    _, pp = card["add"]("2611.10010")
    html = _card(_page(c, card), "2611.10010")
    bar = html[html.index('<div class="card-actions">'):html.index('<div id="transfer-')]
    assert 'data-act="ask"' in bar and f'hx-target="#chat-{pp.id}"' in bar and "Ask about this paper <kbd>A</kbd>" in bar
    assert re.search(rf'<div id="chat-{pp.id}">\s*</div>', html)


def test_key_a_opens_the_chat_and_is_in_the_key_hint():
    page = Path(__file__).resolve().parent.parent.joinpath("app/templates/project_detail.html").read_text()
    assert "const ACT = { c: 'keep', x: 'trash', r: 'restore', a: 'ask' };" in page
    assert "<kbd>A</kbd> ask" in page


def test_my_collection_tags_move_into_the_action_bar(c, card):
    card["add"]("2611.10011", tag="related")
    html = _card(_page(c, card), "2611.10011")
    bar = html[html.index('<div class="card-actions">'):html.index('<div id="transfer-')]
    for label in ("Important", "To Read", "Discuss", "Move to New Papers"):
        assert label in bar
    assert "Important" not in html[:html.index('<div class="card-actions">')]


def test_connections_link_to_the_cards_in_this_project(c, card):
    seed, _ = card["add"]("2401.20001", tag="related", title="Tuned Lens", authors=json.dumps(["Nora B"]))
    cited, _ = card["add"]("2401.20002", tag="related", title="Patchscopes", authors=json.dumps(["Asma G"]))
    new, pp = card["add"]("2611.10012", title="MicroLens", authors=json.dumps(["Nora B"]))
    pp.found_via = json.dumps([{"via": "cites", "arxiv_id": "2401.20001", "title": "Tuned Lens"}])
    db.session.add(PaperReference(paper_id=new.id, ref_key="arxiv:2401.20002"))
    html = _card(_page(c, card), "2611.10012")
    side = html[html.index('<aside class="card-side">'):html.index("</aside>")]
    assert "Connections" in side
    assert f'Found because it cites <a href="/projects/{SLUG}?paper={seed.id}"' in side
    assert f'Also cites <a href="/projects/{SLUG}?paper={cited.id}"' in side and "from My Collection" in side
    assert "Nora B also wrote" in side


def test_authors_who_share_papers_make_one_line_and_the_rest_unfolds(c, card):
    """Five authors of one My Collection paper: one line, three names, "and 2 more" opens to the other two
    (before 2026-11-12: a line per author, the rest only in a tooltip)."""
    team = ["Sangmin Bae", "Adam Fisch", "Hrayr Harutyunyan", "Ziwei Ji", "Tal Schuster"]
    rrt, _ = card["add"]("2410.20672", tag="related", title="Relaxed Recursive Transformers", authors=json.dumps(team))
    solo, _ = card["add"]("2401.20003", tag="related", title="Solo Paper", authors=json.dumps(["Aaron Courville"]))
    card["add"]("2507.10524", title="Mixture-of-Recursions", authors=json.dumps(team + ["Aaron Courville"]))
    side = _card(_page(c, card), "2507.10524")
    side = side[side.index('<aside class="card-side">'):side.index("</aside>")]
    text = " ".join(re.sub(r"<[^>]+>", " ", side).split())
    assert "Sangmin Bae, Adam Fisch, Hrayr Harutyunyan and 2 more Ziwei Ji, Tal Schuster also wrote Relaxed Recursive" in text
    assert '<details class="conn-more"><summary>and 2 more</summary>Ziwei Ji, Tal Schuster</details>' in side
    assert "Aaron Courville also wrote" in text and text.count("also wrote") == 2


def test_no_connections_no_section(c, card):
    card["add"]("2611.10013", authors=json.dumps(["Nobody Known"]))
    assert "Connections" not in _card(_page(c, card), "2611.10013")


def test_the_vote_sits_in_the_header_under_rate_it_in_per_person_projects(c, card):
    project = card["project"]
    project.per_person = True
    project.notification_emails = "ann.lee@example.org"
    members.sync_from_emails(project)
    card["add"]("2611.10014")
    db.session.commit()
    c.set_cookie(f"rw_member_{project.id}", str(members.members(project)[0].id))
    html = _card(c.get(f"/projects/{SLUG}").data.decode(), "2611.10014")
    head = html[html.index('class="card-head"'):html.index('class="card-grid"')]
    assert "Rate it" in head and "vote-star" in head
    assert 'class="card-people"' not in html  # nobody else marked it: no row


def test_narrow_cards_stack_the_sidebar_under_the_text():
    base = Path(__file__).resolve().parent.parent.joinpath("app/templates/base.html").read_text()
    assert ".paper-card { container-type: inline-size; container-name: card; }" in base
    assert "@container card (max-width: 720px)" in base
    assert 'grid-template-areas: "main" "side" "bottom"' in base


def test_the_share_button_next_to_the_title_links_to_this_card(c, card):
    p, _ = card["add"]("2611.40001")
    html = _card(_page(c, card), "2611.40001")
    title_to_authors = html[html.index("Paper 2611.40001"):html.index('class="card-authors"')]
    assert f'data-share="/projects/{SLUG}?paper={p.id}"' in title_to_authors
    assert "#icon-share" in title_to_authors


@pytest.mark.slow
def test_share_copies_the_link_and_the_link_opens_the_card(live_server, browser_session, app):
    """Its own data, committed: the card fixture keeps a write open, which locks the live server out."""
    with app.app_context():
        project = Project(name="Share Proj", slug="share-proj")
        db.session.add(project)
        db.session.flush()
        papers = [Paper(arxiv_id=a, title=f"Paper {a}", source="arxiv") for a in ("2611.40002", "2611.40003")]
        db.session.add_all(papers)
        db.session.flush()
        db.session.add_all([ProjectPaper(project_id=project.id, paper_id=p.id, added_at=datetime.utcnow()) for p in papers])
        db.session.commit()
        pid, ids, proj_id = papers[1].id, [p.id for p in papers], project.id
    browser, state = browser_session
    context = browser.new_context(storage_state=state, permissions=["clipboard-read", "clipboard-write"])
    pg = context.new_page()
    try:
        pg.goto(f"{live_server}/projects/share-proj?paper={pid}")
        pg.locator(f'button.card-share[data-share$="?paper={pid}"]').click()
        pg.get_by_text("Link copied").wait_for()
        url = pg.evaluate("navigator.clipboard.readText()")
        assert url == f"{live_server}/projects/share-proj?paper={pid}"
        other = context.new_page()
        other.goto(url)
        assert other.locator(f'button.card-share[data-share$="?paper={pid}"]').is_visible()  # that card is the one shown
    finally:
        context.close()
        with app.app_context():
            ProjectPaper.query.filter_by(project_id=proj_id).delete()
            Paper.query.filter(Paper.id.in_(ids)).delete()
            Project.query.filter_by(id=proj_id).delete()
            db.session.commit()
