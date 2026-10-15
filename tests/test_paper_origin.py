"""
How a paper came into a project (ProjectPaper.origin), shown on its card: found by a crawl, added by hand,
copied / moved from another project, or unknown for older links where it can't be told safely.
"""

import re
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from app import ORIGIN_BACKFILL_SQL, db
from app.models import Paper, Project, ProjectPaper


@pytest.fixture
def two(app):
    with app.app_context():
        a = Project(name="Origin A", slug="origin-a", research_interest="attention")
        b = Project(name="Origin B", slug="origin-b", research_interest="robots")
        db.session.add_all([a, b])
        db.session.commit()
        ids = {"a": a.id, "b": b.id}
    yield ids
    with app.app_context():
        pps = ProjectPaper.query.filter(ProjectPaper.project_id.in_([ids["a"], ids["b"]])).all()
        paper_ids = {pp.paper_id for pp in pps}
        for pp in pps:
            db.session.delete(pp)
        Paper.query.filter(Paper.id.in_(paper_ids)).delete()
        Project.query.filter(Project.id.in_([ids["a"], ids["b"]])).delete()
        db.session.commit()


@pytest.fixture
def c(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    return client


def _paper(arxiv_id, source):
    p = Paper(arxiv_id=arxiv_id, title=f"T {arxiv_id}", source=source, abstract="A.")
    db.session.add(p)
    db.session.flush()
    return p


def _date_line(html, arxiv_id):
    card = html.split(f'id="paper-card-{arxiv_id.replace(".", "-").replace(":", "-")}-')[1]
    m = re.search(r'data-tip="([^"]*)">\s*<svg[^>]*>\s*<use href="/static/icons.svg#icon-calendar"></use>\s*</svg>\s*([^<]*)<', card)
    return re.sub(r"\s+", " ", m.group(2)).strip(), m.group(1)


def test_backfill_labels_only_what_is_certain(app, two):
    with app.app_context():
        crawled, by_url, web, shared = (_paper("2612.00001", "arxiv"), _paper("2612.00002", "manual"),
                                        _paper("web:origin", "web"), _paper("2612.00003", "manual"))
        for p in (crawled, by_url, web, shared):
            db.session.add(ProjectPaper(project_id=two["a"], paper_id=p.id))
        db.session.add(ProjectPaper(project_id=two["b"], paper_id=shared.id))  # in two projects: unknowable
        db.session.commit()
        db.session.execute(db.text(ORIGIN_BACKFILL_SQL))
        db.session.commit()
        origin = {pp.paper.arxiv_id: pp.origin for pp in ProjectPaper.query.filter_by(project_id=two["a"])}
        assert origin == {"2612.00001": "crawl", "2612.00002": "manual", "web:origin": "manual", "2612.00003": None}
        # a set origin is never overwritten
        ProjectPaper.query.filter_by(project_id=two["a"], paper_id=crawled.id).one().origin = "copied"
        db.session.commit()
        db.session.execute(db.text(ORIGIN_BACKFILL_SQL))
        db.session.commit()
        assert ProjectPaper.query.filter_by(project_id=two["a"], paper_id=crawled.id).one().origin == "copied"


def test_adding_by_url_records_manual(c, app, two):
    with patch("app.routes.projects.start_collection_add_async"):
        c.post("/projects/origin-a/collection-add", data={"url": "https://arxiv.org/abs/2612.00009"})
    with app.app_context():
        pp = ProjectPaper.query.filter_by(project_id=two["a"]).one()
        assert pp.origin == "manual"


def test_copy_and_move_record_where_from(c, app, two):
    with app.app_context():
        p = _paper("2612.00004", "arxiv")
        db.session.add(ProjectPaper(project_id=two["a"], paper_id=p.id, manual_tag="related", origin="crawl"))
        q = _paper("2612.00005", "arxiv")
        db.session.add(ProjectPaper(project_id=two["a"], paper_id=q.id, manual_tag="related", origin="crawl"))
        db.session.commit()
    with patch("app.routes.papers.start_relevance_reason_async"):
        c.post("/paper/2612.00004/transfer", data={"project_id": two["a"], "target_project_id": two["b"], "mode": "copy"})
        c.post("/paper/2612.00005/transfer", data={"project_id": two["a"], "target_project_id": two["b"], "mode": "move"},
               headers={"HX-Request": "true"})
    with app.app_context():
        got = {pp.paper.arxiv_id: (pp.origin, pp.origin_project_id) for pp in ProjectPaper.query.filter_by(project_id=two["b"])}
        assert got == {"2612.00004": ("copied", two["a"]), "2612.00005": ("moved", two["a"])}
    html = c.get("/projects/origin-b").data.decode()
    text, tip = _date_line(html, "2612.00004")
    assert text.startswith("Copied from Origin A") and "Collected" not in text and "Copied here from the project Origin A" in tip


def test_card_line_per_origin(c, app, two):
    now = datetime.utcnow()
    with app.app_context():
        for arxiv_id, origin, added, collected in [
            ("2612.00006", "crawl", now - timedelta(days=3), now),
            ("2612.00007", "manual", now, now),
            ("2612.00008", None, now - timedelta(days=1), now - timedelta(days=1)),
        ]:
            p = _paper(arxiv_id, "arxiv")
            db.session.add(ProjectPaper(project_id=two["a"], paper_id=p.id, manual_tag="related", origin=origin,
                                        added_at=added, collected_at=collected))
        db.session.commit()
    html = c.get("/projects/origin-a").data.decode()
    text, tip = _date_line(html, "2612.00006")
    assert text.startswith("Found by crawl") and "| Collected" in text and "Found by a crawl" in tip
    text, tip = _date_line(html, "2612.00007")
    assert text.startswith("Added by hand") and "Collected" not in text and "pasted link" in tip
    text, tip = _date_line(html, "2612.00008")
    assert text.startswith("Added ") and "| Collected" in text and "how is not known" in tip
