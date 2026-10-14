"""
CSV export (/project/<slug>/export.csv): the url column links the same place as the paper card.
"""

import csv
import io

import pytest

from app import db
from app.models import Paper, Project, ProjectPaper


@pytest.fixture
def export_project(app):
    """One project with an old-style arXiv id (contains a "v"), a new-style one, a web paper and a trashed one."""
    papers = [
        Paper(arxiv_id="solv-int/9901001", title="Old Style", source="arxiv"),
        Paper(arxiv_id="2601.00042", title="New Style", source="arxiv"),
        Paper(arxiv_id="web:abc123def456", title="Web Paper", source="web", pdf_url="https://aclanthology.org/x.pdf"),
        Paper(arxiv_id="2601.00043", title="Trashed", source="arxiv"),
    ]
    with app.app_context():
        project = Project(name="Export Proj", slug="ex-proj")
        db.session.add(project)
        db.session.add_all(papers)
        db.session.flush()
        for paper in papers:
            db.session.add(ProjectPaper(project_id=project.id, paper_id=paper.id,
                                        trashed_at=db.func.now() if paper.title == "Trashed" else None))
        db.session.commit()
        ids = {"project": project.id, "papers": [p.id for p in papers]}
    yield ids
    with app.app_context():
        ProjectPaper.query.filter_by(project_id=ids["project"]).delete()
        Paper.query.filter(Paper.id.in_(ids["papers"])).delete()
        Project.query.filter_by(id=ids["project"]).delete()
        db.session.commit()


def test_export_links_each_paper_like_its_card(client, export_project):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    resp = client.get("/project/ex-proj/export.csv")
    assert resp.status_code == 200
    urls = {row["title"]: row["url"] for row in csv.DictReader(io.StringIO(resp.data.decode()))}
    assert urls == {
        "Old Style": "https://arxiv.org/abs/solv-int/9901001",
        "New Style": "https://arxiv.org/abs/2601.00042",
        "Web Paper": "https://aclanthology.org/x.pdf",
    }


def test_html_responses_are_compressed(client, export_project):
    """Project pages are several MB raw; phones must get them compressed."""
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    resp = client.get("/projects/ex-proj", headers={"Accept-Encoding": "gzip"})
    assert resp.status_code == 200
    assert resp.headers.get("Content-Encoding") == "gzip"
