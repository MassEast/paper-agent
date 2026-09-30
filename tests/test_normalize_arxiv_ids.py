import importlib.util
import json
import pathlib
import sqlite3

from sqlalchemy import create_engine

from app import db

_spec = importlib.util.spec_from_file_location(
    "normalize_arxiv_ids", pathlib.Path(__file__).parent.parent / "scripts" / "normalize_arxiv_ids.py"
)
mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mig)


def _conn(tmp_path):
    path = tmp_path / "m.db"
    db.metadata.create_all(create_engine(f"sqlite:///{path}"))
    c = sqlite3.connect(path)
    c.execute("INSERT INTO projects (id, name, slug) VALUES (1, 'p', 'p'), (2, 'q', 'q')")
    return c


def _paper(c, pid, arxiv_id):
    c.execute("INSERT INTO papers (id, arxiv_id, title, pdf_url) VALUES (?, ?, 't', ?)", (pid, arxiv_id, f"https://arxiv.org/pdf/{arxiv_id}"))


def _link(c, pid, project, tag=None, tags="[]", notes=None, trashed=None):
    c.execute("INSERT INTO project_papers (paper_id, project_id, manual_tag, paper_tags, notes, trashed_at) VALUES (?,?,?,?,?,?)",
              (pid, project, tag, tags, notes, trashed))


def test_merge_versions_concats_notes_unions_tags(tmp_path):
    c = _conn(tmp_path)
    _paper(c, 1, "2402.02834v1"); _paper(c, 2, "2402.02834v2"); _paper(c, 3, "2501.00001v3")
    _link(c, 1, 1, "related", '["important"]', "note A")
    _link(c, 2, 1, "related", '["to_read"]', "note B")
    _link(c, 1, 2, None)            # only the old version is linked in project 2 -> link moves
    _link(c, 3, 1, None)
    stats = mig.normalize(c, apply=True)
    c.row_factory = sqlite3.Row

    assert sorted(r[0] for r in c.execute("SELECT arxiv_id FROM papers")) == ["2402.02834", "2501.00001"]
    assert c.execute("SELECT pdf_url FROM papers WHERE arxiv_id='2402.02834'").fetchone()[0] == "https://arxiv.org/pdf/2402.02834"
    link = c.execute("SELECT * FROM project_papers WHERE project_id=1 AND paper_id=(SELECT id FROM papers WHERE arxiv_id='2402.02834')").fetchone()
    assert json.loads(link["paper_tags"]) == ["important", "to_read"]
    assert "note A" in link["notes"] and "note B" in link["notes"]
    assert c.execute("SELECT count(*) FROM project_papers WHERE project_id=2").fetchone()[0] == 1
    assert stats["merged_groups"] == 1 and stats["links_merged"] == 1 and stats["links_moved"] == 1
    assert c.execute("PRAGMA foreign_key_check").fetchall() == []


def test_collection_beats_trash_and_dry_run_rolls_back(tmp_path):
    c = _conn(tmp_path)
    _paper(c, 1, "2402.02834"); _paper(c, 2, "2402.02834v2")
    _link(c, 1, 1, "related", trashed=None)
    _link(c, 2, 1, None, trashed="2026-01-01")
    c.commit()
    mig.normalize(c, apply=False)
    assert c.execute("SELECT count(*) FROM papers").fetchone()[0] == 2   # rolled back
    mig.normalize(c, apply=True)
    assert [tuple(r) for r in c.execute("SELECT manual_tag, trashed_at FROM project_papers")] == [("related", None)]


def test_new_plus_collection_merges_into_collection(tmp_path):
    c = _conn(tmp_path)
    _paper(c, 1, "2402.02834v1"); _paper(c, 2, "2402.02834v2")
    _link(c, 1, 1, None)                                   # old version: New Papers
    _link(c, 2, 1, "related", '["to_read"]', "my note")    # newer version: My Collection
    c.commit()
    mig.normalize(c, apply=True)
    rows = [tuple(r) for r in c.execute("SELECT manual_tag, paper_tags, notes, trashed_at FROM project_papers")]
    assert rows == [("related", '["to_read"]', "my note", None)]


def test_unversioned_id_with_versioned_pdf_url_is_cleaned(tmp_path):
    c = _conn(tmp_path)
    c.execute("INSERT INTO papers (id, arxiv_id, title, pdf_url) VALUES (1, '2402.02834', 't', 'https://arxiv.org/pdf/2402.02834v3')")
    c.commit()
    mig.normalize(c, apply=True)
    assert c.execute("SELECT pdf_url FROM papers").fetchone()[0] == "https://arxiv.org/pdf/2402.02834"
