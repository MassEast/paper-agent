"""Data behind the redesigned card (2026-11-12): arXiv version/comment (app/arxiv_meta.py), references and the
connections sidebar (app/connections.py), and the nightly passes that fill them in for older papers."""
import json
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app import arxiv_meta, connections, crawl, db, resources
from app.models import Paper, PaperReference, Project, ProjectPaper
from tests.conftest import REAL_CARD_PASSES


def _result(entry_id, updated=datetime(2026, 11, 10, 9), comment=" Accepted at ICLR 2026 ", journal_ref=""):
    return SimpleNamespace(entry_id=entry_id, updated=updated, comment=comment, journal_ref=journal_ref)


@pytest.fixture
def proj(app):
    with app.app_context():
        p = Project(name="Card Data", slug="card-data", research_interest="Interpretability of language models.")
        other = Project(name="Other Proj", slug="other-proj")
        db.session.add_all([p, other])
        db.session.commit()
        ids = {"project": p.id, "other": other.id}
    yield ids
    with app.app_context():
        pids = [pp.paper_id for pp in ProjectPaper.query.filter(ProjectPaper.project_id.in_(ids.values()))]
        from app.models import CrawlLog, LlmUsage, PaperSummary, ScreenedPaper
        PaperReference.query.filter(PaperReference.paper_id.in_(pids)).delete()
        PaperSummary.query.filter(PaperSummary.paper_id.in_(pids)).delete()
        for model in (CrawlLog, LlmUsage, ScreenedPaper):
            model.query.filter(model.project_id.in_(ids.values())).delete()
        ProjectPaper.query.filter(ProjectPaper.project_id.in_(ids.values())).delete()
        Paper.query.filter(Paper.id.in_(pids)).delete()
        Project.query.filter(Project.id.in_(ids.values())).delete()
        db.session.commit()


def _add(pid, arxiv_id, title=None, authors=("A. Author",), tag=None, trashed=False, **paper_kw):
    p = Paper.query.filter_by(arxiv_id=arxiv_id).first()
    if not p:
        p = Paper(arxiv_id=arxiv_id, title=title or arxiv_id, source="arxiv", authors=json.dumps(list(authors)),
                  **paper_kw)
        db.session.add(p)
        db.session.flush()
    pp = ProjectPaper(project_id=pid, paper_id=p.id, manual_tag=tag,
                      trashed_at=datetime.utcnow() if trashed else None)
    db.session.add(pp)
    db.session.flush()
    return p, pp


# ── arXiv version, comment, journal ref ──────────────────────────────────────

def test_from_result_reads_version_date_comment_and_journal_ref():
    meta = arxiv_meta.from_result(_result("http://arxiv.org/abs/2611.00001v3", journal_ref="TMLR 2026"))
    assert meta == {"arxiv_version": 3, "arxiv_updated": date(2026, 11, 10),
                    "arxiv_comment": "Accepted at ICLR 2026", "journal_ref": "TMLR 2026"}
    empty = arxiv_meta.from_result(_result("http://arxiv.org/abs/2611.00001", updated=None, comment="", journal_ref=None))
    assert empty == {"arxiv_version": None, "arxiv_updated": None, "arxiv_comment": None, "journal_ref": None}


def test_a_new_version_has_links_and_references_read_again(app):
    p = Paper(arxiv_id="2611.00001", title="t", arxiv_version=1, resources_checked_at=datetime.utcnow(),
              references_fetched_at=datetime.utcnow())
    arxiv_meta.apply(p, arxiv_meta.from_result(_result("http://arxiv.org/abs/2611.00001v1")))
    assert p.resources_checked_at and p.references_fetched_at  # same version: kept
    arxiv_meta.apply(p, arxiv_meta.from_result(_result("http://arxiv.org/abs/2611.00001v2")))
    assert p.arxiv_version == 2 and p.resources_checked_at is None and p.references_fetched_at is None
    assert p.arxiv_meta_at


def test_nightly_refresh_asks_in_batches_only_for_visible_stale_papers(app, proj, monkeypatch):
    with app.app_context():
        fresh, _ = _add(proj["project"], "2611.00011", arxiv_meta_at=datetime.utcnow())
        stale, _ = _add(proj["project"], "2611.00012", arxiv_meta_at=datetime.utcnow() - timedelta(days=8))
        never, _ = _add(proj["project"], "2611.00013")
        trashed, _ = _add(proj["project"], "2611.00014", trashed=True)
        web, _ = _add(proj["project"], "web:abc")
        db.session.commit()
        asked = []

        def fake_fetch(ids):
            asked.append(sorted(ids))
            return {i: arxiv_meta.from_result(_result(f"http://arxiv.org/abs/{i}v2")) for i in ids if i != "2611.00012"}
        monkeypatch.setattr(arxiv_meta, "_fetch", fake_fetch)
        monkeypatch.setattr(arxiv_meta, "META_BATCH", 1)
        assert REAL_CARD_PASSES["arxiv_meta.refresh"](batches=5) == 1
        assert asked == [["2611.00013"], ["2611.00012"]]  # never asked first, then the oldest; one per batch
        assert never.arxiv_version == 2 and never.arxiv_comment == "Accepted at ICLR 2026"
        assert stale.arxiv_version is None and stale.arxiv_meta_at > datetime.utcnow() - timedelta(minutes=1)
        assert trashed.arxiv_meta_at is None and web.arxiv_meta_at is None and fresh.arxiv_version is None


def test_nightly_refresh_stops_at_a_failed_request(app, proj, monkeypatch):
    with app.app_context():
        p, _ = _add(proj["project"], "2611.00021")
        db.session.commit()
        monkeypatch.setattr(arxiv_meta, "_fetch", MagicMock(side_effect=RuntimeError("429")))
        assert REAL_CARD_PASSES["arxiv_meta.refresh"]() == 0
        assert arxiv_meta._fetch.call_count == 1 and p.arxiv_meta_at is None  # asked again next run


def test_the_crawl_stores_arxiv_data_and_what_the_paper_says(app, proj):
    """A crawled paper gets its arXiv data from the search result and its links from the text the deep check read."""
    from tests.test_crawl import MockPatches
    paper = {"arxiv_id": "2611.00031", "title": "Linked Paper", "authors": ["Ann Lee", "Bo Chen"],
             "abstract": "We study things.", "pdf_url": "", "published_date": date(2026, 11, 1), "year": 2026,
             "categories": ["cs.LG"],
             "arxiv_meta": arxiv_meta.from_result(_result("http://arxiv.org/abs/2611.00031v2"))}
    text = ("Linked Paper Ann Lee* Bo Chen *ann@uni.edu Code: https://github.com/ann/linked "
            + "Body. " * 300)
    answer = ({"code_url": "https://github.com/ann/linked", "project_url": None,
               "corresponding": [{"author": "Ann Lee", "email": "ann@uni.edu"}], "venue": "ICLR 2026"},
              "m", 1, {"total_tokens": 50})
    with app.app_context():
        with MockPatches([paper]), patch("app.crawl.get_paper_content", return_value=text), \
                patch.object(resources, "_llm_json", return_value=answer):
            crawl.run_crawl(proj["project"], date(2026, 11, 1), date(2026, 11, 5), keywords_override=["kw"])
        p = Paper.query.filter_by(arxiv_id="2611.00031").one()
        assert (p.arxiv_version, p.arxiv_updated, p.arxiv_comment) == (2, date(2026, 11, 10), "Accepted at ICLR 2026")
        assert p.code_url == "https://github.com/ann/linked" and p.venue == "ICLR 2026"
        assert p.corresponding_emails == {"Ann Lee": "ann@uni.edu"} and p.resources_checked_at


def test_without_full_text_the_crawl_leaves_links_to_the_nightly_backfill(app, proj):
    from tests.test_crawl import MockPatches
    paper = {"arxiv_id": "2611.00032", "title": "No Text", "authors": ["A"], "abstract": "x", "pdf_url": "",
             "published_date": date(2026, 11, 1), "year": 2026, "categories": []}
    with app.app_context():
        with MockPatches([paper]), patch("app.crawl.get_paper_content", return_value=None), \
                patch.object(resources, "_llm_json", side_effect=AssertionError("no text, no call")):
            crawl.run_crawl(proj["project"], date(2026, 11, 1), date(2026, 11, 5), keywords_override=["kw"])
        assert Paper.query.filter_by(arxiv_id="2611.00032").one().resources_checked_at is None


# ── Nightly links/emails/venue backfill ──────────────────────────────────────

def test_backfill_reads_each_visible_paper_once_and_uses_the_chats_text(app, proj, monkeypatch):
    from app.models import PaperText
    with app.app_context():
        done, _ = _add(proj["project"], "2611.00041", resources_checked_at=datetime.utcnow())
        cached, _ = _add(proj["project"], "2611.00042", authors=("Ann Lee",))
        fetched, _ = _add(proj["project"], "2611.00043")
        trashed, _ = _add(proj["project"], "2611.00044", trashed=True)
        db.session.add(PaperText(paper_id=cached.id, text="Ann Lee ann@uni.edu " + "x " * 100, source="html"))
        db.session.commit()
        fetch = MagicMock(return_value=None)
        monkeypatch.setattr(crawl, "get_paper_content", fetch)
        answer = ({"corresponding": [{"author": "Ann Lee", "email": "ann@uni.edu"}]}, "m", 1, {"total_tokens": 9})
        with patch.object(resources, "_llm_json", return_value=answer):
            assert REAL_CARD_PASSES["crawl.backfill_resources"]() == 1
        assert [c.args[0] for c in fetch.call_args_list] == ["2611.00043"]  # the cached one isn't fetched
        assert cached.corresponding_emails == {"Ann Lee": "ann@uni.edu"}
        assert fetched.resources_checked_at and not trashed.resources_checked_at  # read (nothing found) / skipped
        with patch.object(resources, "_llm_json", side_effect=AssertionError("each paper once")):
            assert REAL_CARD_PASSES["crawl.backfill_resources"]() == 0
        PaperText.query.filter_by(paper_id=cached.id).delete()
        db.session.commit()


def test_backfill_stops_when_the_model_is_down(app, proj, monkeypatch):
    from app.llm import LLMUnavailableError
    with app.app_context():
        a, _ = _add(proj["project"], "2611.00051")
        db.session.commit()
        monkeypatch.setattr(crawl, "get_paper_content", lambda *a: "mail me at x@y.org " + "t " * 100)
        with patch.object(resources, "_llm_json", side_effect=LLMUnavailableError("down")):
            assert REAL_CARD_PASSES["crawl.backfill_resources"]() == 0
        assert a.resources_checked_at is None  # tried again next run


def test_the_nightly_backfill_runs_the_card_passes(app, monkeypatch):
    calls = []
    monkeypatch.setattr(arxiv_meta, "refresh", lambda: calls.append("meta") or 0)
    monkeypatch.setattr(crawl, "backfill_resources", lambda: calls.append("resources") or 0)
    monkeypatch.setattr(connections, "fetch_references", lambda: calls.append("references") or 0)
    with app.app_context(), patch.object(crawl, "redo_pdf_figures", return_value=0):
        crawl.backfill_missing_paper_metadata()
    assert calls == ["meta", "resources", "references"]  # version/comment first: the links pass reads them


# ── References and connections ───────────────────────────────────────────────

def test_reference_keys_match_papers_however_they_are_known():
    item = {"paperId": "abc", "externalIds": {"ArXiv": "2401.00001v2", "DOI": "10.1/X"}}
    assert connections.reference_keys(item) == {"arxiv:2401.00001", "doi:10.1/x", "s2:abc"}
    arxiv_doi = {"paperId": None, "externalIds": {"DOI": "10.48550/arXiv.2402.00002"}}
    assert connections.reference_keys(arxiv_doi) == {"arxiv:2402.00002"}
    web = Paper(arxiv_id="web:x", title="t", doi="10.1/X", semantic_scholar_id="def")
    assert connections.paper_keys(web) == {"doi:10.1/x", "s2:def"}


def test_references_are_fetched_in_a_batch_and_kept(app, proj, monkeypatch):
    with app.app_context():
        a, _ = _add(proj["project"], "2611.00061")
        b, _ = _add(proj["project"], "web:nodoi")  # Semantic Scholar can't know it: not asked
        c, _ = _add(proj["project"], "2611.00062")
        db.session.commit()
        resp = MagicMock(status_code=200)
        resp.json.return_value = [None, {"references": [{"paperId": "p1", "externalIds": {"ArXiv": "2401.00001"}}]}]
        call = MagicMock(return_value=resp)
        monkeypatch.setattr(crawl, "_scholar_call", call)
        assert REAL_CARD_PASSES["connections.fetch_references"]() == 1
        assert call.call_args.kwargs["json"] == {"ids": ["ArXiv:2611.00062", "ArXiv:2611.00061"]}  # newest first
        assert {r.ref_key for r in PaperReference.query.filter_by(paper_id=a.id)} == {"arxiv:2401.00001", "s2:p1"}
        assert a.references_fetched_at and c.references_fetched_at is None and c.references_attempted_at
        assert b.references_attempted_at is None
        call.reset_mock()
        assert REAL_CARD_PASSES["connections.fetch_references"]() == 0
        call.assert_not_called()  # c was just tried: again in REFERENCES_RETRY_DAYS


def test_reference_fetch_stops_on_an_error_answer(app, proj, monkeypatch):
    with app.app_context():
        a, _ = _add(proj["project"], "2611.00071")
        db.session.commit()
        monkeypatch.setattr(crawl, "_scholar_call", MagicMock(return_value=MagicMock(status_code=429, text="slow")))
        assert REAL_CARD_PASSES["connections.fetch_references"]() == 0
        assert a.references_attempted_at is None


def test_card_connections(app, proj):
    with app.app_context():
        seed, _ = _add(proj["project"], "2401.00001", "Tuned Lens", authors=("Nora B",), tag="related")
        cited, _ = _add(proj["project"], "2401.00002", "Patchscopes", authors=("Asma G",), tag="related")
        same_author, _ = _add(proj["project"], "2401.00003", "Attribution Patching", authors=("Anna Wu",), tag="related")
        _add(proj["project"], "2401.00004", "Trashed One", authors=("Anna Wu",), tag="related", trashed=True)
        new, pp = _add(proj["project"], "2611.00081", "MicroLens", authors=("Nika Z", "Anna  Wu"))
        pp.found_via = json.dumps([{"via": "cites", "arxiv_id": "2401.00001", "title": "Tuned Lens"},
                                   {"via": "cites", "seed": "DOI:10.9/gone", "title": "Gone Paper"}])
        for key in ("arxiv:2401.00001", "arxiv:2401.00002", "s2:unknown"):
            db.session.add(PaperReference(paper_id=new.id, ref_key=key))
        _add(proj["other"], "2611.00081", tag="related")
        db.session.commit()
        conn = connections.card_connections(pp)
        assert conn["found_via"][0] is seed and conn["found_via"][1] == {"title": "Gone Paper"}
        assert conn["also_cites"] == [cited]  # the found-via one isn't repeated
        assert conn["shared_authors"] == [("Anna  Wu", [same_author])]  # not the trashed paper
        assert [(p.name, sec) for p, sec in conn["other_projects"]] == [("Other Proj", "My Collection")]
