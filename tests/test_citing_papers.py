"""
Papers citing My Collection as a third crawl source (get_citing_papers, gather_candidates), and how the
crawl records and shows the ways a paper was found (ProjectPaper.found_via, the card's "Found via" line).

Semantic Scholar is a mocked httpx transport; the arXiv search and the recommendations are patched.
"""

import contextlib
import json
from datetime import date
from unittest.mock import patch

import httpx
import pytest

from app import crawl, db
from app.models import CrawlLog, LlmUsage, Paper, PaperSummary, Project, ProjectPaper, ScreenedPaper

D_FROM, D_TO = date(2026, 10, 10), date(2026, 10, 15)


def _citing(aid, day, title=None, arxiv=True):
    return {"citingPaper": {"title": title or f"Paper {aid}", "abstract": "abs", "authors": [{"name": "A"}],
                            "publicationDate": day, "externalIds": {"ArXiv": aid} if arxiv else {}}}


@pytest.fixture
def scholar(monkeypatch):
    """pages[seed arXiv id] = list of response bodies, one per offset page. Records every request."""
    pages, seen = {}, []

    def handler(request):
        seed = request.url.path.split("ArXiv:")[1].split("/")[0]
        seen.append((seed, int(request.url.params["offset"])))
        if seed not in pages:
            return httpx.Response(404, json={"error": "Paper not found"})
        return httpx.Response(200, json=pages[seed][int(request.url.params["offset"]) // 100])

    real = httpx.Client
    monkeypatch.setattr(crawl, "_http_client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(crawl, "_ScholarLockCtx", contextlib.nullcontext)
    return pages, seen


def test_citing_papers_in_the_window_come_back_once_with_every_cited_seed(scholar):
    pages, seen = scholar
    pages["2401.00001"] = [{"data": [
        _citing("2610.00042", "2026-10-12"),
        _citing("2610.00001", "2026-10-01"),             # before the window
        _citing("2610.99999", "2026-10-13", arxiv=False),  # not on arXiv
        _citing("2610.15219", "2026-12-01"),             # dated after the window (journal date)
    ]}]
    pages["2404.00001"] = [{"data": [_citing("2610.00042v2", "2026-10-12")]}]
    found = crawl.get_citing_papers([("2401.00001", "Seed Paper A"), ("2404.00001", "Other Seed"),
                                     ("2401.00404", "Unknown To Scholar")], D_FROM, D_TO)
    assert [p["arxiv_id"] for p in found] == ["2610.00042"]
    assert found[0]["found_via"] == [
        {"via": "cites", "arxiv_id": "2401.00001", "title": "Seed Paper A"},
        {"via": "cites", "arxiv_id": "2404.00001", "title": "Other Seed"},
    ]
    assert found[0]["source"] == "semantic_scholar" and found[0]["published_date"] == date(2026, 10, 12)
    assert ("2401.00404", 0) in seen  # a seed Scholar doesn't know is skipped, not an error


def test_pages_on_only_while_the_page_still_reaches_into_the_window(scholar):
    pages, seen = scholar
    pages["2401.00001"] = [
        {"next": 100, "data": [_citing(f"2610.1{i:04d}", "2026-10-14") for i in range(100)]},
        {"next": 200, "data": [_citing("2610.20000", "2026-10-11"), _citing("2610.20001", "2026-10-02")]},
        {"data": [_citing("2610.30000", "2026-10-12")]},  # never asked for: page 2 already went past the window
    ]
    found = crawl.get_citing_papers([("2401.00001", "Seed Paper A")], D_FROM, D_TO)
    assert seen == [("2401.00001", 0), ("2401.00001", 100)]
    assert len(found) == 101


def test_keyword_names_are_found_in_title_and_abstract():
    paper = {"title": "Sparse Attention in LMs", "abstract": "We study layer-wise  pruning of experts."}
    assert crawl._matched_keywords(paper, ["sparse attention", "layer wise pruning", "state space"]) == [
        "sparse attention", "layer wise pruning"]


def test_a_paper_found_several_ways_is_one_candidate_listing_all_of_them():
    def p(aid, title="t", abstract="", day=date(2026, 10, 12), **kw):
        return {"arxiv_id": aid, "title": title, "abstract": abstract, "published_date": day, **kw}

    arxiv_hits = [p("2610.00001", title="Sparse attention at scale"), p("2610.00002v1")]
    recs = [p("2610.00002", source="semantic_scholar"), p("2601.00009", day=date(2026, 1, 1), source="semantic_scholar")]
    citing = [p("2610.00001", source="semantic_scholar",
                found_via=[{"via": "cites", "arxiv_id": "2401.00001", "title": "Seed Paper A"}]),
              p("2610.00003", source="semantic_scholar",
                found_via=[{"via": "cites", "arxiv_id": "2401.00001", "title": "Seed Paper A"}])]
    with patch.object(crawl, "search_arxiv_with_full_papers", return_value=arxiv_hits), \
         patch.object(crawl, "get_semantic_scholar_recommendations", return_value=recs), \
         patch.object(crawl, "get_citing_papers", return_value=citing):
        out = crawl.gather_candidates(["sparse attention"], [("2401.00001", "Seed Paper A")], D_FROM, D_TO)
    by_id = {base: c for c in out for base in [crawl.base_arxiv_id(c["arxiv_id"])]}
    assert list(by_id) == ["2610.00001", "2610.00002", "2610.00003"]  # the January recommendation is outside the window
    assert by_id["2610.00001"]["found_via"] == [
        {"via": "keyword", "name": "sparse attention"},
        {"via": "cites", "arxiv_id": "2401.00001", "title": "Seed Paper A"},
    ]
    assert by_id["2610.00001"].get("source") != "semantic_scholar"  # counts as an arXiv find in the email
    assert by_id["2610.00002"]["found_via"] == [{"via": "keyword"}, {"via": "recommended"}]


def test_sources_that_are_off_are_not_asked():
    with patch.object(crawl, "search_arxiv_with_full_papers") as search, \
         patch.object(crawl, "get_semantic_scholar_recommendations") as recs, \
         patch.object(crawl, "get_citing_papers", return_value=[]) as citing:
        crawl.gather_candidates(["kw"], [("2401.00001", "Seed Paper A")], D_FROM, D_TO,
                                use_arxiv=False, use_scholar=False)
    search.assert_not_called()
    recs.assert_not_called()
    citing.assert_called_once()


def test_web_papers_are_not_seeds():
    links = [ProjectPaper(paper=Paper(arxiv_id="2401.00001", title="Seed Paper A")),
             ProjectPaper(paper=Paper(arxiv_id="web:abc", title="A blog post"))]
    assert crawl.collection_seeds(links) == [("2401.00001", "Seed Paper A")]


@pytest.fixture
def proj(app):
    with app.app_context():
        p = Project(name="Citations Proj", slug="citations-proj", research_interest="layers")
        db.session.add(p)
        db.session.flush()
        seed = Paper(arxiv_id="2401.00001", title="Seed Paper A", abstract="x", source="arxiv")
        db.session.add(seed)
        db.session.flush()
        db.session.add(ProjectPaper(project_id=p.id, paper_id=seed.id, manual_tag="related"))
        db.session.commit()
        pid = p.id
    yield pid
    with app.app_context():
        db.session.rollback()
        LlmUsage.query.filter_by(project_id=pid).delete()
        CrawlLog.query.filter_by(project_id=pid).delete()
        ScreenedPaper.query.filter_by(project_id=pid).delete()
        for pp in ProjectPaper.query.filter_by(project_id=pid).all():
            PaperSummary.query.filter_by(paper_id=pp.paper_id).delete()
            Paper.query.filter_by(id=pp.paper_id).delete()
            db.session.delete(pp)
        Project.query.filter_by(id=pid).delete()
        db.session.commit()


def test_crawl_stores_how_it_found_the_paper_and_the_card_shows_it(app, proj, client):
    from tests.test_crawl import MockPatches

    citing = {"arxiv_id": "2610.00042", "title": "A Citing Paper", "authors": ["A"], "abstract": "abs",
              "pdf_url": "", "published_date": date(2026, 10, 12), "year": 2026, "categories": [],
              "source": "semantic_scholar",
              "found_via": [{"via": "cites", "arxiv_id": "2401.00001", "title": "Seed Paper A"},
                            {"via": "cites", "arxiv_id": "2404.00001", "title": "Another Seed"}]}
    with app.app_context():
        with MockPatches([]), patch.object(crawl, "get_citing_papers", return_value=[citing]) as get_citing:
            log = crawl.run_crawl(proj, D_FROM, D_TO, keywords_override=["kw"])
            assert get_citing.call_args.args[0] == [("2401.00001", "Seed Paper A")]
        assert "citations" in log.sources_used.split(",")
        link = ProjectPaper.query.join(Paper).filter(Paper.arxiv_id == "2610.00042", ProjectPaper.project_id == proj).one()
        assert json.loads(link.found_via) == citing["found_via"]
        assert link.found_via_text() == 'cites "Seed Paper A" (+1)'

    with client.session_transaction() as sess:
        sess["logged_in"] = True
    html = client.get("/projects/citations-proj").get_data(as_text=True)
    assert 'Found via cites &#34;Seed Paper A&#34; (+1),' in html
    assert 'Found via cites &#34;Seed Paper A&#34;; &#34;Another Seed&#34;' in html  # tooltip


def test_found_via_text_names_keywords_and_the_recommendation():
    pp = ProjectPaper(found_via=json.dumps([{"via": "keyword", "name": "sparse attention"},
                                            {"via": "keyword", "name": "state space"},
                                            {"via": "recommended"}]))
    assert pp.found_via_text() == 'keywords "sparse attention", "state space" · Semantic Scholar recommendation'
    assert ProjectPaper(found_via='[{"via": "keyword"}]').found_via_text() == "arXiv keyword search"
    long = "A" * 60
    assert ProjectPaper(found_via=json.dumps([{"via": "cites", "arxiv_id": "1", "title": long}])).found_via_text() \
        == f'cites "{"A" * 48}…"'
