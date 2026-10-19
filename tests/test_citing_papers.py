"""
Papers citing My Collection as a third crawl source (get_citing_papers, gather_candidates), and how the
crawl records and shows the ways a paper was found (ProjectPaper.found_via, the card's "Found via" line).

Semantic Scholar is a mocked httpx transport; the arXiv search and the recommendations are patched.
"""

import contextlib
import json
import re
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


def test_checks_only_the_first_capped_seeds_and_reports_which_got_an_answer(scholar, monkeypatch):
    pages, seen = scholar
    monkeypatch.setattr(crawl, "CITATION_MAX_SEEDS", 3)
    monkeypatch.setattr(crawl.time, "sleep", lambda s: None)
    pages["2401.00001"] = [{"data": []}]
    real = crawl._http_client

    def rate_limited_for_one_seed(**kw):
        def handler(request):
            if "2401.00002" in request.url.path:
                return httpx.Response(429, json={"message": "Too Many Requests"})
            return httpx.Response(200, json={"data": []}) if "2401.00001" in request.url.path else httpx.Response(404)
        return httpx.Client(transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(crawl, "_http_client", rate_limited_for_one_seed)
    checked = []
    crawl.get_citing_papers([(f"2401.{i:05d}", f"Seed {i}") for i in range(1, 6)], D_FROM, D_TO, checked=checked)
    # 00001 answered, 00002 given up on after 429s (stays first in line next time), 00003 unknown (404, counts
    # as checked); 00004 and 00005 are over the cap
    assert checked == ["2401.00001", "2401.00003"]


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
    assert 'It cites 2 papers from My Collection: &#34;Seed Paper A&#34;; &#34;Another Seed&#34;.' in html  # tooltip


def test_email_counts_each_source_with_how_many_were_added(app, proj):
    """A paper found several ways counts in each of its rows; the total counts it once."""
    from tests.test_crawl import MockPatches

    both = {"arxiv_id": "2610.00043", "title": "Found Twice", "authors": ["A"], "abstract": "abs", "pdf_url": "",
            "published_date": date(2026, 10, 12), "year": 2026, "categories": [], "source": "arxiv",
            "found_via": [{"via": "keyword", "name": "kw"},
                          {"via": "cites", "arxiv_id": "2401.00001", "title": "Seed Paper A"}]}
    cited = dict(both, arxiv_id="2610.00044", title="Only Cites", source="semantic_scholar",
                 found_via=[{"via": "cites", "arxiv_id": "2401.00001", "title": "Seed Paper A"}])
    with app.app_context():
        with MockPatches([]), patch.object(crawl, "gather_candidates", return_value=[both, cited]), \
                patch.object(crawl, "send_notification", return_value=True) as send:
            crawl.run_crawl(proj, D_FROM, D_TO, keywords_override=["kw"])
    subject, text = send.call_args.args[:2]
    # "18 (1 added)" stays on one line next to a long source name
    assert len(re.findall(r'white-space: nowrap;">\d+<span', send.call_args.kwargs["content_html"])) == 3
    html = re.sub(r"<[^>]+>", " ", send.call_args.kwargs["content_html"])
    html = re.sub(r"\s+", " ", html)
    assert "arXiv keyword search 1 (1 added)" in html
    assert "Semantic Scholar recommendations 0 (0 added)" in html
    assert "Papers citing My Collection 2 (2 added)" in html
    assert "Total found A paper found several ways counts in each row" in html and "nothing is screened twice" in html
    assert "Papers citing My Collection: 2 (2 added)" in text and "Total found: 2" in text


def test_found_via_text_names_keywords_and_the_recommendation():
    pp = ProjectPaper(found_via=json.dumps([{"via": "keyword", "name": "sparse attention"},
                                            {"via": "keyword", "name": "state space"},
                                            {"via": "recommended"}]))
    assert pp.found_via_text() == 'keywords "sparse attention", "state space" · Semantic Scholar recommendation'
    assert ProjectPaper(found_via='[{"via": "keyword"}]').found_via_text() == "arXiv keyword search"
    long = "A" * 60
    assert ProjectPaper(found_via=json.dumps([{"via": "cites", "arxiv_id": "1", "title": long}])).found_via_text() \
        == f'cites "{"A" * 48}…"'


def test_found_via_tip_says_in_words_how_each_source_found_it():
    pp = ProjectPaper(found_via=json.dumps([{"via": "keyword", "name": "sparse attention"},
                                            {"via": "keyword", "name": "state space"},
                                            {"via": "recommended"},
                                            {"via": "cites", "arxiv_id": "1", "title": "Seed Paper A"}]))
    assert pp.found_via_tip() == (
        'Its title or abstract contains your keywords "sparse attention" and "state space" (arXiv keyword search). '
        "Semantic Scholar recommends it as similar to papers in My Collection. "
        'It cites "Seed Paper A" from My Collection.')
    assert ProjectPaper(found_via='[{"via": "keyword"}]').found_via_tip() == "The arXiv keyword search found it."


def test_the_citations_source_rotates_through_a_large_collection(app, proj, scholar, monkeypatch):
    """Each crawl checks the CITATION_MAX_SEEDS My Collection papers checked longest ago."""
    from tests.test_crawl import MockPatches

    pages, seen = scholar
    monkeypatch.setattr(crawl, "CITATION_MAX_SEEDS", 2)
    with app.app_context():
        for i in range(2, 6):
            paper = Paper(arxiv_id=f"2401.0000{i}", title=f"Seed {i}", abstract="x", source="arxiv")
            db.session.add(paper)
            db.session.flush()
            db.session.add(ProjectPaper(project_id=proj, paper_id=paper.id, manual_tag="related"))
        db.session.commit()
        per_crawl = []
        for _ in range(3):
            seen.clear()
            real_citing = crawl.get_citing_papers  # MockPatches stubs it; here the mocked Scholar answers
            with MockPatches([]), patch.object(crawl, "get_citing_papers", real_citing):
                crawl.run_crawl(proj, D_FROM, D_TO, keywords_override=["kw"])
            per_crawl.append([s[0] for s in seen])
    assert per_crawl == [["2401.00001", "2401.00002"], ["2401.00003", "2401.00004"], ["2401.00005", "2401.00001"]]


def test_the_find_papers_count_step_also_marks_the_seeds_it_checked(app, proj, client, scholar):
    import time as _time

    pages, seen = scholar
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    r = client.post("/projects/citations-proj/arxiv-count", data={
        "keywords_json": '["kw"]', "date_from": str(D_FROM), "date_to": str(D_TO), "use_arxiv": "0", "use_scholar": "0"})
    task_id = r.get_json()["task_id"]
    for _ in range(100):
        status = client.get(f"/projects/citations-proj/arxiv-count-status/{task_id}").get_json()
        if status["status"] != "running":
            break
        _time.sleep(0.05)
    assert status["status"] == "done" and seen == [("2401.00001", 0)]
    with app.app_context():
        assert ProjectPaper.query.filter_by(project_id=proj).one().citations_checked_at is not None
