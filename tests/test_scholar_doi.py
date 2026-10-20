"""
Semantic Scholar sources beyond arXiv: My Collection papers that are not on arXiv seed the recommendations and
citations by DOI or by their Semantic Scholar id (found by title), and the two sources keep bioRxiv and journal
papers instead of dropping everything without an arXiv id. DOIs are stored wherever a source gives one.

Semantic Scholar is a mocked httpx transport; no network.
"""

import contextlib
import json
from datetime import date, datetime
from unittest.mock import patch

import httpx
import pytest

from app import biorxiv, crawl, db
from app.models import Paper, Project, ProjectPaper

D_FROM, D_TO = date(2026, 10, 10), date(2026, 10, 15)


def _item(title, ext, abstract="An abstract.", venue="", oa=None, day="2026-10-12"):
    return {"title": title, "abstract": abstract, "authors": [{"name": "A"}], "publicationDate": day,
            "externalIds": ext, "venue": venue, "openAccessPdf": {"url": oa} if oa else None}


class TestScholarPaper:
    def test_arxiv_bioRxiv_journal_and_unusable_results(self):
        d = date(2026, 10, 12)
        arx = crawl._scholar_paper(_item("A", {"ArXiv": "2610.00042", "DOI": "10.1/x"}), d)
        assert (arx["arxiv_id"], arx["pdf_url"], arx["doi"]) == ("2610.00042", "https://arxiv.org/pdf/2610.00042", "10.1/x")

        bio = crawl._scholar_paper(_item("B", {"DOI": "10.64898/2026.10.10.1"}, venue="bioRxiv"), d)
        assert bio["arxiv_id"] == biorxiv.biorxiv_paper_id("10.64898/2026.10.10.1")  # same paper as a keyword find
        assert bio["pdf_url"] == "https://www.biorxiv.org/content/10.64898/2026.10.10.1v1.full.pdf"

        journal = crawl._scholar_paper(_item("C", {"DOI": "10.3389/fmicb.2026.1"}, venue="Frontiers in Microbiology",
                                             oa="https://www.frontiersin.org/articles/x/pdf"), d)
        assert journal["arxiv_id"] == crawl.journal_paper_id("10.3389/fmicb.2026.1")
        assert journal["arxiv_id"].startswith("web:doi-") and "/" not in journal["arxiv_id"] and "." not in journal["arxiv_id"]
        # its PDF is not on the open-access list: only the DOI link, nothing is downloaded from the publisher
        assert journal["pdf_url"] == "https://doi.org/10.3389/fmicb.2026.1"
        oa = crawl._scholar_paper(_item("D", {"DOI": "10.18653/v1/x"}, oa="https://aclanthology.org/x.pdf"), d)
        assert oa["pdf_url"] == "https://aclanthology.org/x.pdf"

        assert crawl._scholar_paper(_item("E", {}), d) is None                       # no DOI: nothing to key it on
        assert crawl._scholar_paper(_item("F", {"DOI": "10.1/y"}, abstract=""), d) is None  # nothing to screen

    def test_a_journal_paper_is_read_only_from_an_open_access_pdf(self):
        with patch("app.utils.fetch_pdf_text", return_value="text") as pdf:
            assert crawl.get_paper_content(crawl.journal_paper_id("10.1/x"), "https://doi.org/10.1/x") is None
            pdf.assert_not_called()


class TestSeeds:
    def test_seed_is_the_arxiv_id_else_the_doi_else_the_scholar_id(self):
        links = [ProjectPaper(paper=Paper(arxiv_id="2401.00001v2", title="On arXiv")),
                 ProjectPaper(paper=Paper(arxiv_id="web:a", title="Journal", doi="10.3389/fmicb.2024.1450804")),
                 ProjectPaper(paper=Paper(arxiv_id="web:b", title="By title", semantic_scholar_id="f" * 40)),
                 ProjectPaper(paper=Paper(arxiv_id="web:c", title="A blog post"))]
        assert crawl.collection_seeds(links) == [("2401.00001", "On arXiv"), ("DOI:10.3389/fmicb.2024.1450804", "Journal"),
                                                 ("f" * 40, "By title")]
        assert [crawl._scholar_ref(s) for s, _ in crawl.collection_seeds(links)] == [
            "ArXiv:2401.00001", "DOI:10.3389/fmicb.2024.1450804", "f" * 40]

    def test_doi_from_url(self):
        assert crawl.doi_from_url("https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2024.1450804/pdf") \
            == "10.3389/fmicb.2024.1450804"
        assert crawl.doi_from_url("https://journals.plos.org/plospathogens/article/file?id=10.1371/journal.ppat.1013546&type=printable") \
            == "10.1371/journal.ppat.1013546"
        assert crawl.doi_from_url("https://doi.org/10.1002/jrs.6402") == "10.1002/jrs.6402"
        assert crawl.doi_from_url("https://www.biorxiv.org/content/10.1101/2024.01.01.123456v2.full.pdf") \
            == "10.1101/2024.01.01.123456"
        assert crawl.doi_from_url("https://www.biorxiv.org/content/10.64898/2026.10.14.759182v1") == "10.64898/2026.10.14.759182"
        assert crawl.doi_from_url("https://dnhkng.github.io/posts/rys/") is None

    def test_recommendations_are_asked_with_doi_and_scholar_id_seeds(self):
        sent = []

        def handler(request):
            sent.append(json.loads(request.content)["positivePaperIds"])
            return httpx.Response(200, json={"recommendedPapers": [
                _item("Journal rec", {"DOI": "10.1/j"}, venue="PLoS ONE"),
                _item("bioRxiv rec", {"DOI": "10.64898/b"}, venue="bioRxiv"),
                _item("No DOI", {})]})

        real = httpx.Client
        with patch.object(crawl.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)), \
                patch.object(crawl, "_ScholarLockCtx", contextlib.nullcontext):
            recs = crawl.get_semantic_scholar_recommendations(["2401.00001", "DOI:10.1/seed", "f" * 40])
        assert sent == [["ArXiv:2401.00001", "DOI:10.1/seed", "f" * 40]]
        assert [p["title"] for p in recs] == ["Journal rec", "bioRxiv rec"]

    def test_citing_papers_of_a_doi_seed_are_kept_with_the_seed_named(self, monkeypatch):
        paths = []

        def handler(request):
            paths.append(request.url.path)
            return httpx.Response(200, json={"data": [
                {"citingPaper": _item("Journal citer", {"DOI": "10.1/c"}, venue="Bacteria")},
                {"citingPaper": _item("Too old", {"DOI": "10.1/o"}, day="2026-09-01")}]})

        real = httpx.Client
        monkeypatch.setattr(crawl, "_http_client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
        monkeypatch.setattr(crawl, "_ScholarLockCtx", contextlib.nullcontext)
        checked = []
        found = crawl.get_citing_papers([("DOI:10.3389/x", "Seed")], D_FROM, D_TO, checked=checked)
        assert paths == ["/graph/v1/paper/DOI:10.3389/x/citations"]
        assert [(p["title"], p["found_via"]) for p in found] == [
            ("Journal citer", [{"via": "cites", "seed": "DOI:10.3389/x", "title": "Seed"}])]
        assert checked == ["DOI:10.3389/x"]


@pytest.fixture
def proj(app):
    with app.app_context():
        p = Project(name="DOI Proj", slug="doi-proj", research_interest="bacterial morphology")
        db.session.add(p)
        db.session.commit()
        yield p.id
        ProjectPaper.query.filter_by(project_id=p.id).delete()
        db.session.delete(db.session.get(Project, p.id))
        Paper.query.filter(Paper.arxiv_id.like("web:t-%")).delete(synchronize_session=False)
        db.session.commit()


def _link(pid, arxiv_id, title, **paper):
    p = Paper(arxiv_id=arxiv_id, title=title, **paper)
    db.session.add(p)
    db.session.flush()
    pp = ProjectPaper(project_id=pid, paper_id=p.id, manual_tag="related")
    db.session.add(pp)
    db.session.commit()
    return pp


class TestResolve:
    def test_doi_from_the_url_else_a_checked_title_match_once(self, app, proj, monkeypatch):
        with app.app_context():
            by_url = _link(proj, "web:t-1", "Bacterial shape", pdf_url="https://doi.org/10.1002/jrs.6402")
            by_title = _link(proj, "web:t-2", "Cellpose-SAM: superhuman generalization for cellular segmentation",
                             pdf_url="https://www.biorxiv.org/content/biorxiv/early/2025/05/01/2025.04.28.651001.full.pdf")
            wrong = _link(proj, "web:t-3", "LLM Neuroanatomy: how I topped the leaderboard", pdf_url="https://x.github.io/a")
            asked = []

            def handler(request):
                asked.append(request.url.params["query"])
                if request.url.params["query"].startswith("Cellpose"):
                    return httpx.Response(200, json={"data": [{"paperId": "c" * 40, "title": "Cellpose-SAM: superhuman "
                                                     "generalization for cellular segmentation",
                                                     "externalIds": {"DOI": "10.1101/2025.04.28.651001"}}]})
                return httpx.Response(200, json={"data": [{"paperId": "d" * 40, "title": "Something else entirely"}]})

            real = httpx.Client
            monkeypatch.setattr(crawl, "_http_client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
            monkeypatch.setattr(crawl, "_ScholarLockCtx", contextlib.nullcontext)
            links = [by_url, by_title, wrong]
            crawl.resolve_scholar_ids(links)
            assert by_url.paper.doi == "10.1002/jrs.6402" and by_url.paper.scholar_lookup_at is None  # no request
            assert (by_title.paper.semantic_scholar_id, by_title.paper.doi) == ("c" * 40, "10.1101/2025.04.28.651001")
            assert wrong.paper.semantic_scholar_id is None and wrong.paper.scholar_lookup_at is not None
            assert len(asked) == 2
            crawl.resolve_scholar_ids(links)  # looked up once
            assert len(asked) == 2
            assert [s for s, _ in crawl.collection_seeds(links)] == ["DOI:10.1002/jrs.6402", "DOI:10.1101/2025.04.28.651001"]

    def test_a_429_is_tried_again_next_time(self, app, proj, monkeypatch):
        with app.app_context():
            pp = _link(proj, "web:t-4", "Some paper title here", pdf_url="https://x.github.io/b")
            real = httpx.Client
            monkeypatch.setattr(crawl, "_http_client", lambda **kw: real(
                transport=httpx.MockTransport(lambda r: httpx.Response(429)), **kw))
            monkeypatch.setattr(crawl, "_ScholarLockCtx", contextlib.nullcontext)
            crawl.resolve_scholar_ids([pp])
            assert pp.paper.scholar_lookup_at is None

    def test_a_doi_seed_is_marked_checked(self, app, proj):
        with app.app_context():
            pp = _link(proj, "web:t-5", "Journal seed", doi="10.3389/fmicb.2024.1450804")
            crawl.mark_citations_checked(proj, ["DOI:10.3389/fmicb.2024.1450804"])
            assert db.session.get(ProjectPaper, pp.id).citations_checked_at is not None


class TestStored:
    def test_a_journal_paper_from_a_recommendation_is_stored_with_its_doi(self, app, project):
        from tests.test_crawl import MockPatches
        cand = crawl._scholar_paper(_item("Label-free bacteria ID", {"DOI": "10.1/lf"}, venue="Bacteria"), date(2026, 10, 12))
        cand["found_via"] = [{"via": "recommended"}]
        _link(project.id, "web:t-seed", "Seed journal paper", doi="10.3389/seed",  # recommendations need a seed
              page_count=9, institutions="[]")  # complete, so the backfill test below leaves it alone
        with MockPatches([], relevant=True), \
                patch("app.crawl.get_semantic_scholar_recommendations", return_value=[cand]), \
                patch("app.crawl._fetch_citations", return_value=4) as citations:
            log = crawl.run_crawl(project.id, D_FROM, D_TO, keywords_override=["k"], use_arxiv=False,
                                  use_citations=False)
        assert log.papers_added == 1
        paper = Paper.query.filter_by(arxiv_id=crawl.journal_paper_id("10.1/lf")).one()
        assert (paper.source, paper.doi, paper.pdf_url, paper.citation_count) == (
            "journal", "10.1/lf", "https://doi.org/10.1/lf", 4)
        citations.assert_called_once_with("DOI:10.1/lf")

    def test_the_nightly_publisher_backfill_skips_crawl_finds(self, app):
        with app.app_context():
            paper = Paper(arxiv_id=crawl.journal_paper_id("10.1/skip"), title="J", source="journal",
                          pdf_url="https://doi.org/10.1/skip", enrich_complete=1)
            db.session.add(paper)
            db.session.commit()
            try:
                with patch("app.crawl._ArxivLockCtx"), patch("app.crawl.time.sleep"), \
                        patch("app.crawl.arxiv.Client") as client, \
                        patch("app.crawl.get_citation_count", return_value=None), \
                        patch("app.crawl.extract_paper_info_from_url") as extract:
                    client.return_value.results.return_value = []
                    crawl.backfill_missing_paper_metadata()
                extract.assert_not_called()
            finally:
                db.session.delete(db.session.get(Paper, paper.id))
                db.session.commit()

    def test_the_arxiv_search_keeps_the_journal_doi(self):
        from types import SimpleNamespace
        result = SimpleNamespace(entry_id="http://arxiv.org/abs/2610.00042v1", title="T", authors=["A"], summary="S",
                                 pdf_url="http://arxiv.org/pdf/2610.00042v1", published=datetime(2026, 10, 12),
                                 categories=["cs.LG"], doi="10.1/journal")
        with patch("app.crawl.arxiv.Client") as client, patch("app.crawl._ArxivLockCtx"):
            client.return_value.results.return_value = [result]
            papers, *_ = crawl._fetch_arxiv_window(["k"], D_FROM, D_TO, 100)
        assert papers[0]["doi"] == "10.1/journal"


def test_the_monthly_citation_refresh_covers_hand_added_arxiv_papers_and_doi_papers(app, proj):
    """It used to pick arXiv papers by source "arxiv", which left out the ones added by hand (source
    "manual"); papers that are not on arXiv are refreshed by DOI or Semantic Scholar id."""
    from scripts import nightly_crawl
    with app.app_context():
        old = datetime(2026, 6, 1)
        manual = _link(proj, "2401.77777", "Added by link", source="manual", citation_count=1, citation_fetched_at=old)
        doi = _link(proj, "web:t-6", "Journal paper", source="web", doi="10.3389/x", citation_count=1, citation_fetched_at=old)
        blog = _link(proj, "web:t-7", "A blog post", source="web", citation_count=0, citation_fetched_at=old)
        with patch.object(nightly_crawl, "get_citation_count", return_value=5) as by_arxiv, \
                patch.object(nightly_crawl, "get_citation_count_by_scholar_id", return_value=9) as by_ref, \
                patch.object(nightly_crawl.time, "sleep"):
            nightly_crawl._refresh_collection_citations()
        by_arxiv.assert_called_once_with("2401.77777")
        by_ref.assert_called_once_with("DOI:10.3389/x")
        assert (manual.paper.citation_count, doi.paper.citation_count, blog.paper.citation_count) == (5, 9, 0)
        ProjectPaper.query.filter_by(project_id=proj).delete()
        db.session.delete(Paper.query.filter_by(arxiv_id="2401.77777").one())
        db.session.commit()
