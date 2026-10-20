"""
bioRxiv as a crawl source (app/biorxiv.py and its place in the crawl): the keyword filter, paging, the
candidate shape, the crawl that stores such a paper, and the places that must not treat it as an arXiv paper.
All HTTP is mocked.
"""

import json
from datetime import date, datetime, timedelta
from unittest.mock import patch

import httpx
import pytest

from app import biorxiv, db
from app.models import Paper, PaperSummary, Project, ProjectPaper


@pytest.fixture(autouse=True)
def _no_cached_pages():
    from app import utils
    utils._biorxiv_pages.clear()
    yield
    utils._biorxiv_pages.clear()


def _item(doi, title="A morphology study", abstract="Antimicrobial resistance changes cell shape.",
          version="1", **extra):
    return {"doi": doi, "title": title, "abstract": abstract, "version": version, "date": "2026-10-10",
            "authors": "Smith, A. B.; Jones, C.", "author_corresponding_institution": "HTW Berlin",
            "category": "microbiology", **extra}


def _serve(monkeypatch, pages, status=200):
    """Mock api.biorxiv.org: pages[i] is the list of items for cursor i * page size."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if status != 200:
            return httpx.Response(status)
        cursor = int(request.url.path.split("/")[-2])
        size = len(pages[0])
        page = pages[cursor // size] if cursor // size < len(pages) else []
        total = sum(len(p) for p in pages)
        return httpx.Response(200, json={"messages": [{"total": str(total), "cursor": cursor}],
                                         "collection": page})

    real = biorxiv._http_client
    monkeypatch.setattr(biorxiv, "_http_client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(biorxiv.time, "sleep", lambda s: None)
    return seen


class TestKeywordMatch:
    def test_a_keyword_matches_as_a_phrase_with_plurals_and_hyphens_ignored(self):
        text = "Tabular foundation models falter on genetic data; in context learning helps"
        assert biorxiv.matched_keywords(text, ["tabular foundation model", "in-context learning"]) == [
            "tabular foundation model", "in-context learning"]

    def test_the_words_must_be_next_to_each_other_and_in_order(self):
        text = "We interpret the mechanistic basis of resistance; learning in context"
        assert biorxiv.matched_keywords(text, ["mechanistic interpretability", "in-context learning"]) == []

    def test_whole_words_only(self):
        # "generator" is not "general", "VLA" is not "Vladimir"
        assert biorxiv.matched_keywords("A general model by Vladimir", ["generator", "VLA"]) == []


class TestSearch:
    def test_pages_are_read_until_the_total_and_only_first_versions_count(self, monkeypatch):
        pages = [[_item("10.64898/a1"), _item("10.64898/a2", version="2")],
                 [_item("10.64898/a3"), _item("10.64898/a4", title="Unrelated", abstract="Whale song.")]]
        seen = _serve(monkeypatch, pages)
        found = biorxiv.search_biorxiv(["antimicrobial resistance"], date(2026, 10, 8), date(2026, 10, 12))
        assert [p["doi"] for p in found] == ["10.64898/a1", "10.64898/a3"]  # v2 and the whale paper are out
        assert len(seen) == 2

    def test_candidate_has_the_shape_the_crawl_expects(self, monkeypatch):
        _serve(monkeypatch, [[_item("10.64898/2026.04.02.715584", version="1")]])
        (p,) = biorxiv.search_biorxiv(["cell shape"], date(2026, 10, 8), date(2026, 10, 12))
        assert p["arxiv_id"].startswith("web:biorxiv-") and "/" not in p["arxiv_id"] and "." not in p["arxiv_id"]
        assert p["arxiv_id"] == biorxiv.biorxiv_paper_id("10.64898/2026.04.02.715584")
        assert p["source"] == "biorxiv"
        assert p["authors"] == ["Smith, A. B.", "Jones, C."]
        assert p["institutions"] == ["HTW Berlin"]
        assert p["pdf_url"] == "https://www.biorxiv.org/content/10.64898/2026.04.02.715584v1.full.pdf"
        assert p["published_date"] == date(2026, 10, 10) and p["year"] == 2026
        assert p["found_via"] == [{"via": "biorxiv", "name": "cell shape"}]

    def test_a_listing_that_breaks_off_keeps_what_was_read_and_warns(self, monkeypatch):
        _serve(monkeypatch, [[_item("10.64898/a1")]], status=500)
        warnings = []
        assert biorxiv.search_biorxiv(["cell shape"], date(2026, 10, 8), date(2026, 10, 12), warnings) == []
        assert len(warnings) == 1 and "bioRxiv did not answer" in warnings[0]

    def test_no_keywords_means_no_request(self, monkeypatch):
        seen = _serve(monkeypatch, [[_item("10.64898/a1")]])
        assert biorxiv.search_biorxiv([], date(2026, 10, 8), date(2026, 10, 12)) == []
        assert seen == []


class TestGather:
    def test_off_by_default_and_added_when_asked(self):
        from app.crawl import gather_candidates
        cand = {"arxiv_id": "web:biorxiv-abc123def456", "title": "T", "abstract": "A", "source": "biorxiv",
                "found_via": [{"via": "biorxiv", "name": "k"}]}
        with patch("app.crawl.search_biorxiv", return_value=[cand]) as search:
            assert gather_candidates(["k"], [], date(2026, 10, 1), date(2026, 10, 5), use_arxiv=False,
                                     use_scholar=False, use_citations=False) == []
            search.assert_not_called()
            got = gather_candidates(["k"], [], date(2026, 10, 1), date(2026, 10, 5), use_arxiv=False,
                                    use_scholar=False, use_citations=False, use_biorxiv=True)
        assert [p["arxiv_id"] for p in got] == ["web:biorxiv-abc123def456"]


class TestCrawl:
    DOI = "10.64898/2026.10.10.999001"

    def _candidate(self):
        return {
            "arxiv_id": biorxiv.biorxiv_paper_id(self.DOI), "doi": self.DOI,
            "title": "Cell shape reveals beta-lactamase resistance", "authors": ["Smith, A. B."],
            "abstract": "Resistant cells are wider.", "pdf_url": f"https://www.biorxiv.org/content/{self.DOI}v1.full.pdf",
            "published_date": date(2026, 10, 10), "year": 2026, "institutions": ["HTW Berlin"],
            "categories": ["microbiology"], "source": "biorxiv",
            "found_via": [{"via": "biorxiv", "name": "cell shape"}],
        }

    def test_a_biorxiv_paper_is_screened_on_its_pdf_and_stored_as_a_non_arxiv_paper(self, app, project):
        from app.crawl import run_crawl
        from tests.test_crawl import MockPatches

        with MockPatches([], relevant=True), \
                patch("app.crawl.search_biorxiv", return_value=[self._candidate()]), \
                patch("app.crawl.get_paper_content", return_value="Full bioRxiv text. " * 100) as content, \
                patch("app.crawl.get_paper_figure") as figure, \
                patch("app.crawl.fetch_biorxiv_figure", return_value=("https://www.biorxiv.org/x/F1.large.jpg",
                                                                      "Figure 1: Cells.")) as bio_figure, \
                patch("app.crawl._fetch_figure_html", return_value=""), \
                patch("app.crawl.get_citation_count") as arxiv_citations, \
                patch("app.crawl._fetch_citations", return_value=7) as citations:
            log = run_crawl(project.id, date(2026, 10, 8), date(2026, 10, 12), keywords_override=["cell shape"],
                            use_arxiv=False, use_scholar=False, use_citations=False, use_biorxiv=True)

        cand = self._candidate()
        content.assert_called_once_with(cand["arxiv_id"], cand["pdf_url"])  # its bioRxiv PDF
        figure.assert_not_called()           # not the arXiv figure ...
        bio_figure.assert_called_once_with(cand["pdf_url"])  # ... but figure 1 from the bioRxiv HTML
        arxiv_citations.assert_not_called()
        citations.assert_called_once_with(f"DOI:{self.DOI}")  # Semantic Scholar knows it by DOI
        assert log.status == "success" and log.papers_added == 1
        assert "biorxiv" in log.sources_used

        db.session.expire_all()
        paper = Paper.query.filter_by(arxiv_id=biorxiv.biorxiv_paper_id(self.DOI)).one()
        assert paper.source == "biorxiv" and not paper.is_arxiv
        assert paper.citation_count == 7
        assert (paper.figure_url, paper.figure_caption) == ("https://www.biorxiv.org/x/F1.large.jpg", "Figure 1: Cells.")
        assert json.loads(paper.institutions) == ["HTW Berlin"]
        assert PaperSummary.query.filter_by(paper_id=paper.id).one().content_source == "full_text"
        link = ProjectPaper.query.filter_by(project_id=project.id, paper_id=paper.id).one()
        assert link.origin == "crawl" and link.screen_source == "full_text"
        assert link.found_via_text() == 'bioRxiv keyword "cell shape"'

    def test_the_nightly_backfill_leaves_it_alone(self, app):
        from app.crawl import backfill_missing_paper_metadata

        with app.app_context():
            paper = Paper(arxiv_id="web:biorxiv-0123456789ab", title="B", source="biorxiv",
                          pdf_url="https://www.biorxiv.org/content/10.64898/x", page_count=None,
                          institutions=None, enrich_complete=1)
            db.session.add(paper)
            db.session.commit()
            try:
                with patch("app.crawl._ArxivLockCtx"), patch("app.crawl.time.sleep"), \
                        patch("app.crawl.arxiv.Client") as client, \
                        patch("app.crawl.get_citation_count", return_value=None), \
                        patch("app.crawl.extract_paper_info_from_url") as extract:
                    client.return_value.results.return_value = []
                    backfill_missing_paper_metadata()
                extract.assert_not_called()
                assert Paper.query.get(paper.id).web_enrich_attempts in (0, None)
            finally:
                db.session.delete(Paper.query.get(paper.id))
                db.session.commit()


class TestFullText:
    PDF = "https://www.biorxiv.org/content/10.64898/2026.10.10.999001v1.full.pdf"

    PAGE = ("<html><nav>Home Submit ALERTS</nav><div class=\"panel-pane pane-highwire-markup fulltext-view\">"
            "<h2>Introduction</h2><p>" + "Resistant cells grow wider. " * 200 + "</p><div id=\"F1\"><a>Download figure</a>"
            "<a>Open in new tab</a></div><h2>References</h2><p>Smith 2020. OpenUrl CrossRef PubMed Google Scholar ↵</p>"
            "<a>Back to top</a></div><div>Tweet Share Subject Areas Zoology (2420)</div></html>")

    def test_a_biorxiv_paper_is_read_from_its_html_else_its_pdf_and_no_other_web_paper_from_either(self):
        from app.crawl import get_paper_content
        with patch("app.utils.fetch_biorxiv_full_text", return_value="html text") as html, \
                patch("app.utils.fetch_pdf_text", return_value="pdf text") as pdf:
            assert get_paper_content("web:biorxiv-0123456789ab", self.PDF) == "html text"
            html.assert_called_once_with(self.PDF)
            pdf.assert_not_called()
            html.return_value = None  # a fresh preprint: no HTML version yet
            assert get_paper_content("web:biorxiv-0123456789ab", self.PDF) == "pdf text"
            pdf.assert_called_once_with(self.PDF, arxiv=False)
        with patch("app.utils.fetch_biorxiv_full_text") as html, \
                patch("app.utils.fetch_pdf_text", return_value="text") as pdf, \
                patch("app.utils.fetch_paper_full_text") as arxiv_html:
            pdf.reset_mock()
            assert get_paper_content("web:abc", "https://www.sciencedirect.com/x.pdf") is None
            assert get_paper_content("web:abc", None) is None
            pdf.assert_not_called()
        arxiv_html.assert_not_called()

    def test_the_html_text_is_the_paper_only(self):
        from app import utils
        text = utils._biorxiv_html_to_text(self.PAGE)
        assert text.startswith("Introduction Resistant cells grow wider.") and text.endswith("Smith 2020.")
        for chrome in ("Home", "Download figure", "Open in new tab", "OpenUrl", "CrossRef", "Back to top", "Tweet", "Zoology"):
            assert chrome not in text
        assert utils._biorxiv_html_to_text("<html>A page without the full text yet</html>") == ""

    def test_the_html_is_the_pdf_url_without_pdf_and_waits_its_turn(self, monkeypatch):
        from app import utils
        seen, held = [], []
        monkeypatch.setattr(utils.time, "sleep", lambda s: None)
        real = utils._http_client

        def handler(request):
            seen.append(str(request.url))
            held.append(utils._biorxiv_lock.locked())
            return httpx.Response(200, text=self.PAGE)
        monkeypatch.setattr(utils, "_http_client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
        assert utils.fetch_biorxiv_full_text(self.PDF).startswith("Introduction")
        assert seen == [self.PDF.removesuffix(".pdf")] and held == [True]
        assert seen[0].endswith("v1.full")

    def test_the_chat_reads_the_html_first(self):
        from app import chat
        paper = Paper(arxiv_id="web:biorxiv-0123456789ab", pdf_url=self.PDF)
        with patch.object(chat, "fetch_biorxiv_full_text", return_value="H" * 5000), \
                patch.object(chat, "_pdf_text") as pdf:
            assert chat._fetch_text(paper) == ("H" * 5000, "html")
            pdf.assert_not_called()
        with patch.object(chat, "fetch_biorxiv_full_text", return_value=None), \
                patch.object(chat, "_pdf_text", return_value="P" * 5000):
            assert chat._fetch_text(paper) == ("P" * 5000, "pdf")

    def test_figure_1_comes_from_the_html_with_its_caption(self):
        from app import utils
        page = ('<div class="fulltext-view"><div id="F1" class="fig"><a href="https://www.biorxiv.org/content/biorxiv/'
                'early/2026/10/16/2026.10.14.759182/F1.large.jpg?width=800&amp;height=600" data-hide-link-title="0" '
                'title="short">x</a><div class="fig-caption"><span class="fig-label">Figure 1:</span> <p>Viability of '
                'TRAIL-resistant cells (p &lt; 0.05).</p><div class="sb-div caption-clear"></div></div></div></div>')
        assert utils.biorxiv_figure(page) == (
            "https://www.biorxiv.org/content/biorxiv/early/2026/10/16/2026.10.14.759182/F1.large.jpg",
            "Figure 1: Viability of TRAIL-resistant cells (p < 0.05).")
        assert utils.biorxiv_figure("<div class='fulltext-view'>no figures</div>") == (None, None)
        assert utils.biorxiv_figure(None) == (None, None)

    def test_text_and_figure_share_one_download(self, monkeypatch):
        from app import utils
        seen = []
        monkeypatch.setattr(utils.time, "sleep", lambda s: None)
        real = utils._http_client
        monkeypatch.setattr(utils, "_http_client", lambda **kw: real(transport=httpx.MockTransport(
            lambda r: (seen.append(str(r.url)), httpx.Response(200, text=self.PAGE))[1]), **kw))
        utils.fetch_biorxiv_full_text(self.PDF)
        utils.fetch_biorxiv_figure(self.PDF)
        assert len(seen) == 1
        monkeypatch.setattr(utils, "_http_client", lambda **kw: real(transport=httpx.MockTransport(
            lambda r: (seen.append(str(r.url)), httpx.Response(200, text="<html>PDF only so far</html>"))[1]), **kw))
        fresh = self.PDF.replace("999001", "999002")
        assert utils.fetch_biorxiv_full_text(fresh) is None and utils.fetch_biorxiv_figure(fresh) == (None, None)
        assert len(seen) == 2  # "no HTML yet" is kept too

    def test_the_nightly_backfill_adds_figure_1_once_the_html_is_there(self, app):
        from app.crawl import backfill_missing_paper_metadata
        today = date.today()
        papers = {name: Paper(arxiv_id=f"web:biorxiv-fig{name}", title=name, source="biorxiv", pdf_url=f"{self.PDF}?{name}",
                              published_date=today - timedelta(days=days), enrich_complete=1)
                  for name, days in [("fresh", 2), ("ready", 7), ("old", 40)]}
        with app.app_context():
            db.session.add_all(papers.values())
            db.session.commit()
            try:
                with patch("app.crawl._ArxivLockCtx"), patch("app.crawl.time.sleep"), \
                        patch("app.crawl.arxiv.Client") as client, \
                        patch("app.crawl.get_citation_count", return_value=None), \
                        patch("app.crawl.fetch_biorxiv_figure", return_value=("https://www.biorxiv.org/f/F1.large.jpg",
                                                                              "Figure 1: x")) as fig:
                    client.return_value.results.return_value = []
                    backfill_missing_paper_metadata()
                    assert [c.args[0] for c in fig.call_args_list] == [f"{self.PDF}?ready"]  # 4 to 30 days only
                    assert papers["ready"].figure_url == "https://www.biorxiv.org/f/F1.large.jpg"
                    fig.return_value = (None, None)
                    papers["ready"].figure_url = None
                    db.session.commit()
                    backfill_missing_paper_metadata()  # tried today already: not again until tomorrow
                    assert fig.call_count == 1
            finally:
                for p in papers.values():
                    db.session.delete(p)
                db.session.commit()

    def test_biorxiv_downloads_are_one_at_a_time_7_s_apart(self, monkeypatch):
        from app import utils
        sleeps, held, clock = [], [], [1000.0]
        monkeypatch.setattr(utils.time, "sleep", lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)))
        monkeypatch.setattr(utils.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(utils, "_biorxiv_last", [float("-inf")])
        real = utils._http_client

        def handler(request):
            held.append(utils._biorxiv_lock.locked())
            return httpx.Response(200, content=b"not a pdf")
        monkeypatch.setattr(utils, "_http_client", lambda **kw: real(transport=httpx.MockTransport(handler), **{
            k: v for k, v in kw.items() if k != "public_only"}))
        utils.fetch_pdf_text(self.PDF, arxiv=False)
        clock[0] += 2  # the next one comes 2 s later
        utils.fetch_pdf_text(self.PDF, arxiv=False)
        assert held == [True, True] and utils.BIORXIV_GAP_S == 7
        assert sleeps == [0.0, 5.0]  # the first doesn't wait, the second waits out the rest of the 7 s
        utils.fetch_pdf_text("https://aclanthology.org/x.pdf", arxiv=False)
        assert held[-1] is False and len(sleeps) == 2  # other hosts are not slowed down

    def test_the_chat_downloads_it(self):
        from app.chat import _nothing_to_download
        assert not _nothing_to_download(Paper(arxiv_id="web:biorxiv-0123456789ab", pdf_url=self.PDF))

    def test_the_nightly_full_text_reread_includes_it(self, app, project):
        from app.crawl import refresh_from_full_text
        paper = Paper(arxiv_id="web:biorxiv-0123456789ab", title="Cell shape", source="biorxiv",
                      abstract="Resistant cells are wider.", pdf_url=self.PDF, enrich_complete=1)
        db.session.add(paper)
        db.session.flush()
        db.session.add(ProjectPaper(project_id=project.id, paper_id=paper.id, origin="crawl",
                                    screen_source="abstract", added_at=datetime.utcnow()))
        db.session.commit()
        with patch("app.crawl.get_paper_content", return_value=None) as content:
            refresh_from_full_text(project.id)
        content.assert_called_once_with("web:biorxiv-0123456789ab", self.PDF)


class TestPresentation:
    def test_card_text_and_tooltip(self):
        link = ProjectPaper(found_via=json.dumps([{"via": "keyword", "name": "tabular"},
                                                  {"via": "biorxiv", "name": "cell shape"},
                                                  {"via": "biorxiv", "name": "resistance"}]))
        assert link.found_via_text() == 'keyword "tabular" · bioRxiv keywords "cell shape", "resistance"'
        assert 'bioRxiv keyword search' in link.found_via_tip()

    def test_email_lists_biorxiv_only_when_it_found_something(self):
        from app.crawl import _email_source_rows
        zero = {"keyword": {"found": 3, "added": 1}, "biorxiv": {"found": 0, "added": 0}}
        some = {"keyword": {"found": 3, "added": 1}, "biorxiv": {"found": 2, "added": 0}}
        assert "biorxiv" not in [v for v, _ in _email_source_rows(zero)]
        assert "biorxiv" in [v for v, _ in _email_source_rows(some)]
        assert "biorxiv" not in [v for v, _ in _email_source_rows(None)]

    def test_the_card_shows_the_abstract_it_was_judged_on(self, app, client, project):
        with client.session_transaction() as sess:
            sess["logged_in"] = True
        paper = Paper(arxiv_id="web:biorxiv-abcdefabcdef", title="Cell shape", source="biorxiv",
                      abstract="Resistant cells grow wider under beta-lactams.",
                      pdf_url="https://www.biorxiv.org/content/10.64898/x", enrich_complete=1)
        db.session.add(paper)
        db.session.flush()
        db.session.add(ProjectPaper(project_id=project.id, paper_id=paper.id, origin="crawl"))
        db.session.commit()
        page = client.get(f"/projects/{project.slug}").get_data(as_text=True)
        assert "Resistant cells grow wider under beta-lactams." in page
        assert "https://www.biorxiv.org/content/10.64898/x" in page

    def test_featured_paper_email_links_to_biorxiv_not_arxiv(self):
        from types import SimpleNamespace
        from app.crawl import notify_crawl_complete

        top = SimpleNamespace(title="Cell shape", arxiv_id="web:biorxiv-0123456789ab", is_arxiv=False,
                              pdf_url="https://www.biorxiv.org/content/10.64898/x", year=2026, authors_list=[],
                              authors=None, institutions_list=[], institutions=None, citation_count=0,
                              summary=None, figure_url=None, figure_caption=None, main_contributions=None)
        with patch("app.crawl.send_notification", return_value=True) as send:
            notify_crawl_complete("P", "p", 5, 1, [top])
        body = send.call_args.kwargs["content_html"]
        assert "https://www.biorxiv.org/content/10.64898/x" in body
        assert "arxiv.org/abs/web:" not in body


class TestToggle:
    def test_the_setting_is_saved(self, app, client, project):
        with client.session_transaction() as sess:
            sess["logged_in"] = True
        assert client.post(f"/projects/{project.slug}/toggle-biorxiv", data={"enabled": "1"}).status_code == 204
        db.session.expire_all()
        assert Project.query.get(project.id).use_biorxiv is True
        client.post(f"/projects/{project.slug}/toggle-biorxiv", data={"enabled": "0"})
        db.session.expire_all()
        assert Project.query.get(project.id).use_biorxiv is False

    def test_find_papers_and_nightly_boxes_start_from_the_saved_setting(self, app, client, project):
        with client.session_transaction() as sess:
            sess["logged_in"] = True
        import re

        def find_papers_box_checked():
            page = client.get(f"/projects/{project.slug}").get_data(as_text=True)
            assert "Also search bioRxiv" in page
            tag = re.search(r'<input[^>]*id="use-biorxiv-checkbox"[^>]*>', page).group(0)
            return " checked" in tag

        assert find_papers_box_checked() is False
        client.post(f"/projects/{project.slug}/toggle-biorxiv", data={"enabled": "1"})
        assert find_papers_box_checked() is True
        client.post(f"/projects/{project.slug}/toggle-biorxiv", data={"enabled": "0"})

    def test_count_step_reports_biorxiv_papers_separately(self, app, client, project):
        import time
        cand = TestCrawl()._candidate()
        with client.session_transaction() as sess:
            sess["logged_in"] = True
        with patch("app.routes.projects.gather_candidates", return_value=[cand]) as gather:
            r = client.post(f"/projects/{project.slug}/arxiv-count", data={
                "keywords_json": json.dumps(["cell shape"]), "date_from": "2026-10-08", "date_to": "2026-10-12",
                "use_arxiv": "0", "use_scholar": "0", "use_citations": "0", "use_biorxiv": "1"})
            task = r.get_json()["task_id"]
            for _ in range(50):
                status = client.get(f"/projects/{project.slug}/arxiv-count-status/{task}").get_json()
                if status["status"] != "running":
                    break
                time.sleep(0.05)
        assert gather.call_args.kwargs["use_biorxiv"] is True
        assert (status["count"], status["biorxiv_count"], status["arxiv_count"]) == (1, 1, 0)


@pytest.mark.slow
def test_live_one_day_of_biorxiv_has_the_fields_the_crawl_relies_on():
    """Real API, one fixed past day (about 5 requests): the listing pages, and every field we read is there."""
    day = date(2026, 10, 10)
    items = biorxiv.fetch_window(day, day)
    assert len(items) > 20
    assert {it["version"] for it in items} == {"1"}
    candidates = [biorxiv._candidate(it, []) for it in items]
    assert all(c and c["arxiv_id"].startswith("web:biorxiv-") and c["title"] and c["abstract"] for c in candidates)
    assert len({c["arxiv_id"] for c in candidates}) == len(candidates)  # no paper twice
    words = max(len(c["abstract"].split()) for c in candidates)
    found = biorxiv.search_biorxiv([candidates[0]["title"].split()[-1]], day, day)
    assert any(f["doi"] == candidates[0]["doi"] for f in found) and words > 50
