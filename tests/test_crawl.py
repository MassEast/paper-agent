"""
Integration tests for run_crawl (app/crawl.py).

All external calls (arXiv, LLM, SemanticScholar, figure fetch) are mocked.
Tests exercise DB orchestration, cancellation, paper_limit, token tracking,
keyword persistence, and error paths.

Design notes:
- Always pass keywords_override to avoid _llm_json / arXiv calls for keyword gen.
- Use unique arxiv_ids per test to avoid cross-test DB collisions.
- The `project` fixture yields inside an app_context — do NOT nest another one.
"""

import json
import copy
from datetime import date
from unittest.mock import patch
from contextlib import ExitStack


from tests.conftest import (
    FAKE_PAPER,
    FAKE_KEYWORDS,
    FAKE_SUMMARY,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _papers(n: int, id_prefix: str) -> list:
    out = []
    for i in range(n):
        p = copy.deepcopy(FAKE_PAPER)
        p["arxiv_id"] = f"{id_prefix}{i:04d}"
        p["title"] = f"Paper {i}: Efficient Attention {id_prefix}"
        out.append(p)
    return out


class MockPatches:
    """Patches all external calls made inside run_crawl."""

    def __init__(self, papers, relevant=True):
        self._stack = ExitStack()
        self._papers = papers
        self._relevant = relevant

    def __enter__(self):
        s = self._stack
        s.enter_context(patch("app.crawl.search_arxiv_with_full_papers", return_value=self._papers))
        s.enter_context(patch("app.crawl.get_paper_content", return_value="Full paper text."))
        s.enter_context(patch("app.crawl.get_citation_count", return_value=3))
        s.enter_context(patch("app.crawl.get_paper_figure", return_value=(None, None)))
        s.enter_context(patch(
            "app.crawl.is_paper_relevant",
            return_value=(self._relevant, "reason", "test-model", 100, 60),
        ))
        s.enter_context(patch(
            "app.crawl.generate_summary_with_full_content",
            return_value=(FAKE_SUMMARY, "test-model", 200, 200),
        ))
        s.enter_context(patch(
            "app.crawl.extract_main_contributions",
            return_value=("- C1\n- C2", "Summary.", 50, 80),
        ))
        # Post-loop enrichment (institutions, page counts) and featured-paper selection all
        # make real network/LLM calls when unmocked. They only run when papers_added > 0, so
        # tests that don't add papers never exercised this gap — but any test that does add a
        # paper was silently hitting the real (possibly-down) LLM backend for ~90s+ of retries
        # per paper via the institutions LLM fallback. Mock all three for full hermeticity.
        s.enter_context(patch("app.crawl.enrich_institutions_from_scholar", return_value=None))
        s.enter_context(patch("app.crawl.enrich_page_counts", return_value=None))
        s.enter_context(patch(
            "app.crawl.select_featured_paper_llm",
            side_effect=lambda papers, research_interest: (papers[0] if papers else None, 50),
        ))
        return self

    def __exit__(self, *args):
        return self._stack.__exit__(*args)


def run(project_id, papers, relevant=True, paper_limit=50, keywords=None, date_suffix="01"):
    """Helper: run run_crawl with all externals mocked."""
    from app.crawl import run_crawl
    with MockPatches(papers, relevant=relevant):
        run_crawl(
            project_id,
            date(2025, int(date_suffix), 1),
            date(2025, int(date_suffix), 28),
            paper_limit=paper_limit,
            keywords_override=keywords or FAKE_KEYWORDS,
        )


# ── tests ─────────────────────────────────────────────────────────────────────

class TestRunCrawl:

    def test_adds_relevant_papers_to_db(self, app, project):
        from app.models import CrawlLog, ProjectPaper

        papers = _papers(2, "2501.A")
        run(project.id, papers, date_suffix="01")

        log = CrawlLog.query.filter_by(project_id=project.id).order_by(CrawlLog.id.desc()).first()
        assert log.status == "success"
        assert log.papers_added == 2
        assert log.papers_found == 2
        assert ProjectPaper.query.filter_by(project_id=project.id).count() == 2

    def test_skips_irrelevant_papers(self, app, project):
        from app.models import CrawlLog, ProjectPaper
        from app import db

        # Clear any papers from previous test
        for link in ProjectPaper.query.filter_by(project_id=project.id).all():
            db.session.delete(link)
        db.session.commit()

        papers = _papers(3, "2501.B")
        run(project.id, papers, relevant=False, date_suffix="02")

        log = CrawlLog.query.filter_by(project_id=project.id).order_by(CrawlLog.id.desc()).first()
        assert log.status == "success"
        assert log.papers_added == 0

    def test_paper_limit_caps_processing(self, app, project):
        from app.models import CrawlLog

        papers = _papers(6, "2501.C")
        run(project.id, papers, paper_limit=2, date_suffix="03")

        log = CrawlLog.query.filter_by(project_id=project.id).order_by(CrawlLog.id.desc()).first()
        assert log.papers_found == 6         # arXiv returned 6
        assert log.papers_added <= 2         # only processed up to limit

    def test_paper_limit_minus_one_processes_all(self, app, project):
        from app.models import CrawlLog, ProjectPaper, Paper
        from app import db

        papers = _papers(4, "2501.D")

        # Ensure clean slate for these arxiv IDs
        for p in papers:
            ex = Paper.query.filter_by(arxiv_id=p["arxiv_id"]).first()
            if ex:
                for link in ProjectPaper.query.filter_by(paper_id=ex.id).all():
                    db.session.delete(link)
                db.session.delete(ex)
        db.session.commit()

        run(project.id, papers, paper_limit=-1, date_suffix="04")

        log = CrawlLog.query.filter_by(project_id=project.id).order_by(CrawlLog.id.desc()).first()
        assert log.papers_added == 4

    def test_tokens_tracked_in_crawl_log(self, app, project):
        from app.models import CrawlLog

        papers = _papers(1, "2501.E")
        run(project.id, papers, date_suffix="05")

        log = CrawlLog.query.filter_by(project_id=project.id).order_by(CrawlLog.id.desc()).first()
        assert log.total_tokens is not None and log.total_tokens > 0

    def test_tokens_stored_per_paper(self, app, project):
        from app.models import Paper

        papers = _papers(1, "2501.F")
        run(project.id, papers, date_suffix="06")

        paper = Paper.query.filter_by(arxiv_id=papers[0]["arxiv_id"]).first()
        assert paper is not None
        assert paper.summary is not None
        assert paper.summary.tokens_used > 0

    def test_keywords_override_skips_keyword_extraction(self, app, project):
        from app.crawl import run_crawl

        papers = _papers(1, "2501.G")
        called = []

        def spy_extract(*args, **kwargs):
            called.append(True)
            return FAKE_KEYWORDS, {"total_tokens": 50}

        with MockPatches(papers), patch("app.crawl.extract_keywords_with_usage", side_effect=spy_extract):
            run_crawl(
                project.id,
                date(2025, 7, 1), date(2025, 7, 28),
                keywords_override=["custom keyword"],
            )

        assert len(called) == 0, "extract_keywords_with_usage must not be called when keywords_override given"

    def test_keywords_stored_in_crawl_log(self, app, project):
        from app.models import CrawlLog

        papers = _papers(1, "2501.H")
        keywords = ["efficient attention", "long context LLM"]
        run(project.id, papers, keywords=keywords, date_suffix="08")

        log = CrawlLog.query.filter_by(project_id=project.id).order_by(CrawlLog.id.desc()).first()
        assert log.keywords_list == keywords

    def test_cancel_stops_loop(self, app, project):
        """Simulates UI cancellation: flip crawl status to 'cancelled' on first relevance check."""
        from app.crawl import run_crawl
        from app.models import CrawlLog
        from app import db

        papers = _papers(5, "2501.I")
        cancelled = [False]

        def cancelling_relevance(title, abstract, paper_content, research_interest, collection_papers, **kw):
            # Flip the running crawl to cancelled on first call — must push app context since
            # this runs in a ThreadPoolExecutor worker thread.
            with app.app_context():
                log = CrawlLog.query.filter_by(project_id=project.id, status="running").first()
                if log and not cancelled[0]:
                    log.status = "cancelled"
                    db.session.commit()
                    cancelled[0] = True
            return (True, "relevant", "test-model", 100, 60)

        with MockPatches(papers):
            with patch("app.crawl.is_paper_relevant", side_effect=cancelling_relevance):
                run_crawl(
                    project.id,
                    date(2025, 9, 1), date(2025, 9, 28),
                    keywords_override=FAKE_KEYWORDS,
                )

        log = CrawlLog.query.filter_by(project_id=project.id).order_by(CrawlLog.id.desc()).first()
        assert log.status == "cancelled"
        assert log.papers_checked < 5

    def test_no_papers_found_completes_successfully(self, app, project):
        from app.crawl import run_crawl
        from app.models import CrawlLog

        with patch("app.crawl.search_arxiv_with_full_papers", return_value=[]):
            run_crawl(project.id, date(2025, 10, 1), date(2025, 10, 28), keywords_override=["test"])

        log = CrawlLog.query.filter_by(project_id=project.id).order_by(CrawlLog.id.desc()).first()
        assert log.status == "success"
        assert log.papers_added == 0

    def test_no_research_interest_no_refs_sets_error(self, app):
        from app.crawl import run_crawl
        from app.models import CrawlLog, Project
        from app import db

        with app.app_context():
            bare = Project(name="Bare Proj", slug="bare-proj-error-test")
            db.session.add(bare)
            db.session.commit()

            run_crawl(bare.id, date(2025, 11, 1), date(2025, 11, 28))

            log = CrawlLog.query.filter_by(project_id=bare.id).order_by(CrawlLog.id.desc()).first()
            assert log.status == "error"

            db.session.delete(bare)
            db.session.commit()

    def test_llm_unavailable_during_screening_marks_crawl_error(self, app, project):
        """Regression test: an LLM backend outage during per-paper screening must abort the
        crawl and mark it as failed, not get swallowed as a generic per-paper error and reported
        as a false 'success: 0 added'. See docs/PROGRESS.md 2026-09-22 for the incident this
        covers — nightly crawls silently reported success for weeks while every paper's
        screening call was failing with LLMUnavailableError."""
        from app.crawl import run_crawl
        from app.llm import LLMUnavailableError
        from app.models import CrawlLog

        papers = _papers(3, "2501.J")

        with MockPatches(papers):
            with patch("app.crawl.is_paper_relevant", side_effect=LLMUnavailableError("Connection refused")):
                run_crawl(
                    project.id,
                    date(2025, 6, 1), date(2025, 6, 28),
                    keywords_override=FAKE_KEYWORDS,
                )

        log = CrawlLog.query.filter_by(project_id=project.id).order_by(CrawlLog.id.desc()).first()
        assert log.status == "error"
        assert "LLM unavailable" in log.error_message
        assert log.papers_added == 0

    def test_existing_paper_linked_not_resummmarized(self, app, project):
        """Paper already in DB should be linked to project without calling generate_summary."""
        from app.crawl import run_crawl
        from app.models import Paper, ProjectPaper
        from app import db

        shared_id = "2501.SHARED1"
        papers = [dict(FAKE_PAPER, arxiv_id=shared_id, title="Shared Paper")]

        # Create the paper in DB first (simulating it exists from a prior crawl)
        existing = Paper.query.filter_by(arxiv_id=shared_id).first()
        if not existing:
            existing = Paper(
                arxiv_id=shared_id, title="Shared Paper",
                authors=json.dumps(["A. Author"]), abstract="Abstract.",
                source="arxiv", year=2025,
            )
            db.session.add(existing)
            db.session.commit()
        paper_id = existing.id

        summary_calls = []

        def counting_summary(*a, **kw):
            summary_calls.append(1)
            return FAKE_SUMMARY, "test-model", 100, 100

        with MockPatches(papers):
            with patch("app.crawl.generate_summary_with_full_content", side_effect=counting_summary):
                run_crawl(project.id, date(2025, 12, 1), date(2025, 12, 28), keywords_override=["test"])

        assert len(summary_calls) == 0, "Should not re-summarize a paper already in DB"
        assert ProjectPaper.query.filter_by(project_id=project.id, paper_id=paper_id).first() is not None
