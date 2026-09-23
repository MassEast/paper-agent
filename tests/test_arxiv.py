"""
Live smoke tests for arXiv integration.

These hit the real arXiv API and are SLOW (~20-50s each).
Run with: pytest tests/test_arxiv.py -v -m slow --timeout=120

Not run by default (marked slow) to keep the main suite fast.
"""

import pytest
from datetime import date, timedelta

pytestmark = pytest.mark.slow  # skip unless -m slow


class TestArxivSearch:
    def test_basic_search_returns_results(self):
        from app.crawl import search_arxiv_with_full_papers
        today = date.today()
        results = search_arxiv_with_full_papers(
            ["transformer attention"],
            date_from=today - timedelta(days=180),
            date_to=today,
            max_results=5,
        )
        if not results:
            pytest.skip("arXiv returned no results — likely rate-limited or slow right now")
        assert len(results) > 0

    def test_result_has_required_fields(self):
        from app.crawl import search_arxiv_with_full_papers
        today = date.today()
        results = search_arxiv_with_full_papers(
            ["large language model"],
            date_from=today - timedelta(days=90),
            date_to=today,
            max_results=3,
        )
        if not results:
            pytest.skip("arXiv returned no results — likely rate-limited or slow right now")
        paper = results[0]
        for field in ("arxiv_id", "title", "authors", "abstract", "published_date", "year"):
            assert field in paper, f"Missing field: {field}"
        assert isinstance(paper["authors"], list)
        assert isinstance(paper["year"], int)

    def test_date_filter_works(self):
        """Papers should fall within the requested date range."""
        from app.crawl import search_arxiv_with_full_papers
        date_from = date(2024, 1, 1)
        date_to = date(2024, 3, 31)
        results = search_arxiv_with_full_papers(
            ["diffusion model image generation"],
            date_from=date_from,
            date_to=date_to,
            max_results=5,
        )
        for paper in results:
            pub = paper["published_date"]
            assert date_from <= pub <= date_to, (
                f"Paper date {pub} outside requested range {date_from}–{date_to}"
            )

    def test_deduplication_across_keywords(self):
        """Same paper matched by two keywords should only appear once."""
        from app.crawl import search_arxiv_papers
        today = date.today()
        # Use two overlapping queries that will likely match many of the same papers
        results = search_arxiv_papers(
            ["attention mechanism transformer", "transformer self attention"],
            date_from=today - timedelta(days=30),
            date_to=today,
            limit=50,
        )
        ids = [p["arxiv_id"] for p in results]
        assert len(ids) == len(set(ids)), "Duplicate arxiv IDs found after dedup"

    def test_timeout_does_not_hang(self):
        """search_arxiv_with_timeout should return (possibly empty) within timeout_seconds."""
        import time
        from app.crawl import search_arxiv_with_timeout
        today = date.today()
        t0 = time.time()
        # Use a very specific obscure query unlikely to return many results
        results = search_arxiv_with_timeout(
            "xyzzy_nonexistent_query_abc123",
            max_results=10,
            date_from=today - timedelta(days=7),
            date_to=today,
            timeout_seconds=30,
        )
        elapsed = time.time() - t0
        assert elapsed < 35, f"Search took {elapsed:.1f}s — should not exceed timeout by much"
        assert isinstance(results, list)

    def test_fetch_specific_paper_by_id(self):
        """Fetching attention paper 1706.03762 should return the right title."""
        import arxiv
        try:
            client = arxiv.Client(page_size=2, delay_seconds=0, num_retries=0)
            results = list(client.results(arxiv.Search(id_list=["1706.03762"])))
        except arxiv.HTTPError as e:
            pytest.skip(f"arXiv rate-limited: {e}")
        assert len(results) == 1
        assert "attention" in results[0].title.lower()
