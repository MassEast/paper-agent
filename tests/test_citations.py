"""
Live smoke tests for Semantic Scholar citation count fetching.

Marked `slow` — not run by default (hits real SS API).
Run: SCHOLAR_API_KEY=... pytest tests/test_citations.py -v -m slow
"""

import os
import httpx
import pytest

pytestmark = pytest.mark.slow


class TestCitationCount:
    def test_known_paper_has_citations(self):
        """Attention Is All You Need (1706.03762) should have many citations."""
        from app.crawl import get_citation_count
        count = get_citation_count("1706.03762")
        assert isinstance(count, int), f"Expected int, got {count!r}"
        assert count > 1000, f"Expected >1000 citations for AIAYN, got {count}"

    def test_returns_none_for_unknown_paper(self):
        """Non-existent arxiv ID → Scholar 404 → None (caller should not update stored value)."""
        from app.crawl import get_citation_count
        count = get_citation_count("0000.00000")
        assert count is None, f"Expected None for unknown paper, got {count!r}"

    def test_recent_paper_returns_int(self):
        """Recent paper may have 0 citations but must return an int, not None."""
        from app.crawl import get_citation_count
        count = get_citation_count("2604.16042")
        assert isinstance(count, int), f"Expected int, got {count!r}"
        assert count >= 0

    def test_scholar_id_lookup_matches_arxiv_lookup(self):
        """get_citation_count_by_scholar_id should return same count as arXiv lookup."""
        from app.crawl import get_citation_count_by_scholar_id

        # Derive Scholar paper ID for AIAYN from the arXiv endpoint
        api_key = os.environ.get("SCHOLAR_API_KEY", "")
        r = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.03762?fields=paperId,citationCount",
            headers={"x-api-key": api_key} if api_key else {},
            timeout=15,
        )
        assert r.status_code == 200, f"Scholar lookup failed: {r.status_code}"
        data = r.json()
        scholar_id = data["paperId"]
        expected = data["citationCount"]

        count = get_citation_count_by_scholar_id(scholar_id)
        assert isinstance(count, int), f"Expected int, got {count!r}"
        assert count == expected, f"Scholar-ID count {count} != arXiv count {expected}"

    def test_scholar_id_returns_none_for_bogus_id(self):
        """Bogus Scholar paper ID should return None without crashing."""
        from app.crawl import get_citation_count_by_scholar_id
        count = get_citation_count_by_scholar_id("0000000000000000000000000000000000000000")
        assert count is None, f"Expected None for bogus Scholar ID, got {count!r}"
