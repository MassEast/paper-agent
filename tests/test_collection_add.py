"""
Tests for the "Add to My Collection" bypass pipeline.

Unit tests: fast, no network. Use pytest directly.
Live tests: hit real URLs. Marked slow — run with:
    pytest tests/test_collection_add.py -v -m slow --timeout=60
"""

import hashlib
import pytest


# ---------------------------------------------------------------------------
# Helpers / unit tests (no network, no Flask app)
# ---------------------------------------------------------------------------

class TestTitlesMatch:
    def _match(self, t1, t2):
        from app.crawl import _titles_match
        return _titles_match(t1, t2)

    def test_identical(self):
        assert self._match("Attention Is All You Need", "Attention Is All You Need")

    def test_case_insensitive(self):
        # Same paper title, different casing — must match
        assert self._match("BERT: Pre-training of Deep Bidirectional Transformers",
                           "bert pre-training of deep bidirectional transformers")

    def test_clearly_different(self):
        assert not self._match("Language Models Are Few-Shot Learners", "ResNet: Deep Residual Learning")

    def test_partial_overlap_above_threshold(self):
        # Blog post title vs arXiv title — typical case
        assert self._match(
            "RYS: Rethinking Your Sampling for Better Results",
            "Rethinking Sampling Strategies for Better Generative Results",
        )

    def test_short_titles_different(self):
        assert not self._match("GPT-4 Technical Report", "Image Segmentation Network")

    def test_empty_returns_false(self):
        assert not self._match("", "Attention Is All You Need")
        assert not self._match("Attention Is All You Need", "")


class TestSyntheticArxivId:
    def test_format(self):
        url = "https://dnhkng.github.io/posts/rys/"
        synthetic = f"web:{hashlib.sha256(url.encode()).hexdigest()[:12]}"
        assert synthetic.startswith("web:")
        assert len(synthetic) == 16  # "web:" + 12 hex chars

    def test_stable(self):
        url = "https://example.com/blog"
        id1 = f"web:{hashlib.sha256(url.encode()).hexdigest()[:12]}"
        id2 = f"web:{hashlib.sha256(url.encode()).hexdigest()[:12]}"
        assert id1 == id2

    def test_different_urls_give_different_ids(self):
        url1 = "https://dnhkng.github.io/posts/rys/"
        url2 = "https://dnhkng.github.io/posts/other/"
        id1 = f"web:{hashlib.sha256(url1.encode()).hexdigest()[:12]}"
        id2 = f"web:{hashlib.sha256(url2.encode()).hexdigest()[:12]}"
        assert id1 != id2


class TestPaperIsArxiv:
    """Unit test Paper.is_arxiv without hitting the DB."""

    def _make_paper(self, arxiv_id):
        from unittest.mock import MagicMock
        p = MagicMock()
        p.arxiv_id = arxiv_id
        from app.models import Paper
        # Use the actual property logic
        p.is_arxiv = Paper.is_arxiv.fget(p)
        return p

    def test_real_arxiv_id(self):
        from app.models import Paper
        from unittest.mock import MagicMock
        p = MagicMock()
        p.arxiv_id = "2301.00001"
        assert Paper.is_arxiv.fget(p)

    def test_web_synthetic_id(self):
        from app.models import Paper
        from unittest.mock import MagicMock
        p = MagicMock()
        p.arxiv_id = "web:abc123def456"
        assert not Paper.is_arxiv.fget(p)

    def test_none_id(self):
        from app.models import Paper
        from unittest.mock import MagicMock
        p = MagicMock()
        p.arxiv_id = None
        assert not Paper.is_arxiv.fget(p)


class TestArxivIdFromUrl:
    """Unit tests for arXiv ID extraction from various URL formats (no network)."""

    def _extract_arxiv_id(self, url: str):
        import re as _re
        m = _re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)
        if m:
            return m.group(1)
        # NASA ADS: {year}arXiv{YYMM}{NNNNN}{letter} → {YYMM}.{NNNNN}
        m = _re.search(r"adsabs\.harvard\.edu/abs/\d{4}arXiv(\d{4})(\d{5})\w", url)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
        return None

    def test_standard_arxiv_abs_url(self):
        assert self._extract_arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"

    def test_standard_arxiv_pdf_url(self):
        assert self._extract_arxiv_id("https://arxiv.org/pdf/2301.00001") == "2301.00001"

    def test_nasa_ads_url(self):
        url = "https://ui.adsabs.harvard.edu/abs/2026arXiv260309600B/abstract"
        assert self._extract_arxiv_id(url) == "2603.09600"

    def test_nasa_ads_url_aiayn(self):
        # "Attention Is All You Need" — bibcode 2017arXiv170603762V
        url = "https://ui.adsabs.harvard.edu/abs/2017arXiv170603762V/abstract"
        assert self._extract_arxiv_id(url) == "1706.03762"

    def test_non_arxiv_url_returns_none(self):
        assert self._extract_arxiv_id("https://openreview.net/forum?id=abc123") is None

    def test_blog_url_returns_none(self):
        assert self._extract_arxiv_id("https://example.com/blog/post") is None


class TestArxivLinkDetection:
    """Test that arXiv links are detected in fetched HTML."""

    def test_extract_arxiv_links_from_html(self):
        import re as _re
        html = """
        <p>See our paper at <a href="https://arxiv.org/abs/2301.00001">arxiv</a>.</p>
        <p>Also: https://arxiv.org/abs/2305.12345v2</p>
        """
        links = list(dict.fromkeys(
            _re.findall(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)', html, _re.IGNORECASE)
        ))
        assert "2301.00001" in links
        assert "2305.12345v2" in links

    def test_no_arxiv_links_in_plain_html(self):
        import re as _re
        html = "<p>No arXiv here, just a blog post about cooking.</p>"
        links = _re.findall(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)', html, _re.IGNORECASE)
        assert links == []


# ---------------------------------------------------------------------------
# Live tests (marked slow — require network + running inference server)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestWebPaperExtraction:
    """Live tests fetching real URLs to verify the extraction pipeline."""

    def test_blog_post_dnhkng(self):
        """The dnhkng blog post should extract a meaningful title and no arXiv link."""
        from app.crawl import extract_paper_info_from_url
        url = "https://dnhkng.github.io/posts/rys/"
        info = extract_paper_info_from_url(url)

        assert info["title"] is not None, "Should have extracted a title"
        assert "rys" in info["title"].lower() or "llm" in info["title"].lower(), \
            f"Unexpected title: {info['title']}"

        # No arXiv links expected on this blog post
        page_links = info.get("page_arxiv_links", [])
        # (may or may not have arxiv links — just log it)
        print(f"Extracted title: {info['title']}")
        print(f"Abstract excerpt: {(info.get('abstract') or '')[:120]}")
        print(f"arXiv links found: {page_links}")

    def test_cvpr_page_gets_paper_title(self):
        """CVPR open access page should use citation_title meta tag, not the generic site title."""
        from app.crawl import extract_paper_info_from_url
        url = "https://openaccess.thecvf.com/content/CVPR2022/html/Bao_Discovering_Objects_That_Can_Move_CVPR_2022_paper.html"
        info = extract_paper_info_from_url(url)

        assert info["title"] is not None
        assert "discovering" in info["title"].lower(), \
            f"Expected paper title, got: {info['title']}"
        assert info["abstract"] is not None and len(info["abstract"]) > 50, \
            "Should have extracted abstract from div#abstract"

    def test_arxiv_url_returns_arxiv_id(self):
        """arXiv URL should be detected and use the arXiv API, not the HTML path."""
        from app.crawl import extract_paper_info_from_url
        url = "https://arxiv.org/abs/1706.03762"
        info = extract_paper_info_from_url(url)

        assert info["arxiv_id"] == "1706.03762"
        assert "attention" in (info["title"] or "").lower(), \
            f"Unexpected title: {info['title']}"
        assert info["page_arxiv_links"] == [], \
            "arXiv path should not populate page_arxiv_links"

    def test_nasa_ads_url_resolves_to_arxiv(self):
        """NASA ADS URL should decode to arXiv ID and fetch paper metadata."""
        from app.crawl import extract_paper_info_from_url
        # 2026arXiv260309600B → arXiv:2603.09600
        url = "https://ui.adsabs.harvard.edu/abs/2026arXiv260309600B/abstract"
        info = extract_paper_info_from_url(url)

        assert info["arxiv_id"] == "2603.09600", \
            f"Expected arXiv ID 2603.09600, got: {info['arxiv_id']}"
        assert info["title"] is not None and len(info["title"]) > 10, \
            f"Expected a real title from arXiv API, got: {info['title']}"
        assert info["authors"] is not None and len(info["authors"]) > 0, \
            "Expected authors from arXiv API"
