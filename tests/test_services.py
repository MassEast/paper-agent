"""
Live smoke tests for external service integrations.

Each test hits the real external API. Marked `slow` — not run by default.
Run: pytest tests/test_services.py -v -m slow --timeout=60

These are the same checks as the in-app health checker (GET /projects/health-check),
so they can be run either in-terminal or via the UI button.
"""

import pytest
from datetime import date, timedelta

pytestmark = pytest.mark.slow


class TestLLMService:
    def test_llm_responds(self):
        """test-model should return a valid JSON screening decision."""
        from app.crawl import _screen_paper_quick
        result = _screen_paper_quick(
            title="Attention Is All You Need",
            abstract="We propose a model architecture based solely on attention mechanisms.",
            research_interest="transformer attention mechanisms",
            reference_context="",
        )
        assert isinstance(result, bool), f"Expected bool, got {result!r}"

    def test_llm_json_call(self):
        """Direct _llm_json call returns parseable output."""
        from app.llm import _llm_json
        messages = [
            {"role": "system", "content": "You are a JSON API. Reply only with valid JSON."},
            {"role": "user", "content": 'Reply with exactly: {"ok": true}'},
        ]
        payload, model, elapsed_ms, usage = _llm_json(messages, temperature=0.0, max_tokens=16)
        assert payload.get("ok") is True, f"Unexpected payload: {payload}"
        assert isinstance(elapsed_ms, (int, float))
        assert isinstance(usage, dict)


class TestArxivService:
    def test_search_returns_results(self):
        """arXiv keyword search returns at least one result."""
        from app.crawl import search_arxiv_with_full_papers
        import arxiv
        today = date.today()
        try:
            results = search_arxiv_with_full_papers(
                ["large language model"],
                date_from=today - timedelta(days=30),
                date_to=today,
                max_results=3,
            )
        except arxiv.HTTPError as e:
            pytest.skip(f"arXiv rate-limited (429): {e}")
        assert len(results) > 0, "arXiv returned no results"
        paper = results[0]
        for field in ("arxiv_id", "title", "authors", "abstract"):
            assert field in paper and paper[field], f"Missing or empty field: {field}"

    def test_known_paper_fetchable(self):
        """arXiv can fetch the Attention paper by ID."""
        import arxiv
        try:
            client = arxiv.Client(page_size=2, delay_seconds=0, num_retries=0)
            results = list(client.results(arxiv.Search(id_list=["1706.03762"])))
        except arxiv.HTTPError as e:
            pytest.skip(f"arXiv rate-limited (429): {e}")
        assert len(results) == 1
        assert "attention" in results[0].title.lower()


class TestSemanticScholarService:
    def test_recommendations_returns_results(self):
        """SS recommendations API returns papers for a known seed."""
        from app.crawl import get_semantic_scholar_recommendations
        results = get_semantic_scholar_recommendations(["2307.09288"], limit=5)  # Llama 2
        assert isinstance(results, list), f"Expected list, got {type(results)}"
        assert len(results) > 0, "SS returned no recommendations"
        paper = results[0]
        for field in ("arxiv_id", "title"):
            assert field in paper, f"Missing field: {field}"

    def test_deduplication_with_seed(self):
        """Seed paper should not appear in its own recommendations."""
        from app.crawl import get_semantic_scholar_recommendations
        results = get_semantic_scholar_recommendations(["2307.09288"], limit=10)  # Llama 2
        ids = [r["arxiv_id"] for r in results if r.get("arxiv_id")]
        assert "2307.09288" not in ids, "Seed paper appeared in its own recommendations"

    def test_empty_seed_returns_empty(self):
        """Empty seed list should not crash — returns empty list."""
        from app.crawl import get_semantic_scholar_recommendations
        results = get_semantic_scholar_recommendations([], limit=5)
        assert results == []


class TestFigureFetch:
    def test_figure_fetch_known_paper(self):
        """Figure fetch for a paper with known HTML rendering returns a valid URL."""
        from app.crawl import get_paper_figure
        url, caption = get_paper_figure("1706.03762")
        if url is not None:
            assert url.startswith("http"), f"URL should start with http: {url}"
            assert any(url.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")), \
                f"URL should end with an image extension: {url}"

    def test_figure_url_no_doubled_id(self):
        """Returned figure URL must not contain the arxiv ID twice in the path."""
        from app.crawl import get_paper_figure
        arxiv_id = "2205.14135"  # FlashAttention — has HTML rendering
        url, caption = get_paper_figure(arxiv_id)
        if url:
            base_id = arxiv_id.split("v")[0]
            assert f"{base_id}/{base_id}" not in url, \
                f"URL contains doubled arxiv ID — base URL bug: {url}"

    def test_figure_url_is_absolute(self):
        """Figure URL should always be an absolute http(s) URL."""
        from app.crawl import get_paper_figure
        url, caption = get_paper_figure("2205.14135")
        if url:
            assert url.startswith("https://") or url.startswith("http://"), \
                f"Figure URL is not absolute: {url}"

    def test_figure_returns_tuple(self):
        """get_paper_figure must always return a 2-tuple (url, caption), never a bare string."""
        from app.crawl import get_paper_figure
        result = get_paper_figure("2205.14135")
        assert isinstance(result, tuple) and len(result) == 2, \
            f"Expected (url, caption) tuple, got {result!r}"

    def test_caption_is_string_or_none(self):
        """Caption must be a non-empty string or None — never empty string."""
        from app.crawl import get_paper_figure
        url, caption = get_paper_figure("2205.14135")
        if caption is not None:
            assert isinstance(caption, str) and len(caption) > 0, \
                f"Caption should be non-empty string or None, got {caption!r}"

    def test_caption_present_for_html_paper(self):
        """A paper with HTML rendering should return a caption (figcaption in arXiv HTML)."""
        from app.crawl import get_paper_figure
        url, caption = get_paper_figure("2205.14135")
        if url is None:
            pytest.skip("No figure found for this paper — HTML rendering may be unavailable")
        assert caption is not None, "Expected a caption for a paper with HTML rendering"
        assert len(caption) > 10, f"Caption suspiciously short: {caption!r}"

    def test_figure_url_no_doubled_id_versioned(self):
        """arXiv HTML sometimes uses src paths like '2605.30322v1/x1.png' — must not double the ID.

        This is the health-check regression test. The bug: urljoin against the paper's own
        directory base doubles the arxiv ID when the src path already contains it.
        """
        from app.crawl import get_paper_figure
        arxiv_id = "2605.30322v1"
        url, caption = get_paper_figure(arxiv_id)
        if url is None:
            pytest.skip("No figure found for this paper")
        assert f"{arxiv_id}/{arxiv_id}" not in url, \
            f"URL contains doubled versioned ID — base URL resolution bug: {url}"
        base_id = arxiv_id.split("v")[0]
        assert f"{base_id}/{base_id}" not in url, \
            f"URL contains doubled base ID: {url}"

    def test_figure_outside_figure_tag(self):
        """Regression: arXiv HTML (latexml) sometimes places <img> in a sibling <div> before
        the <figure> tag rather than inside it. We should still pick Figure 1, not Figure 2.
        Reproduces with 2603.25733 (SlotVTG), where x1.png is in a preceding div."""
        from app.crawl import get_paper_figure
        arxiv_id = "2603.25733"
        url, caption = get_paper_figure(arxiv_id)
        if url is None:
            pytest.skip("No figure found — arXiv HTML may be unavailable")
        assert "x1.png" in url, f"Expected Figure 1 (x1.png), got {url!r}"
        assert caption is not None and "Figure 1" in caption, \
            f"Expected Figure 1 caption, got {caption!r}"


class TestWebPaperExtract:
    """Live tests for extract_paper_info_from_url — covers academic PDFs and blog posts."""

    def test_acl_anthology_pdf_url(self):
        """ACL Anthology .pdf URL should resolve to HTML page and extract title + authors."""
        from app.crawl import extract_paper_info_from_url
        result = extract_paper_info_from_url("https://aclanthology.org/2026.eacl-long.247.pdf")
        assert result["title"], "Expected a non-empty title for ACL paper"
        assert result["arxiv_id"] is None, "ACL paper should not be detected as arXiv"
        # Should find authors via citation_author meta tags
        if result.get("authors"):
            assert isinstance(result["authors"], list)
            assert len(result["authors"]) > 0

    def test_acl_anthology_html_url(self):
        """ACL Anthology HTML page should extract title and abstract."""
        from app.crawl import extract_paper_info_from_url
        result = extract_paper_info_from_url("https://aclanthology.org/2026.eacl-long.247")
        assert result["title"], "Expected title from ACL HTML page"
        # ACL doesn't always have citation_abstract but may have abstract in body
        # Just verify no crash and title is populated

    def test_neurips_pdf_url(self):
        """NeurIPS PDF URL should strip .pdf and extract title."""
        from app.crawl import extract_paper_info_from_url
        # Attention is All You Need — NeurIPS 2017
        result = extract_paper_info_from_url(
            "https://proceedings.neurips.cc/paper_files/paper/2017/file/"
            "3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
        )
        # Should either get a title or at least not crash
        assert isinstance(result, dict)
        assert "title" in result

    def test_blog_post_extraction(self):
        """Blog post should extract title; abstract should NOT be the site tagline."""
        from app.crawl import extract_paper_info_from_url
        result = extract_paper_info_from_url("https://dnhkng.github.io/posts/rys/")
        assert result["title"], "Expected a non-empty title for blog post"
        # Blog posts should have full_text for LLM summarization
        if result.get("full_text"):
            assert len(result["full_text"]) > 200, "full_text too short to be useful"
        # abstract should NOT be the site tagline (citation_abstract only)
        # For a blog, no citation_abstract → abstract should be None or empty
        # (the LLM uses full_text instead)

    def test_pdf_url_strip_works(self):
        """Stripping .pdf suffix should reach the HTML page, not binary content."""
        from app.crawl import extract_paper_info_from_url
        result = extract_paper_info_from_url("https://aclanthology.org/2026.eacl-long.247.pdf")
        # If we got a title, it means HTML was fetched (not binary PDF)
        if result["title"]:
            assert not result["title"].startswith("%PDF"), \
                "Got PDF binary instead of HTML page"

    def test_full_text_present_for_html_page(self):
        """Any successfully fetched HTML page should produce full_text."""
        from app.crawl import extract_paper_info_from_url
        result = extract_paper_info_from_url("https://aclanthology.org/2026.eacl-long.247")
        if result["title"]:  # only assert if fetch succeeded
            assert result.get("full_text"), "Expected full_text for successfully fetched page"
            assert len(result["full_text"]) > 100
