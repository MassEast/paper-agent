"""
Live smoke tests for get_paper_figure().

Marked `slow` — not run by default (hits real arXiv).
Run: pytest tests/test_figures.py -v -m slow
"""

import pytest

pytestmark = pytest.mark.slow


class TestGetPaperFigure:
    def test_paper_with_html_version_returns_figure(self):
        """A paper with an arXiv HTML version should return a real figure URL."""
        from app.crawl import get_paper_figure
        # Attention Is All You Need — has HTML and a well-known Figure 1
        url, caption = get_paper_figure("1706.03762")
        assert url is not None, "Expected a figure URL"
        assert url.startswith("http"), f"Expected absolute URL, got {url!r}"
        assert "static/base" not in url, f"Got a UI-chrome image instead of a paper figure: {url!r}"
        assert "funder" not in url.lower(), f"Got a funder logo instead of a paper figure: {url!r}"

    def test_no_html_version_does_not_return_funder_logo(self):
        """2605.09403 has no arXiv HTML version; fallback abs page only has UI/funder images.
        We should return (None, None) rather than a funder/chrome image."""
        from app.crawl import get_paper_figure
        url, caption = get_paper_figure("2605.09403")
        if url is not None:
            assert "funder" not in url.lower(), (
                f"Returned a funder logo as figure: {url!r}"
            )
            assert "static/base" not in url, (
                f"Returned a UI-chrome image as figure: {url!r}"
            )
            assert "static/browse" not in url, (
                f"Returned a browser-UI image as figure: {url!r}"
            )
            assert url.startswith("http"), f"Expected absolute URL, got {url!r}"
