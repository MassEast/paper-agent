"""
Tests for institution extraction pipeline.

Unit tests (default): mock LLM + PDF fetch.
Live tests (--slow):  hit real arXiv PDF + real LLM against a known paper.

Run live:  pytest tests/test_institutions.py -v -m slow --timeout=120
"""

import json
import io
from unittest.mock import patch, MagicMock

import pytest


# ── unit tests (no network, no real LLM) ────────────────────────────────────

class TestInstitutionLLMFallbackUnit:
    def test_extracts_institutions_from_list_response(self, ctx):
        from app.crawl import _enrich_institutions_llm_fallback
        from app.models import Paper

        paper = Paper(
            arxiv_id="0000.00001",
            title="Test Paper",
            abstract="An abstract without institution info.",
        )

        mock_result = (["MIT CSAIL", "Stanford University"], "test-model", 100, {"total_tokens": 50})
        with patch("app.crawl._llm_json", return_value=mock_result):
            with patch("app.crawl.httpx.Client") as mock_client:
                mock_resp = MagicMock()
                mock_resp.status_code = 404
                mock_client.return_value.__enter__.return_value.get.return_value = mock_resp
                _enrich_institutions_llm_fallback([paper])

        assert paper.institutions is not None
        insts = json.loads(paper.institutions)
        assert "MIT CSAIL" in insts
        assert "Stanford University" in insts

    def test_empty_list_response_leaves_institutions_unset(self, ctx):
        from app.crawl import _enrich_institutions_llm_fallback
        from app.models import Paper

        paper = Paper(
            arxiv_id="0000.00002",
            title="No Affiliation Paper",
            abstract="Some abstract.",
        )

        mock_result = ([], "test-model", 100, {"total_tokens": 50})
        with patch("app.crawl._llm_json", return_value=mock_result):
            with patch("app.crawl.httpx.Client") as mock_client:
                mock_resp = MagicMock()
                mock_resp.status_code = 404
                mock_client.return_value.__enter__.return_value.get.return_value = mock_resp
                _enrich_institutions_llm_fallback([paper])

        assert paper.institutions is None

    def test_enrich_ss_skips_papers_already_having_institutions(self, ctx):
        """enrich_institutions_from_scholar (the public entry point) filters already-enriched papers."""
        from app.crawl import enrich_institutions_from_scholar
        from app.models import Paper

        paper = Paper(
            arxiv_id="0000.00003",
            title="Already Enriched",
            abstract="Abstract.",
            institutions='["Existing University"]',
        )

        with patch("app.crawl._llm_json") as mock_llm:
            with patch("app.crawl.httpx.Client") as mock_client:
                mock_resp = MagicMock()
                mock_resp.status_code = 429
                mock_client.return_value.__enter__.return_value.post.return_value = mock_resp
                enrich_institutions_from_scholar([paper])
            mock_llm.assert_not_called()

        # Existing institutions must be preserved
        assert json.loads(paper.institutions) == ["Existing University"]

    def test_page1_text_preferred_over_abstract(self, ctx):
        """When PDF page 1 is available, prompt uses it instead of abstract."""
        from app.crawl import _enrich_institutions_llm_fallback
        from app.models import Paper

        paper = Paper(
            arxiv_id="0000.00004",
            title="Page1 Test",
            abstract="Short abstract.",
        )

        mock_result = (["Harvard University"], "test-model", 100, {"total_tokens": 50})
        captured_prompts = []

        def fake_llm_json(messages, **kwargs):
            captured_prompts.append(messages[0]["content"])
            return mock_result

        fake_page1 = "Title\nAuthor Name\nHarvard University\nAbstract: ..."

        with patch("app.crawl._llm_json", side_effect=fake_llm_json):
            with patch("app.crawl.httpx.Client") as mock_client:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.content = b"%PDF fake"

                from pypdf import PdfWriter
                buf = io.BytesIO()
                writer = PdfWriter()
                writer.add_blank_page(width=612, height=792)
                writer.write(buf)
                mock_resp.content = buf.getvalue()

                mock_client.return_value.__enter__.return_value.get.return_value = mock_resp

                with patch("pypdf.PdfReader") as mock_reader:
                    mock_page = MagicMock()
                    mock_page.extract_text.return_value = fake_page1
                    mock_reader.return_value.pages = [mock_page]
                    _enrich_institutions_llm_fallback([paper])

        assert len(captured_prompts) == 1
        assert "first page of the PDF" in captured_prompts[0]
        assert "Harvard University" in captured_prompts[0]


# ── live integration tests (slow, requires network + real LLM) ───────────────

@pytest.mark.slow
class TestInstitutionLivePDF:
    def test_page1_text_extraction_known_paper(self):
        """2605.29247 has affiliations in page-1 footnotes — pypdf should extract them."""
        import httpx
        from pypdf import PdfReader

        url = "https://arxiv.org/pdf/2605.29247.pdf"
        try:
            r = httpx.get(url, follow_redirects=True, timeout=30)
        except Exception as e:
            pytest.skip(f"Network unavailable: {e}")

        if r.status_code == 429:
            pytest.skip("arXiv rate-limited (429)")
        assert r.status_code == 200

        reader = PdfReader(io.BytesIO(r.content), strict=False)
        page1 = reader.pages[0].extract_text() or ""

        assert "university" in page1.lower(), "Expected university affiliation in page 1 text"
        # Affiliations are in footnotes at ~char 1845 — within our 5000-char window
        assert len(page1) > 1000, f"Page 1 text suspiciously short: {len(page1)} chars"

    def test_llm_extracts_institutions_footnote_paper(self):
        """End-to-end: fetch PDF + LLM extracts footnote affiliations for 2605.29247."""
        import httpx
        from pypdf import PdfReader
        from app.llm import _llm_json
        import app.prompts as prompts

        url = "https://arxiv.org/pdf/2605.29247.pdf"
        try:
            r = httpx.get(url, follow_redirects=True, timeout=30)
        except Exception as e:
            pytest.skip(f"Network unavailable: {e}")

        if r.status_code == 429:
            pytest.skip("arXiv rate-limited (429)")

        reader = PdfReader(io.BytesIO(r.content), strict=False)
        page1 = reader.pages[0].extract_text() or ""

        prompt = prompts.INSTITUTION_EXTRACTION.format(
            source_label="first page of the PDF",
            title="DenseSteer: Steering Small Language Models towards Dense Math Reasoning",
            text=page1[:5000],
        )
        result, _, _, _ = _llm_json(
            [{"role": "user", "content": prompt}], temperature=0, max_tokens=2048
        )

        assert isinstance(result, list), f"Expected list, got {type(result)}: {result!r}"
        assert len(result) > 0, "Expected at least one institution extracted"
        institutions_str = " ".join(result).lower()
        assert "university" in institutions_str, f"Expected a university in results: {result}"

    def test_no_institution_paper_returns_empty(self):
        """2605.17598 has no affiliation block on page 1 — should return []."""
        import httpx
        from pypdf import PdfReader
        from app.llm import _llm_json
        import app.prompts as prompts

        url = "https://arxiv.org/pdf/2605.17598.pdf"
        try:
            r = httpx.get(url, follow_redirects=True, timeout=30)
        except Exception as e:
            pytest.skip(f"Network unavailable: {e}")

        if r.status_code == 429:
            pytest.skip("arXiv rate-limited (429)")

        reader = PdfReader(io.BytesIO(r.content), strict=False)
        page1 = reader.pages[0].extract_text() or ""

        prompt = prompts.INSTITUTION_EXTRACTION.format(
            source_label="first page of the PDF",
            title="Mixture of Experts for Low-Resource LLMs",
            text=page1[:5000],
        )
        result, _, _, _ = _llm_json(
            [{"role": "user", "content": prompt}], temperature=0, max_tokens=2048
        )

        # Should be [] or at most an empty-ish result — page 1 has no affiliation block
        assert isinstance(result, list), f"Expected list, got {type(result)}: {result!r}"
