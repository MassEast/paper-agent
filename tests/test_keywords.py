"""
Tests for keyword extraction (app/crawl.py::extract_keywords_with_usage).

All LLM calls are mocked — these run without network access.
"""

import pytest
from unittest.mock import patch

from tests.conftest import FAKE_KEYWORDS, make_llm_json_response


def _mock_llm_json(payload, **kw):
    """Monkeypatch target: _llm_json in app.crawl namespace."""
    def _inner(messages, **kwargs):
        return make_llm_json_response(payload)
    return _inner


class TestExtractKeywordsWithUsage:
    def test_returns_list_of_strings(self, app):
        with app.app_context():
            from app.crawl import extract_keywords_with_usage
            with patch("app.crawl._llm_json", return_value=make_llm_json_response(FAKE_KEYWORDS)):
                keywords, usage = extract_keywords_with_usage("transformer attention", [])
            assert isinstance(keywords, list)
            assert all(isinstance(k, str) for k in keywords)
            assert len(keywords) <= 6

    def test_llm_returns_dict_with_keywords_key(self, app):
        with app.app_context():
            from app.crawl import extract_keywords_with_usage
            payload = {"keywords": ["sparse attention", "efficient LLM"]}
            with patch("app.crawl._llm_json", return_value=make_llm_json_response(payload)):
                keywords, usage = extract_keywords_with_usage("efficient LLMs", [])
            assert "sparse attention" in keywords
            assert "efficient LLM" in keywords

    def test_caps_at_six_keywords(self, app):
        with app.app_context():
            from app.crawl import extract_keywords_with_usage
            long_list = [f"keyword {i}" for i in range(20)]
            with patch("app.crawl._llm_json", return_value=make_llm_json_response(long_list)):
                keywords, _ = extract_keywords_with_usage("something", [])
            assert len(keywords) <= 6

    def test_llm_unavailable_propagates(self, app):
        with app.app_context():
            from app.crawl import extract_keywords_with_usage
            from app.llm import LLMUnavailableError
            with patch("app.crawl._llm_json", side_effect=LLMUnavailableError("down")):
                with pytest.raises(LLMUnavailableError):
                    extract_keywords_with_usage("hallucination in LLMs", [])

    def test_llm_returns_empty_dict_yields_empty_keywords(self, app):
        with app.app_context():
            from app.crawl import extract_keywords_with_usage
            # Simulates server returning nothing → _llm_json falls back to {}
            with patch("app.crawl._llm_json", return_value=make_llm_json_response({})):
                keywords, usage = extract_keywords_with_usage("hallucination in LLMs", [])
            assert keywords == []

    def test_llm_exception_yields_empty_keywords(self, app):
        with app.app_context():
            from app.crawl import extract_keywords_with_usage
            with patch("app.crawl._llm_json", side_effect=RuntimeError("timeout")):
                keywords, usage = extract_keywords_with_usage("hallucination in LLMs", [])
            assert keywords == []
            assert usage["total_tokens"] == 0

    def test_with_collection_papers_calls_llm(self, app):
        with app.app_context():
            from app.crawl import extract_keywords_with_usage
            with patch("app.crawl._llm_json", return_value=make_llm_json_response(FAKE_KEYWORDS)):
                keywords, usage = extract_keywords_with_usage("efficient attention", [])
            assert len(keywords) > 0
            assert usage.get("total_tokens", 0) > 0

    def test_usage_dict_has_total_tokens(self, app):
        with app.app_context():
            from app.crawl import extract_keywords_with_usage
            with patch("app.crawl._llm_json", return_value=make_llm_json_response(FAKE_KEYWORDS, tokens=123)):
                _, usage = extract_keywords_with_usage("transformers", [])
            assert "total_tokens" in usage
            assert usage["total_tokens"] == 123
