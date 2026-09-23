"""
Tests for paper relevance screening (app/crawl.py::is_paper_relevant).

Mocks _llm_json — no network calls.
"""

from unittest.mock import patch

from tests.conftest import make_llm_json_response, FAKE_RELEVANCE_YES, FAKE_RELEVANCE_NO

TITLE = "SuperSparse Attention for Long Sequences"
ABSTRACT = "We propose SuperSparse attention reducing complexity to O(n log n)."
CONTENT = "Introduction: We contribute a novel sparse attention pattern..."
RESEARCH = "efficient transformer attention for long-context LLMs"


class TestIsPaperRelevant:
    def test_relevant_paper_returns_true(self, app):
        with app.app_context():
            from app.crawl import is_paper_relevant
            with patch("app.crawl._llm_json", return_value=make_llm_json_response(FAKE_RELEVANCE_YES)):
                relevant, reason, model, elapsed, tokens = is_paper_relevant(
                    TITLE, ABSTRACT, CONTENT, RESEARCH, []
                )
            assert relevant is True
            assert isinstance(reason, str) and len(reason) > 0

    def test_irrelevant_paper_returns_false(self, app):
        with app.app_context():
            from app.crawl import is_paper_relevant
            with patch("app.crawl._llm_json", return_value=make_llm_json_response(FAKE_RELEVANCE_NO)):
                relevant, reason, model, elapsed, tokens = is_paper_relevant(
                    "Quantum Chemistry Methods",
                    "We study molecular bond energies.",
                    "",
                    RESEARCH,
                    [],
                )
            assert relevant is False

    def test_return_signature_is_five_tuple(self, app):
        with app.app_context():
            from app.crawl import is_paper_relevant
            with patch("app.crawl._llm_json", return_value=make_llm_json_response(FAKE_RELEVANCE_YES, tokens=77)):
                result = is_paper_relevant(TITLE, ABSTRACT, CONTENT, RESEARCH, [])
            assert len(result) == 5
            relevant, reason, model, elapsed, tokens = result
            assert isinstance(relevant, bool)
            assert isinstance(tokens, int)
            assert tokens > 0

    def test_invalid_llm_response_skips_paper(self, app):
        with app.app_context():
            from app.crawl import is_paper_relevant
            # LLM returns garbage (empty dict, no "relevant" key) — paper is skipped (not relevant)
            with patch("app.crawl._llm_json", return_value=make_llm_json_response({})):
                relevant, reason, model, elapsed, tokens = is_paper_relevant(TITLE, ABSTRACT, CONTENT, RESEARCH, [])
            assert relevant is False

    def test_quick_check_only_when_not_relevant(self, app):
        """If quick check says not relevant, deep check should not be called."""
        with app.app_context():
            from app.crawl import is_paper_relevant
            call_count = 0

            def counting_llm_json(messages, **kwargs):
                nonlocal call_count
                call_count += 1
                return make_llm_json_response(FAKE_RELEVANCE_NO)

            with patch("app.crawl._llm_json", side_effect=counting_llm_json):
                relevant, *_ = is_paper_relevant(TITLE, ABSTRACT, CONTENT, RESEARCH, [])

            assert not relevant
            assert call_count == 1, "Deep check should be skipped when quick says irrelevant"

    def test_deep_check_triggered_when_relevant_and_has_content(self, app):
        """If quick check says relevant and content is available, deep check fires."""
        with app.app_context():
            from app.crawl import is_paper_relevant
            call_count = 0

            def counting_llm_json(messages, **kwargs):
                nonlocal call_count
                call_count += 1
                return make_llm_json_response(FAKE_RELEVANCE_YES)

            with patch("app.crawl._llm_json", side_effect=counting_llm_json):
                is_paper_relevant(TITLE, ABSTRACT, CONTENT, RESEARCH, [])

            assert call_count == 2, "Deep check should fire when quick says relevant and content exists"

    def test_skip_deep_check_flag(self, app):
        """skip_deep_check=True should suppress phase 2 even with content."""
        with app.app_context():
            from app.crawl import is_paper_relevant
            call_count = 0

            def counting_llm_json(messages, **kwargs):
                nonlocal call_count
                call_count += 1
                return make_llm_json_response(FAKE_RELEVANCE_YES)

            with patch("app.crawl._llm_json", side_effect=counting_llm_json):
                is_paper_relevant(TITLE, ABSTRACT, CONTENT, RESEARCH, [], skip_deep_check=True)

            assert call_count == 1

    def test_no_content_skips_deep_check(self, app):
        """Empty paper_content should skip phase 2."""
        with app.app_context():
            from app.crawl import is_paper_relevant
            call_count = 0

            def counting_llm_json(messages, **kwargs):
                nonlocal call_count
                call_count += 1
                return make_llm_json_response(FAKE_RELEVANCE_YES)

            with patch("app.crawl._llm_json", side_effect=counting_llm_json):
                is_paper_relevant(TITLE, ABSTRACT, "", RESEARCH, [])

            assert call_count == 1
