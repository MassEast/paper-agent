"""
Tests for the two relevance-screening phases (app/crawl.py::_screen_paper_quick / _screen_paper_deep).
How they are chained (deep only after quick passes) is covered in test_crawl.py and
test_paper_actions.py (re-screen).

Mocks _llm_json — no network calls.
"""

from unittest.mock import patch

from tests.conftest import make_llm_json_response, FAKE_RELEVANCE_YES, FAKE_RELEVANCE_NO

TITLE = "SuperSparse Attention for Long Sequences"
ABSTRACT = "We propose SuperSparse attention reducing complexity to O(n log n)."
CONTENT = "Introduction: We contribute a novel sparse attention pattern..."
RESEARCH = "efficient transformer attention for long-context LLMs"


class TestQuickCheck:
    def test_relevant(self, app):
        from app.crawl import _screen_paper_quick
        with patch("app.crawl._llm_json", return_value=make_llm_json_response(FAKE_RELEVANCE_YES, tokens=77)):
            relevant, reason, model, elapsed, tokens = _screen_paper_quick(TITLE, ABSTRACT, RESEARCH, [])
        assert (relevant, reason, tokens) == (True, "relevant", 77)

    def test_not_relevant(self, app):
        from app.crawl import _screen_paper_quick
        with patch("app.crawl._llm_json", return_value=make_llm_json_response(FAKE_RELEVANCE_NO)):
            relevant, reason, *_ = _screen_paper_quick("Quantum Chemistry", "Bond energies.", RESEARCH, [])
        assert (relevant, reason) == (False, "not-relevant")

    def test_unparseable_answer_skips_paper(self, app):
        from app.crawl import _screen_paper_quick
        with patch("app.crawl._llm_json", return_value=make_llm_json_response({})):
            relevant, reason, *_ = _screen_paper_quick(TITLE, ABSTRACT, RESEARCH, [])
        assert (relevant, reason) == (False, "parse-failed")

    def test_prompt_has_no_full_text(self, app):
        from app.crawl import _screen_paper_quick
        with patch("app.crawl._llm_json", return_value=make_llm_json_response(FAKE_RELEVANCE_YES)) as llm:
            _screen_paper_quick(TITLE, ABSTRACT, RESEARCH, [{"title": "Ref paper", "abstract": "Ref abstract"}])
        prompt = llm.call_args[0][0][0]["content"]
        assert ABSTRACT in prompt and "Ref paper" in prompt and CONTENT not in prompt


class TestDeepCheck:
    def test_relevant_with_why(self, app):
        from app.crawl import _screen_paper_deep
        with patch("app.crawl._llm_json", return_value=make_llm_json_response(FAKE_RELEVANCE_YES)) as llm:
            relevant, reason, _, _, tokens, why = _screen_paper_deep(TITLE, ABSTRACT, CONTENT, RESEARCH, [])
        assert (relevant, reason, why) == (True, "relevant", FAKE_RELEVANCE_YES["reason"])
        assert CONTENT in llm.call_args[0][0][0]["content"]

    def test_not_relevant_has_no_why(self, app):
        from app.crawl import _screen_paper_deep
        with patch("app.crawl._llm_json", return_value=make_llm_json_response(FAKE_RELEVANCE_NO)):
            relevant, reason, _, _, _, why = _screen_paper_deep(TITLE, ABSTRACT, CONTENT, RESEARCH, [])
        assert (relevant, reason, why) == (False, "not-relevant", "")

    def test_unparseable_answer_is_not_relevant(self, app):
        from app.crawl import _screen_paper_deep
        with patch("app.crawl._llm_json", return_value=make_llm_json_response({})):
            relevant, reason, *_ = _screen_paper_deep(TITLE, ABSTRACT, CONTENT, RESEARCH, [])
        assert (relevant, reason) == (False, "deep-parse-failed")
