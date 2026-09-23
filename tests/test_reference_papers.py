"""
Tests for _select_reference_papers (app/crawl.py) — the My Collection paper selection
used to build LLM reference/calibration context for keyword extraction, relevance
screening, and research-interest generation/refinement. Pure selection logic for the
first class below, no DB needed — uses lightweight stand-ins for ProjectPaper/Paper
rather than real ORM objects.
"""
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from app.crawl import _select_reference_papers, _collection_context_text


def _pp(title, tags, collected_at, trashed=False):
    """Build a minimal ProjectPaper-like stand-in."""
    return SimpleNamespace(
        paper=SimpleNamespace(title=title, abstract=f"Abstract for {title}"),
        trashed_at=datetime(2026, 1, 1) if trashed else None,
        manual_tag="related",
        tags_list=tags,
        collected_at=collected_at,
    )


class TestSelectReferencePapers:
    def test_fewer_papers_than_limit_returns_all(self):
        base = datetime(2026, 1, 1)
        pps = [_pp(f"P{i}", [], base + timedelta(days=i)) for i in range(3)]
        selected = _select_reference_papers(pps, limit=8)
        assert len(selected) == 3
        assert {pp.paper.title for pp in selected} == {"P0", "P1", "P2"}

    def test_important_tier_prioritized_over_related(self):
        base = datetime(2026, 1, 1)
        important = [_pp("Important", ["important"], base)]
        related = [_pp(f"Related{i}", [], base + timedelta(days=i)) for i in range(5)]
        selected = _select_reference_papers(important + related, limit=2)
        titles = [pp.paper.title for pp in selected]
        assert "Important" in titles
        assert len(selected) == 2

    def test_tier_priority_order_important_discuss_read_related(self):
        base = datetime(2026, 1, 1)
        pps = [
            _pp("Related", [], base),
            _pp("ToRead", ["to_read"], base),
            _pp("ToDiscuss", ["to_discuss"], base),
            _pp("Important", ["important"], base),
        ]
        selected = _select_reference_papers(pps, limit=4)
        titles = [pp.paper.title for pp in selected]
        assert titles == ["Important", "ToDiscuss", "ToRead", "Related"]

    def test_overflowing_tier_alternates_newest_and_oldest(self):
        base = datetime(2026, 1, 1)
        # 5 important papers, oldest (P0) to newest (P4) by collected_at
        pps = [_pp(f"P{i}", ["important"], base + timedelta(days=i)) for i in range(5)]
        selected = _select_reference_papers(pps, limit=3)
        titles = [pp.paper.title for pp in selected]
        # newest first (P4), then oldest (P0), then 2nd-newest (P3)
        assert titles == ["P4", "P0", "P3"]

    def test_overflow_in_first_tier_stops_before_next_tier(self):
        base = datetime(2026, 1, 1)
        important = [_pp(f"I{i}", ["important"], base + timedelta(days=i)) for i in range(5)]
        related = [_pp("Related", [], base)]
        selected = _select_reference_papers(important + related, limit=2)
        titles = [pp.paper.title for pp in selected]
        assert "Related" not in titles
        assert len(titles) == 2

    def test_fills_remaining_slots_from_next_tier(self):
        base = datetime(2026, 1, 1)
        important = [_pp("Important", ["important"], base)]
        related = [_pp(f"R{i}", [], base + timedelta(days=i)) for i in range(5)]
        selected = _select_reference_papers(important + related, limit=3)
        titles = [pp.paper.title for pp in selected]
        assert titles[0] == "Important"
        assert len(titles) == 3
        assert all(t.startswith("R") for t in titles[1:])

    def test_trashed_and_untagged_papers_excluded(self):
        base = datetime(2026, 1, 1)
        trashed = _pp("Trashed", ["important"], base, trashed=True)
        untagged = SimpleNamespace(
            paper=SimpleNamespace(title="Untagged", abstract=""),
            trashed_at=None,
            manual_tag=None,
            tags_list=[],
            collected_at=base,
        )
        kept = _pp("Kept", [], base)
        selected = _select_reference_papers([trashed, untagged, kept], limit=8)
        assert [pp.paper.title for pp in selected] == ["Kept"]

    def test_empty_input_returns_empty(self):
        assert _select_reference_papers([], limit=8) == []


class TestCollectionContextText:
    def test_empty_collection_returns_placeholder(self):
        text = _collection_context_text([])
        assert "none yet" in text

    def test_formats_title_and_truncated_abstract(self):
        pp = _pp("My Paper", ["important"], datetime(2026, 1, 1))
        pp.paper.abstract = "x" * 600
        text = _collection_context_text([pp])
        assert text.startswith("- My Paper: ")
        # abstract truncated to 500 chars in the context string
        assert len(text) - len("- My Paper: ") == 500


class TestGenerateResearchInterestFromCollection:
    """generate_research_interest_from_collection used to have its own duplicate
    tag-priority-only selection (hardcoded [:15]) — now delegates to
    _select_reference_papers like everything else. These tests confirm that
    delegation actually happens (tier priority + REFERENCE_PAPERS_LIMIT respected)."""

    def test_uses_tag_priority_and_respects_limit(self, app):
        with app.app_context():
            from app.crawl import generate_research_interest_from_collection

            base = datetime(2026, 1, 1)
            important = [_pp(f"Important{i}", ["important"], base + timedelta(days=i)) for i in range(3)]
            related = [_pp(f"Related{i}", [], base + timedelta(days=i)) for i in range(3)]
            captured_prompt = {}

            def _fake_llm(messages, **kwargs):
                captured_prompt["text"] = messages[0]["content"]
                return "generated research interest", "test-model", 42, {"total_tokens": 100}

            with patch("app.crawl._llm", side_effect=_fake_llm), \
                 patch("app.crawl.REFERENCE_PAPERS_LIMIT", 2):
                result = generate_research_interest_from_collection(important + related)

            assert result == "generated research interest"
            prompt_text = captured_prompt["text"]
            # limit=2, all-important tier has 3 candidates -> overflow -> newest/oldest alternation
            assert "Important2" in prompt_text  # newest
            assert "Important0" in prompt_text  # oldest
            assert "Related" not in prompt_text  # important tier already filled the limit
