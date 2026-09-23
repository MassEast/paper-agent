"""
Sanity checks for app/prompts.py.
These run instantly with no external calls — just verify the templates are
well-formed and won't blow up at runtime with the expected placeholders.
"""

import app.prompts as prompts


def test_keyword_extraction_format():
    result = prompts.KEYWORD_EXTRACTION.format(
        research_interest="efficient transformers",
        collection_context="Paper 1: Attention Is All You Need\nAbstract: We propose...",
    )
    assert "efficient transformers" in result
    assert "Attention Is All You Need" in result


def test_relevance_quick_format():
    result = prompts.RELEVANCE_QUICK.format(
        research_interest="long-context LLMs",
        ref_text="Researcher's collection (for context on what is relevant):\n- Some paper",
        title="Flash Attention 3",
        abstract="We improve attention speed.",
    )
    assert "Flash Attention 3" in result
    assert "long-context LLMs" in result
    assert '{"relevant"' in result


def test_relevance_deep_format():
    result = prompts.RELEVANCE_DEEP.format(
        research_interest="long-context LLMs",
        ref_text="",
        title="Some Paper",
        abstract="Some abstract.",
        paper_content="Full text here...",
    )
    assert "Full text here" in result
    assert "long-context LLMs" in result


def test_summary_format():
    result = prompts.SUMMARY_WITH_FULL_CONTENT.format(
        research_interest="efficient attention",
        title="Fast Attention",
        abstract="We propose...",
        content="Introduction...",
    )
    assert "Fast Attention" in result
    assert "efficient attention" in result
    assert "introduces_dataset" in result


def test_main_contributions_format():
    result = prompts.MAIN_CONTRIBUTIONS.format(
        title="My Paper",
        content="Introduction: We contribute...",
    )
    assert "My Paper" in result
    assert "main_contributions" in result


def test_research_interest_from_collection_format():
    result = prompts.RESEARCH_INTEREST_FROM_COLLECTION.format(
        papers_text="Paper 1: Attention...\nPaper 2: Transformers...",
    )
    assert "Paper 1" in result
    assert "Paper 2" in result


def test_all_prompts_are_nonempty():
    for name in ["KEYWORD_EXTRACTION", "RELEVANCE_QUICK", "RELEVANCE_DEEP",
                 "SUMMARY_WITH_FULL_CONTENT", "MAIN_CONTRIBUTIONS",
                 "RESEARCH_INTEREST_FROM_COLLECTION"]:
        tmpl = getattr(prompts, name)
        assert isinstance(tmpl, str) and len(tmpl) > 50, f"{name} is too short"
