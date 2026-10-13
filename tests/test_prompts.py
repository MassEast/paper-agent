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
    assert '"reason"' in result  # why-relevant line, stored per project


def test_relevance_reason_format():
    result = prompts.RELEVANCE_REASON.format(
        research_interest="long-context LLMs", title="Some Paper", abstract="Some abstract."
    )
    assert "long-context LLMs" in result
    assert "Some Paper" in result
    assert '{"reason"' in result


def test_summary_format():
    # No research interest: the summary is shared by every project the paper is in
    assert "{research_interest}" not in prompts.SUMMARY_WITH_FULL_CONTENT
    result = prompts.SUMMARY_WITH_FULL_CONTENT.format(
        title="Fast Attention",
        abstract="We propose...",
        content="Introduction...",
    )
    assert "Fast Attention" in result
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
                 "SUMMARY_WITH_FULL_CONTENT", "MAIN_CONTRIBUTIONS", "RELEVANCE_REASON",
                 "RESEARCH_INTEREST_FROM_COLLECTION"]:
        tmpl = getattr(prompts, name)
        assert isinstance(tmpl, str) and len(tmpl) > 50, f"{name} is too short"


def test_readme_prompt_links_point_at_the_prompts():
    """README links each prompt as app/prompts.py#L<start>-L<end>; editing prompts.py shifts them."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    lines = (root / "app" / "prompts.py").read_text().split("\n")
    links = re.findall(r"\[`([A-Z_]+)`\]\(app/prompts\.py#L(\d+)-L(\d+)\)", (root / "README.md").read_text())
    assert links
    for name, start, end in links:
        assert lines[int(start) - 1].startswith(f'{name} = """'), f"README link for {name} starts at the wrong line"
        assert lines[int(end) - 1].rstrip().endswith('"""'), f"README link for {name} ends at the wrong line"


def test_templates_use_app_tooltips_not_native_title():
    """Hover hints use data-tip (fast, app-styled tooltip in base.html), never the native title attribute."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "app" / "templates"
    offenders = [f"{p.name}:{i}" for p in root.rglob("*.html")
                 for i, line in enumerate(p.read_text().split("\n"), 1) if re.search(r'\stitle="', line)]
    assert not offenders, f"use data-tip instead of title= in {offenders}"
