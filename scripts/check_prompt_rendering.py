"""
Renders the real prompt templates from app/prompts.py through the actual
selection/formatting code in app/crawl.py, using synthetic My Collection
papers, and prints the fully-substituted prompt text for visual review.

No LLM call, no network, no DB — just proves _select_reference_papers,
_collection_context_text, and the prompt .format() calls produce clean
output with no leftover placeholders, no crashes, no truncation surprises.

Run: .venv/bin/python3.11 scripts/check_prompt_rendering.py
"""
import os
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

os.environ.setdefault("LLM_API_KEY", "test-key-not-real")
os.environ.setdefault("LLM_API_BASE", "http://test-llm-endpoint.invalid/v1")
os.environ.setdefault("LLM_MODELS", "test-model-large,test-model-small")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("SITE_PASSWORD", "testpass")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import prompts
from app.crawl import _select_reference_papers, _collection_context_text, REFERENCE_PAPERS_LIMIT


def _pp(title, abstract, tags, days_ago):
    return SimpleNamespace(
        paper=SimpleNamespace(title=title, abstract=abstract),
        trashed_at=None,
        manual_tag="related",
        tags_list=tags,
        collected_at=datetime.now() - timedelta(days=days_ago),
    )


SAMPLE_COLLECTION = [
    _pp("Rys: Understanding LLM Internals via Attribution Graphs",
        "We introduce a method for tracing feature circuits inside transformer "
        "language models using sparse autoencoders and attribution patching, "
        "enabling mechanistic interpretation of multi-step reasoning.",
        ["important"], days_ago=40),
    _pp("Emergent World Models in Sequence Models",
        "We show that sequence models trained purely on next-token prediction "
        "develop internal representations that resemble world models, tested "
        "via linear probes on board-game state.",
        ["important"], days_ago=12),
    _pp("Sparse Feature Circuits for Model Editing",
        "We propose a technique for identifying and editing sparse causal "
        "circuits responsible for specific model behaviors, using it to "
        "localize and suppress unwanted associations.",
        ["important"], days_ago=3),
    _pp("A Survey of Mechanistic Interpretability",
        "We survey the growing literature on mechanistic interpretability, "
        "covering circuit analysis, probing, and causal intervention methods.",
        ["to_discuss"], days_ago=20),
    _pp("Polysemanticity and Superposition in Neural Networks",
        "We study how neural networks represent more features than they have "
        "dimensions, via superposition, and its implications for "
        "interpretability.",
        ["to_read"], days_ago=8),
    _pp("Scaling Laws for Sparse Autoencoders",
        "We study how sparse autoencoder reconstruction quality scales with "
        "dictionary size and training compute across model sizes.",
        [], days_ago=25),
]

RESEARCH_INTEREST = (
    "I'm interested in mechanistic interpretability of large language "
    "models, particularly attribution-based methods for tracing how "
    "multi-step reasoning emerges from individual attention and MLP "
    "components."
)


def _print_header(label):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")


def _check_no_placeholders(rendered: str, label: str) -> bool:
    import re
    leftover = re.findall(r"\{[a-zA-Z_]+\}", rendered)
    if leftover:
        print(f"  !! UNRESOLVED PLACEHOLDERS in {label}: {leftover}")
        return False
    return True


def main():
    ok = True

    print(f"REFERENCE_PAPERS_LIMIT = {REFERENCE_PAPERS_LIMIT}")
    print(f"Sample collection size = {len(SAMPLE_COLLECTION)} "
          f"(3 important, 1 to_discuss, 1 to_read, 1 untagged)")

    selected = _select_reference_papers(SAMPLE_COLLECTION, limit=REFERENCE_PAPERS_LIMIT)
    print(f"\n_select_reference_papers() picked {len(selected)} papers, in order:")
    for pp in selected:
        print(f"  - [{','.join(pp.tags_list) or 'untagged'}] {pp.paper.title}")

    # --- KEYWORD_EXTRACTION ---
    collection_context = _collection_context_text(SAMPLE_COLLECTION)
    rendered = prompts.KEYWORD_EXTRACTION.format(
        research_interest=RESEARCH_INTEREST,
        collection_context=collection_context,
    )
    _print_header("KEYWORD_EXTRACTION")
    print(rendered)
    ok &= _check_no_placeholders(rendered, "KEYWORD_EXTRACTION")

    # --- RELEVANCE_QUICK (mirrors is_paper_relevant's ref_text construction) ---
    collection_ref_dicts = [
        {"title": pp.paper.title, "abstract": (pp.paper.abstract or "")[:400]}
        for pp in selected
    ]
    ref_text = "Researcher's collection (for context on what is relevant):\n" + "\n".join(
        f"- {p.get('title', '')}: {p.get('abstract', '')[:400]}" for p in collection_ref_dicts
    )
    rendered = prompts.RELEVANCE_QUICK.format(
        research_interest=RESEARCH_INTEREST,
        ref_text=ref_text,
        title="Steering Language Models via Sparse Feature Interventions",
        abstract="We show that individual sparse-autoencoder features can be "
                 "used as steering vectors to reliably control specific "
                 "behaviors in language model generation.",
    )
    _print_header("RELEVANCE_QUICK")
    print(rendered)
    ok &= _check_no_placeholders(rendered, "RELEVANCE_QUICK")

    # --- RESEARCH_INTEREST_FROM_COLLECTION (mirrors generate_research_interest_from_collection) ---
    papers_text = "\n\n".join(
        f"Paper {i+1}: {pp.paper.title}\nAbstract: {(pp.paper.abstract or '')[:500]}"
        for i, pp in enumerate(selected)
    )
    rendered = prompts.RESEARCH_INTEREST_FROM_COLLECTION.format(papers_text=papers_text)
    _print_header("RESEARCH_INTEREST_FROM_COLLECTION")
    print(rendered)
    ok &= _check_no_placeholders(rendered, "RESEARCH_INTEREST_FROM_COLLECTION")

    # --- RESEARCH_INTEREST_IMPROVE (the refinement branch, with refs_section populated) ---
    collection_section = f"\nAlso incorporate these papers from the researcher's collection:\n{papers_text}\n"
    rendered = prompts.RESEARCH_INTEREST_IMPROVE.format(
        existing_text=RESEARCH_INTEREST,
        refs_section=collection_section,
    )
    _print_header("RESEARCH_INTEREST_IMPROVE")
    print(rendered)
    ok &= _check_no_placeholders(rendered, "RESEARCH_INTEREST_IMPROVE")

    # --- Empty-collection edge case (new project, nothing tagged yet) ---
    empty_ref_text = ""
    rendered = prompts.RELEVANCE_QUICK.format(
        research_interest=RESEARCH_INTEREST,
        ref_text=empty_ref_text,
        title="Some Candidate Paper",
        abstract="Some abstract.",
    )
    _print_header("RELEVANCE_QUICK (empty collection — ref_text='')")
    print(rendered)
    ok &= _check_no_placeholders(rendered, "RELEVANCE_QUICK (empty)")

    print(f"\n{'=' * 78}")
    if ok:
        print("All templates rendered with no leftover {placeholders}.")
    else:
        print("FAILED — see !! markers above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
