"""
All LLM prompt templates used by the crawl pipeline.
Edit here to tune all prompts in one place.
"""

KEYWORD_EXTRACTION = """\
Extract 3-6 arXiv search keyword phrases from the research interest and the researcher's paper collection below.
Each phrase must be 1-4 words long and suitable for an arXiv abstract search.
Prefer specific technical terms over generic ones.

Return a JSON array of strings and nothing else. Example output:
["mixture of experts", "sparse attention transformer", "long-context language model"]

Research interest:
{research_interest}

Researcher's paper collection (titles + abstracts — use to learn domain vocabulary):
{collection_context}"""

RELEVANCE_QUICK = """\
Evaluate if this paper is relevant for the research project.

Research interest:
{research_interest}

{ref_text}

Candidate paper:
Title: {title}
Abstract: {abstract}

Return JSON only: {{"relevant": true}} or {{"relevant": false}}"""

RELEVANCE_DEEP = """\
Evaluate if this paper is relevant for the research project.
Use the full content for a thorough evaluation.

Research interest:
{research_interest}

{ref_text}

Candidate paper:
Title: {title}
Abstract: {abstract}

Full content (key sections - intro, methods, conclusion):
{paper_content}

If it is relevant, also say why: one sentence of at most 30 words, addressed to the researcher \
("you"), naming the concrete problem, method, or finding of the paper that connects to their \
research interest. No praise, no restating the title.

Return JSON only: {{"relevant": true, "reason": "<why>"}} or {{"relevant": false}}"""

# Deliberately without the research interest: a paper's summary is stored once and shared by
# every project the paper is in. Why it matters for one project lives in
# ProjectPaper.relevance_reason (RELEVANCE_DEEP / RELEVANCE_REASON).
SUMMARY_WITH_FULL_CONTENT = """\
Analyze this paper.

Title: {title}

Abstract: {abstract}

Full paper content (intro, methods, results, conclusion):
{content}

Return JSON with:
- "summary": 3-sentence key-contribution summary
- "introduces_dataset": bool - does it introduce a new dataset?
- "introduces_architecture": bool - does it propose new model architecture?
- "introduces_method": bool - does it introduce new methodology?
- "is_survey": bool - is it a survey/review paper?
- "is_benchmark": bool - does it propose benchmarks?"""

RELEVANCE_REASON = """\
This paper is on a researcher's reading list for the project below. In one sentence of at most \
30 words, addressed to the researcher ("you"), name the concrete problem, method, or finding of the \
paper that connects to their research interest. If the connection is only loose, start with \
"Loosely related:" and name the actual overlap instead of stretching it. No praise, no restating the title.

Research interest:
{research_interest}

Paper:
Title: {title}
Abstract: {abstract}

Return JSON only: {{"reason": "<why>"}}"""

MAIN_CONTRIBUTIONS = """\
Extract the MAIN CONTRIBUTIONS of this paper. Look in the introduction and conclusion/discussion sections for what the authors claim as their key contributions.

Title: {title}

Full paper content:
{content}

Return JSON with:
- "main_contributions": list of 2-4 bullet points describing the paper's main contributions
  (use the original phrasing from the paper, paraphrase minimally)
- "contribution_summary": 1-sentence overview of the primary contribution"""

RESEARCH_INTEREST_FROM_COLLECTION = """\
Based on these papers from the researcher's collection, generate a research interest description
that captures the common themes, methodologies, and research directions.

Papers (title + abstract excerpt):
{papers_text}

Generate a tight 2-3 paragraph description of the research interests they represent, at most 300 words.
Cover: what problems are being solved, what methods/architectures are used,
what domains are being studied, and what the key trends are.
Only state methods, techniques, or architectures that are actually present in the papers above —
do not invent or generalize to methods that aren't there. Prefer concrete, specific claims over
broad ones. Avoid generic filler (e.g. "this promises to revolutionize X") — every sentence should
carry real information.

Return just the description, no headers or extra formatting."""

FEATURED_PAPER_SELECTION = """\
From the list of newly found papers below, pick the single most interesting and relevant one \
to highlight in the crawl-complete email. Prefer papers that introduce a new method, architecture, \
or dataset — not surveys. Papers with more citations are generally more impactful. \
Choose the one most directly relevant to the research interest.

Research interest:
{research_interest}

Papers (numbered, with citation counts):
{papers}

Return JSON only: {{"selected": <number>}}"""

INSTITUTION_EXTRACTION = """\
Extract the author institution/affiliation names from the {source_label} of this paper.
Return a JSON array of strings, e.g. ["MIT CSAIL", "Stanford University"].
Return [] if none can be found.

Title: {title}

{text}

Return JSON array only."""

RESEARCH_INTEREST_IMPROVE = """\
Refine the researcher's existing description of their research interests into a tight 2-3 paragraph version of at most 300 words.
If the existing description is longer, condense it: keep what the research is about and drop design details, formatting and repetition.
Use the provided papers as background context to better understand the research scope — but do NOT mention specific papers, authors, or titles in the output.
The result should read as a first-person description of research goals, problems being solved, methods and approaches of interest, and open questions — not as a literature review.

Do not invent or add specific methods, techniques, or algorithms (e.g. "reinforcement learning",
"contrastive learning", "curriculum learning") that are not stated or clearly implied by the
existing description below. Where the existing description leaves a method unspecified, describe
the goal or problem instead of inventing a technique to fill the gap. Keep any concrete examples,
analogies, or illustrative scenarios already present in the existing description — translate them
into clean research language rather than dropping them for generic phrasing. Avoid generic filler
(e.g. "this promises to revolutionize X") — every sentence should carry real information.
{refs_section}
Current description (refine this, keeping its intent, voice, and concrete examples):
{existing_text}

Return just the improved description, no headers or meta-commentary."""

# System message of the "Ask about this paper" chat on a card (app/chat.py). The earlier turns of the
# conversation follow as plain user/assistant messages.
PAPER_CHAT_SYSTEM = """\
You are helping a researcher understand one paper in light of their own project, described below. Read \
their terms and abbreviations in the sense of that project, assume they know the basics of their field, \
and relate your answers to their project where that helps. Answer from the paper text below. Say so when \
the text does not cover something, and mark anything you add from outside the paper as your own \
background knowledge. Refer to sections, figures, tables and equations by their names or numbers when \
that helps the researcher find them. Be concise; use markdown.

The researcher's project:
{research_interest}
{keywords_line}{notes_line}{relevance_line}
Paper: {title}
Authors: {authors}

{source_note}
---
{text}
---"""

PAPER_CHAT_SOURCE_NOTES = {
    "html": "Full text, from the paper's arXiv HTML version (figures are not included; tables and "
            "math may be garbled):",
    "pdf": "Full text, extracted from the PDF (figures are not included; layout, tables and math "
           "may be garbled):",
    "abstract": "Only the abstract is available, not the full text. When a question cannot be "
                "answered from the abstract alone, say so plainly instead of guessing:",
}
