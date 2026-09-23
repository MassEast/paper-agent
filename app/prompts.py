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

Return JSON only: {{"relevant": true}} or {{"relevant": false}}"""

SUMMARY_WITH_FULL_CONTENT = """\
Analyze this paper for a research project on: {research_interest}

Title: {title}

Abstract: {abstract}

Full paper content (intro, methods, results, conclusion):
{content}

Return JSON with:
- "summary": 3-sentence key-contribution summary
- "key_findings": list of 3-5 main findings
- "methodology": brief description of approach
- "limitations": potential weaknesses if any
- "introduces_dataset": bool - does it introduce a new dataset?
- "introduces_architecture": bool - does it propose new model architecture?
- "introduces_method": bool - does it introduce new methodology?
- "is_survey": bool - is it a survey/review paper?
- "is_benchmark": bool - does it propose benchmarks?"""

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

Generate a tight 2-3 paragraph description of the research interests they represent.
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
Refine the researcher's existing description of their research interests into a tight 2-3 paragraph version.
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
