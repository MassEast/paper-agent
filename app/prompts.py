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

Full text (may be cut off):
{content}

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

PAPER_RESOURCES = """\
From a research paper, pick out its own code repository, its own project page, its corresponding authors' emails
and the venue it was accepted at. Only use the candidates listed below; copy them exactly.

- code_url: the repository where the authors release THIS paper's code (or models/data, if that is all they
  release). Not a tool, library, baseline, dataset or other paper's code that they only use or compare with.
  null if none of the URL candidates is the paper's own code.
- project_url: THIS paper's own website or project page (often a *.github.io page). Not a lab, company or
  personal homepage, not a blog post about something else, not the code repository. null if there is none.
- corresponding: the corresponding authors with their emails. Corresponding authors are marked after their
  name, and a legend says what each mark means (for example "*Equal contribution, †Corresponding author"): a
  mark for equal contribution or anything else does not count. "Correspondence to: <names or emails>" also
  names them. Read the marks from the first page of the PDF when it is given (the HTML text can lose them).
  If the paper says there are corresponding authors but you cannot tell who (for example "Correspondence to:
  , ." with the names lost), return an empty list rather than guess. Only if the paper marks nobody at all,
  take the author of the first email listed. Match an email to an author by the name in it, the order, or the
  context. Use author names exactly as in the author list and emails exactly as in the email candidates. Empty
  list if there are no email candidates or a corresponding author has no email among them.
- venue: where the paper was accepted or published, if the comment, the journal reference or the start of the
  paper says so (for example "Published as a conference paper at ICLR 2025", "Accepted at NeurIPS 2026",
  "To appear in TMLR"). Short common name with year and, for workshops, "Workshop", e.g. "ICLR 2025",
  "NeurIPS 2026 Workshop", "ACL 2025 Findings", "TMLR". null for "under review", "preprint", "submitted to",
  or if nothing is said.

Title: {title}
Authors:
{authors}

arXiv comment: {comment}
Journal reference: {journal_ref}

Start of the paper:
{start}

First page of the PDF (marks after the author names and their legend):
{first_page}

URL candidates:
{urls}

Email candidates:
{emails}

Return JSON only:
{{"code_url": "<url or null>", "project_url": "<url or null>", "corresponding": [{{"author": "<name>", "email": "<email>"}}], "venue": "<venue or null>"}}"""

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
and relate your answers to their project where that helps. Talk to them directly, as "you", not about \
"the user" or "the researcher". Answer from the paper text below. Say so when \
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
