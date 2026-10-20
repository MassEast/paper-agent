# AGENTS.md

Conventions for anyone (human or AI agent) working on this codebase.

## What this is

A Flask-based literature review agent. It crawls arXiv and Semantic Scholar for new papers, screens them with an LLM against your research interest, generates summaries, and sends email notifications. Multiple research projects are tracked independently in one SQLite DB.

## Dev commands

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # app deps + pytest/pytest-timeout/playwright (then: python -m playwright install chromium)
cp .env.example .env   # fill in your LLM_API_BASE / LLM_API_KEY / LLM_MODELS etc.

# Run the dev server (LOG_LLM=1 logs every LLM prompt/response)
LOG_LLM=1 .venv/bin/python -m flask run --port 5001

# Run tests — use python3.11 -m pytest, not bare `pytest` (broken shebang in some venvs)
# always pass -m "not slow": a couple of test files mix fast + slow (real-API) tests
.venv/bin/python3.11 -m pytest tests/ -v -m "not slow"

# Run a single test file
.venv/bin/python3.11 -m pytest tests/test_crawl.py -v -m "not slow"

# Run the live/slow tests (hit real APIs — needs network + a working LLM endpoint)
.venv/bin/python3.11 -m pytest tests/ -v -m slow --timeout=120

# Inspect the DB
sqlite3 /tmp/related_work.db "SELECT id, name, crawl_hour, saved_keywords FROM projects;"
```

## Required environment variables

See `.env.example` for the full list with comments. The hard-required ones (app raises `EnvironmentError` at import time if any are missing): `LLM_API_KEY`, `LLM_API_BASE`, `LLM_MODELS`, `SECRET_KEY`, `SITE_PASSWORD`.

## Architecture

### Request flow

```
Browser → Flask blueprint routes (app/routes/) → crawl.py / llm.py → SQLite (Flask-SQLAlchemy)
```

The UI is server-rendered Jinja2 with **HTMX** for partial updates (no JS framework). TailwindCSS via CDN. Design tokens in `app/static/tokens.css`.

### LLM layer (`app/llm.py`)

Single entry point for all LLM calls, against any endpoint that speaks the standard `/chat/completions` API (vLLM, Ollama, etc.). Two public functions:
- `_llm(messages, temperature, max_tokens)` → `(text, model, elapsed_ms, usage_dict)`
- `_llm_json(messages, ...)` → `(parsed_dict_or_list, model, elapsed_ms, usage_dict)` — strips `<think>` blocks, markdown fences, and falls back to regex extraction if JSON parsing fails

Model cascade: tries each model in `LLM_MODELS` (comma-separated env var) in order, with 5 retries per model and exponential backoff, before falling through to the next. All prompts live in `app/prompts.py` — edit there, not inline.

`_llm_stream(messages, ...)` is the streaming variant used by the paper chat: it yields the visible answer so far (empty while the model is still inside its leading `<think>` block), a `("retry", model, attempt)` tuple before each retry, and finally `{"model", "usage", "finish_reason"}`. It only falls back to the next model before the first chunk arrives; a 400 (prompt too long) is not retried on the same model.

### Paper chat (`app/chat.py`)

"Ask about this paper" on a card: one conversation per `ProjectPaper` (`ChatMessage` rows; the table uses AUTOINCREMENT so ids are never reused after a clear, since they key in-memory state and the pages' polls). Each turn sends `PAPER_CHAT_SYSTEM` (research interest, why-relevant line, the paper's text) plus the earlier finished questions and answers; thinking is never stored or resent. The text is fetched once and kept in `PaperText` (arXiv HTML via `fetch_paper_full_text`, else the arXiv PDF, else an open-access PDF for web papers, else none → "abstract only", retried next question).

One worker thread per chat (`_running`, `_lock`) works through the questions in order: a question asked while an answer streams is stored as `queued`. The visible answer lives in `_live` (in memory, so the single-process rule below applies); the card polls `/chat/message/<id>` every 700 ms, which returns the answer so far and, once it ended, the conversation log plus the head (source label, context meter) out of band, never the input form. Stop sets a flag the worker checks per chunk and closes the HTTP stream. The worker never holds a SQLite write transaction while streaming, and writes results with `UPDATE ... WHERE id`, so clearing a chat mid-answer is safe. A restart leaves a `streaming` row with no worker: `chat.view()` marks it interrupted and restarts the queue. The context meter shows the model's `prompt_tokens` for the last answer plus the answer's length (3.2 chars/token, measured), estimated live while streaming, against `CHAT_CONTEXT_TOKENS`; a question that would not fit is refused instead of dropping old turns. Tests: `tests/test_paper_chat.py` (scripted fake model).

### Token usage (`app/usage.py`)

Every action that calls the model runs inside `usage.counted(project_id, kind)` (or the `@counts(kind)` decorator on `run_crawl`, `send_carry_notification`, `run_rescreen`, `backfill_missing_paper_metadata`). `llm._llm`/`_llm_stream` add each completion's tokens to the current meter (a contextvar; a stopped or broken chat stream counts an estimate), and one `LlmUsage` row (`project_id`, `kind`, `tokens`, `calls`) is written when the action ends, also if it raised. Kinds: `crawl`, `chat`, `add`, `regenerate`, `rescreen`, `relevance_reason`, `keywords`, `research_interest`, `metadata` (shared papers, `project_id` NULL). A thread pool inside a counted action must submit with `contextvars.copy_context().run`, or its calls go uncounted (logged as `[usage] LLM call outside any usage scope`). Project totals (project list, footer, the "LLM Usage" panel under the crawl calendar) sum these rows; `CrawlLog.total_tokens` stays the per-crawl figure shown in the crawl status. Crawls from before the table existed were backfilled from `crawl_logs` (`USAGE_BACKFILL_SQL`); other kinds are only counted since 2026-10-15. `PaperSummary.tokens_used` is just the summary + contributions of the latest run (Regenerate replaces it).

### Full text and what was read (`utils.fetch_paper_full_text`, `crawl.get_paper_content`, `crawl.refresh_from_full_text`)

The full text comes from arXiv's own HTML (`arxiv.org/html/<id>`, there from announcement day for most papers), then ar5iv, then the PDF (`utils.fetch_pdf_text`); only the page's `<article>` is kept. ar5iv converts with a lag of weeks and until then redirects to the abstract page; before 2026-10-15 that page passed for the full text (ar5iv was the only source), so the deep check, summaries and chat of recent papers read the abstract page. A redirect away from `/html/` now counts as no full text. `ProjectPaper.screen_source` records what the deep check read (`full_text`/`abstract`; `unknown` = marked for a re-read; NULL = not screened or from before), `PaperSummary.content_source` what the summary read; both show on the card. The nightly script then runs `refresh_from_full_text` per project: New Papers with `screen_source` `abstract`/`unknown`, newest first, `FULL_TEXT_REFRESH_PER_NIGHT` (50) per night, are deep-checked again with the full text; clear rejects go to Trash with `trashed_reason` (shown in the Trash row, cleared on restore, no email), kept papers get a new why-line and, if the summary wasn't from the full text, a new summary. My Collection is never touched. No full text or an unparsable answer counts a try (`Paper.full_text_attempts`, max 3). Usage kind `refresh`.

### Crawl pipeline (`app/crawl.py`)

`run_crawl(project_id, date_from, date_to, ...)` is the main entrypoint, always called via `start_crawl_async(...)` which wraps it in a daemon thread. Flow:

1. Extract/reuse keywords from research interest via LLM — if `project.saved_keywords` is set, uses those directly (no LLM call). Auto-saves keywords after first successful generation. The LLM prompt also gets titles+abstract excerpts from up to `REFERENCE_PAPERS_LIMIT` "My Collection" papers (`_select_reference_papers`) as domain-vocabulary context.
2. Search arXiv by keyword (`abs:"phrase"` for multi-word) with a server-side `submittedDate` filter. arXiv caps a query at 200 results, newest first, so when a query hits the cap the search continues from the oldest paper's date back to the start of the window (`_collect_arxiv`); a single day over the cap or a timeout is surfaced as a warning in the UI. All of a project's keywords go into **one** OR query (`_fetch_arxiv_window`; same papers as one query per keyword, checked live 2026-10-14) because arXiv answers about 1 in 4 nightly queries with a 429 even at one request per 5 s, so fewer requests means fewer failures. A 429 or 503 is waited out (120 s, then 240 s); a query that still errors out marks the crawl `error` with all keywords in `failed_keywords`, and the nightly script retries it up to `NIGHTLY_RETRY_ROUNDS` (default 2) times at the end of the run, each after `NIGHTLY_RETRY_WAIT_S` (default 600). One email per project per night: every pass but the last defers its email (`defer_notification`/`notify_carry`, numbers added up by `_merge_carry`, including the per-source counts), and the last sends a combined one, with an arXiv-warning banner if arXiv never answered (`send_carry_notification` covers a retry that can't send). If arXiv never answered and nothing else was found, the email says so plainly ("arXiv search could not run tonight"; nothing is lost yet since each night searches the last 5 days again), not "Nightly crawl failed". The featured-paper pick considers all papers added. arXiv ids are stored unversioned (`abs/<id>` always resolves to the latest version), and duplicate checks ignore any version suffix (`base_arxiv_id`).. Separately, if the project has any "My Collection" papers, seed Semantic Scholar's `/recommendations` API with their arXiv ids, or DOI / Semantic Scholar id for papers not on arXiv (see "Semantic Scholar sources beyond arXiv"; `get_semantic_scholar_recommendations`) — **not** keyword-driven; skipped entirely if the Collection is empty. Seeds: up to 100 Collection papers, best-tagged first (the API's maximum), not the capped selection below. Asked for in groups of 10 seeds with limit 500 (the API's maximum) each: a call returns its 500 ranked by similarity over the last few months, so few are from the crawl window, and smaller groups each get their own 500 and together reach more recent ones. In a comparison, smaller groups brought clearly more papers from the window at about the same screening pass rate; one call per seed (`/forpaper` with `from=recent`) brought far more but mostly unrelated ones, so it isn't used. Constants `RECOMMENDATION_*` in `crawl.py`. Third source: new papers that cite a My Collection paper (`get_citing_papers`, Semantic Scholar `/paper/<seed>/citations`, one request per My Collection paper Semantic Scholar knows, at most `CITATION_MAX_SEEDS` (50) per crawl, the ones checked longest ago first (`citation_seeds`, by `ProjectPaper.citations_checked_at`, set by `mark_citations_checked` only for seeds Semantic Scholar answered, so one it gave up on with 429s comes first next time; the Find Papers count step stamps too), so a larger collection is covered in turn over a few nights (174 seeds took 2 h 41 min under Semantic Scholar 429s on 2026-10-19 and held up the next projects), newest first, paged only while a page still reaches into the window; papers dated in the window only, arXiv, bioRxiv and journal ones). `gather_candidates` runs all three (used by `run_crawl` and the manual count step, each source switchable there with a checkbox, always all three nightly, plus bioRxiv for projects that switched it on, see below) and merges a paper found more than once into one candidate whose `found_via` lists every way (keyword by name, guessed by `_matched_keywords` since the OR query doesn't say which matched; `recommended`; `cites` + the seed's id and title), stored per project in `ProjectPaper.found_via` and shown on the card ("Found via …", `found_via_text`; the tooltip says it in sentences, `found_via_tip`). The email has one row per source (`EMAIL_SOURCE_LABELS`) with found and added counts from `found_via` (`_count_sources`, carried across retry passes as `notify_carry["sources"]`); a paper found several ways counts in each of its rows, the total counts it once. Batching the citation lookups (`/paper/batch` with `citations.*` fields returns every citation, newest first) was considered and left out: it returns all citations rather than just the window, and timed out with a very highly cited paper in the batch. It is the slowest source: one request per seed, and Semantic Scholar often answers with 429s, so a large collection takes several minutes, and much longer on a bad night. Widely cited seeds pull in loosely related papers; screening is what filters those. The nightly retry round is arXiv only (`use_scholar=False, use_citations=False`): the first pass already screened the other two.
3. For each candidate paper: quick relevance check (abstract only) → deep check (full text, fetched only for papers that passed the quick check; also returns a one-sentence "why it's relevant", stored per project in `ProjectPaper.relevance_reason`) — both LLM calls in parallel across up to 10 workers, both also given titles+abstract excerpts from up to `REFERENCE_PAPERS_LIMIT` "My Collection" papers as calibration context
4. For accepted papers: fetch figure URL/caption from arXiv HTML `<figcaption>` and citation count, then generate summary + contributions via LLM. `PaperSummary` is one row per paper, shared by every project the paper is in, so `SUMMARY_WITH_FULL_CONTENT` deliberately gets no research interest; project-specific relevance lives in `ProjectPaper.relevance_reason`
5. After all workers finish: batch-enrich institutions via the Semantic Scholar graph API, fall back to LLM per-paper
6. LLM picks the "featured paper" for the crawl-complete email (`select_featured_paper_llm`)
7. Send notification email (if configured)

`use_arxiv` and `use_ss` flags (default True) can disable either source for manual crawls from the UI; the nightly script always uses both. Default `paper_limit=500`.

Semantic Scholar 429s are retried with backoff: recommendations after 15/45/135s (then give up for that crawl), citation lookups after 10/30s, and after 5 papers in a row exhaust their retries citation fetches pause for 15 min (`_CITATION_429_GIVE_UP`, `_CITATION_PAUSE_S`). The nightly backfill re-checks zero-citation papers at most once per `CITATION_BACKFILL_COOLDOWN_DAYS` (3) each, tracked in `Paper.citation_attempted_at`, and stops while fetches are paused. The shared cluster IP also means arXiv 429s are common; the nightly end-of-run retry covers them.

Previously-rejected candidates aren't re-screened on every crawl: `ScreenedPaper` caches each paper's relevance verdict under a `screening_hash` (`compute_screening_hash`, over `research_interest` + sorted My Collection titles). A paper found `is_relevant=False` under the current hash is skipped outright; changing the research interest or the Collection's contents changes the hash and makes everything eligible for re-screening again.

`_select_reference_papers` (used by the keyword-extraction and screening-context steps above, and by `generate_research_interest_from_collection` for the `RESEARCH_INTEREST_FROM_COLLECTION`/`RESEARCH_INTEREST_IMPROVE` prompts) fills by tag tier — `important` > `to_discuss` > `to_read` > untagged `related` — and only within whichever tier runs out of slots does it alternate newest/oldest by `collected_at` (newest, oldest, 2nd-newest, ...) rather than taking an arbitrary subset. Limit is `REFERENCE_PAPERS_LIMIT` (env var, default 15) — this same text block is repeated in every relevance-check call for a crawl, so it's cheap under prefix caching but still adds to KV-cache memory pressure; don't set it arbitrarily high.

### bioRxiv source (`app/biorxiv.py`)

Off unless asked for: `Project.use_biorxiv` (the "Also search bioRxiv" box under Nightly crawl, read by `scripts/nightly_crawl.py`; it is also the starting state of the Find Papers box) and `use_biorxiv=True` on `gather_candidates`/`run_crawl`/`start_crawl_async`. bioRxiv's API (`api.biorxiv.org/details`) has no keyword search, so `search_biorxiv` reads the whole date window, 30 entries per request, keeps first versions only, and keeps a paper when a keyword appears in its title or abstract as a phrase (`matched_keywords`: words in order and adjacent, whole words, a plural "s" and hyphens ignored, like arXiv's `abs:"phrase"`). A looser match (all words anywhere, cut to 6 letters) was tried on 30 days of bioRxiv with the projects' keywords on 2026-10-20 and matched mostly unrelated papers ("generator" matched "general"). A 5-day window is about 800 first versions in about 37 requests, 45 s; 30 days took 4 min (checked 2026-10-20). Do not hardcode the DOI prefix: bioRxiv moved from `10.1101` to `10.64898`.

A bioRxiv paper is a non-arXiv paper: `Paper.arxiv_id` is `web:biorxiv-<12 hex of sha256(doi)>` (no slash or dot, it goes into URLs and DOM ids), `Paper.source` is `biorxiv`, and `pdf_url` is its PDF on bioRxiv (`.full.pdf`, the DOI is in it). bioRxiv is on `OPEN_ACCESS_PDF_DOMAINS`: its TDM page (biorxiv.org/tdm, read 2026-10-20) welcomes machine analysis (authors consent at submission) as long as tools link back to bioRxiv instead of re-hosting, and its robots.txt allows `/content/` with `Crawl-delay: 7`, so every bioRxiv download (`fetch_biorxiv_full_text`, `fetch_pdf_text`) goes through `_biorxiv_gap`: one at a time, `BIORXIV_GAP_S` (7 s) after the previous one. `get_paper_content(arxiv_id, pdf_url)` reads a bioRxiv paper like an arXiv one, its HTML version (`pdf_url` without `.pdf`, cut to the `fulltext-view` div up to "Back to top" by `_biorxiv_html_to_text`), else its PDF; any other `web:` paper only from a PDF on an open-access domain. The HTML is cleaner than the PDF's text (line numbers, the license line of every page) but bioRxiv adds it only 4 to 6 days after posting (0 of 17 papers 1 to 3 days old, 12 of 12 at 6 and 10 days, checked 2026-10-20), so at the nightly crawl a fresh preprint is usually read from its PDF, after one extra request. The chat (`chat._fetch_text`) does the same. The deep check, summary, Re-screen, the nightly full-text re-read (arXiv and bioRxiv papers) and the chat all go through this. Figure 1 (image URL and its full caption, `biorxiv_figure`) also comes from the HTML version: at the crawl from the page the deep check read (`fetch_biorxiv_page` keeps each answer, also "no HTML yet", for an hour, so text and figure are one download), and for papers whose HTML came later from Pass 4 of the nightly backfill (bioRxiv papers without a figure, 4 to `BIORXIV_FIGURE_DAYS` (30) days after posting, at most once a day via `web_enrich_last_attempt_at`). All of 20 pages 6 to 14 days old had figure 1 and the images load when linked from another site (checked 2026-10-20). The email's figure download (`_fetch_figure_html`) uses the bioRxiv gap for a bioRxiv image. Citation count comes from Semantic Scholar by DOI, the corresponding author's institution from the API. It is skipped by everything keyed on an arXiv id: the post-crawl institution and page-count enrichment, the nightly Pass-3 web-paper backfill (`source != "biorxiv"`), Regenerate, and Pass 2 citations (`web:` ids). The email lists a bioRxiv row only when the crawl found something from it (`_email_source_rows`).

A paper added by hand from a bioRxiv link (`start_web_paper_add_async`) is turned into a full bioRxiv paper right away by `upgrade_biorxiv_paper`: DOI found in the link (the post id with prefix `10.1101`, else `10.64898`) and checked against the bioRxiv API, posting date and corresponding institution from it, the crawl's id `biorxiv_paper_id` (kept as it was if a crawl already stored the paper), latest `.full.pdf` as `pdf_url`, a summary from the full text (`write_summary`, shared with Regenerate), figure 1 and citations by DOI. Pass 2b of the nightly backfill does the same for older ones (`source` "web" with a `biorxiv.org` link, once a day each), which covers the papers added before bioRxiv was on the open-access list (they had a summary of the web page and "PDF metadata failed").

### Semantic Scholar sources beyond arXiv

Both Semantic Scholar sources take every My Collection paper Semantic Scholar can know as a seed (`collection_seeds`, `scholar_seed`): the arXiv id for an arXiv paper (`ArXiv:<id>`; Semantic Scholar does not know arXiv's own `10.48550` DOIs, checked 2026-10-20), else `DOI:<Paper.doi>`, else `Paper.semantic_scholar_id`. `resolve_scholar_ids` (run before the seeds are built, in `run_crawl` and the count step) fills these once per non-arXiv paper: the DOI in its URL (`doi_from_url`, no request), else a Semantic Scholar title match checked with `_titles_match` (stamped `scholar_lookup_at`, not after a 429). An unknown seed is ignored by the recommendations API, not an error. Results go through `_scholar_paper`: an arXiv paper by its arXiv id; a bioRxiv preprint (venue `bioRxiv`) as the same paper a bioRxiv keyword search would find (`biorxiv_paper_id`, its PDF); any other paper with a DOI and an abstract under `journal_paper_id` (`web:doi-<12 hex>`, `Paper.source` "journal"), with its open-access PDF as `pdf_url` only if that is on `OPEN_ACCESS_PDF_DOMAINS`, else `https://doi.org/<doi>` and screened on the abstract; nothing without a DOI or an abstract. With seeds from biology and spectroscopy collections most recommendations are journal papers (over 90 % of those in a 5-day window, checked 2026-10-20), with machine learning ones nearly all are on arXiv. Pass 3 of the nightly backfill only touches papers added by hand (`source` "web" or none), so nothing new is fetched from publishers. `Paper.doi` is stored whenever a source gives one: bioRxiv, Semantic Scholar's `externalIds.DOI`, the arXiv API's journal DOI (`result.doi`, when the authors gave one), a pasted page's `citation_doi` or its URL.

### Schema migrations

**No Alembic.** Migrations are plain `ALTER TABLE` statements in the `_migrations` list inside `create_app()` in `app/__init__.py`. They run on every startup and errors (column already exists) are silently swallowed. Add new columns there and to the model simultaneously.

### Paper classification

Two `ProjectPaper` fields together decide where a paper appears:
- `manual_tag` — only ever `None` (untagged → "New Papers", any crawl epoch, not just the latest) or `"related"` (in "My Collection")
- `paper_tags` — a JSON list of visible label badges (`"important"` / `"to_read"` / `"to_discuss"`) shown on a My Collection paper; toggling one on auto-sets `manual_tag = "related"` if it wasn't already set, but toggling the last one off does **not** move the paper back to New Papers — only the explicit "Move to New Papers" action clears both fields (`manual_tag = None`, `paper_tags = []`)
- `trashed_at` set → "Trash" (independent of the above)

Papers do **not** automatically move between sections — only explicit user actions do. Two of those act on more than one card: "Re-screen" in New Papers (`run_rescreen` in `crawl.py`, background thread, progress in `_rescreen_tasks` polled via `/projects/<slug>/rescreen-status`) runs the crawl's quick → deep screen on every New Papers paper against the current context and moves clear rejects to Trash (parse failures are kept; papers already judged relevant under the same `screening_hash` are skipped); and "Copy / Move" on a card (`/paper/<id>/transfer`) puts the paper into another project's My Collection, carrying tags and notes — Move also trashes it in the source project (not a hard delete, so the source's crawls won't re-add it). A transferred paper gets its target-project `relevance_reason` from `RELEVANCE_REASON` in a background thread.

`Paper.arxiv_id` doubles as the identity key for non-arXiv papers too: anything added via "paste any URL" that isn't an arXiv paper gets a synthetic `"web:..."`-prefixed id (also crawl finds: `web:biorxiv-<hash of DOI>` for bioRxiv, `web:doi-<hash of DOI>` for journal papers from Semantic Scholar; `Paper.is_arxiv` is `False` for all of these). Code that builds arXiv URLs/links from `arxiv_id`, or does citation/Scholar lookups keyed on it, needs to branch on `is_arxiv` first — this has broken UI targeting before (a literal `:` in the id collided with card-selector parsing).

### Add paper by URL

`start_web_paper_add_async` (`app/crawl.py`) is the entrypoint for "paste any URL" adds. It calls `extract_paper_info_from_url`, which is guarded by `_is_safe_external_url` (rejects non-http(s) schemes and private/loopback/link-local/reserved addresses — defense-in-depth behind `@login_required`, real SSRF shape since the URL is user-supplied). It then tries, in order, to resolve the paper to an arXiv id: an explicit arXiv link on the page, a DOI match, then (Step 2c) an arXiv title search (`arxiv.Search(query=f'ti:"{title}"')`, verified with `_titles_match`) — on any match it converts to a full arXiv paper via `start_collection_add_async`. If none match, it stays a synthetic `"web:..."`-id paper with only landing-page metadata (title/abstract/authors/DOI).

PDF downloads (for page count + institution extraction, `_enrich_web_paper_pdf_metadata`) only happen for hosts in `OPEN_ACCESS_PDF_DOMAINS` (arxiv.org, ar5iv.labs.arxiv.org, aclanthology.org, proceedings.mlr.press, openreview.net, proceedings.neurips.cc, openaccess.thecvf.com, ojs.aaai.org, ijcai.org, jmlr.org, biorxiv.org) — checked via `_is_open_access_pdf_domain`. Any other domain still gets its landing-page metadata read, just never a PDF fetch. All outbound requests (this pipeline and the arXiv/Scholar crawl paths) carry a `USER_AGENT` identifying the bot with a link to this repo (`app/utils.py`).

While `start_collection_add_async` / `start_web_paper_add_async` run, `crawl._enriching` holds the step each paper is on; `paper_card.html` shows it and polls `/project-paper/<id>/card`, which answers 204 while the step is unchanged and the whole card (HX-Retarget) once it changed or the work is done, so a freshly added paper fills in without a reload. The redraw waits while someone types in that card (`htmx:beforeSwap` in `base.html`). Tests: `tests/test_enrich_status.py`.

`ProjectPaper.origin` records how a paper came into a project (`crawl`, `manual` for a pasted link, `copied` / `moved` with `origin_project_id`), set wherever a link is created and shown on the card's date line. Older links were backfilled only where certain (`ORIGIN_BACKFILL_SQL` in `app/__init__.py`: a paper in exactly one project is `manual` if its `source` is manual/web, else `crawl`); the rest stay `None` and show the plain "Added" date.

`PaperSummary.content_source` records what a summary and its main contributions were made from (`full_text` / `web_page` / `abstract`, via `crawl.summary_source`, which mirrors the generators' own fallback); the card labels it, and `None` (summaries from before) shows nothing.

The nightly Pass-3 backfill (`backfill_missing_paper_metadata` in `crawl.py`) retries `"web:"` papers still missing `page_count`/`institutions`, capped at `MAX_WEB_ENRICH_ATTEMPTS` (3) total via `Paper.web_enrich_attempts`/`web_enrich_last_attempt_at` — not per night. Once capped, `paper_card.html` shows a "Retry" button + attempt count; `POST /paper/<id>/retry-web-enrich` resets the counter and re-runs `_enrich_one_web_paper` in a background thread (same fire-and-forget pattern as `start_web_paper_add_async`).

### Nightly crawl scheduling

Each `Project` has a `crawl_hour` (integer 1–4 UTC, or `None` = disabled). A CronJob (see `k8s/cronjob.yaml.example`) runs `scripts/nightly_crawl.py` on a schedule. The script:
- Filters to projects whose `crawl_hour` matches the run's scheduled UTC hour (`_scheduled_time`, from the pod name, so a late start still picks the right projects; trashed projects excluded), plus catch-ups: projects whose hour already passed today without a nightly (`cron`) CrawlLog (`_projects_for_run`). With `concurrencyPolicy: Forbid` a run still going at the next hour means that hour's run never starts (the controller only starts the latest missed schedule), which dropped the 03:00 projects on 2026-10-16
- Uses `project.saved_keywords` if set, otherwise generates via LLM and auto-saves them
- Date range: five days back to today UTC (`CRAWL_DAYS_BACK` env var overrides, default 5 — arXiv only makes Fri/weekend submissions searchable Mon/Tue 00:00 UTC, so the lag is up to ~4 days; dedup + screening cache absorb the overlap)

Toggle is in the project sidebar (UI POST to `/projects/<slug>/toggle-nightly-crawl`).

### Blueprints

| Blueprint     | Prefix      | File                     |
| ------------- | ----------- | ------------------------ |
| `projects_bp` | `/projects` | `app/routes/projects.py` (+ crawl-status in `app/routes/crawl.py`) |
| `papers_bp`   | (none)      | `app/routes/papers.py`   |
| `auth_bp`     | (none)      | `app/auth.py`            |

`/login` is rate-limited (Flask-Limiter, 20/hour + 5/minute per IP) — nothing else in the app is.

### Templates

- `base.html` — layout, nav, CSS tokens
- `projects.html` — project list; the star on a card keeps that project on top, stored per browser in `localStorage` (`starred-projects`), not in the DB, since everyone shares one site login
- `project_detail.html` — the main working page; large file with inline `<script>` blocks for HTMX orchestration of the crawl count → crawl start flow
- `partials/paper_list.html` — the three sections (`section_new.html`, `section_col.html`, `section_trash.html`). Sorting and the My Collection tag filter run on the server (`app/paper_lists.py`); the page keeps the state in JS and sends it with every htmx request (`htmx:configRequest`), and each list renders `PAGE_SIZE` (10) cards plus a sentinel that loads the next page when it scrolls into view (`/projects/<slug>/papers/more`). New Papers is a one-card carousel (buttons, arrow keys outside text fields, swipe; script at the end of `project_detail.html`): all loaded cards stay in the DOM, all but one hidden, and its sentinel fires on a custom `loadmore` event when the carousel is two cards from the end, with `data-left` giving the counter its total; a re-rendered section (sort, crawl) starts again at the first card (`data-carousel-init` marks a list already seen). A card action that moves a paper (collect, move back, trash, restore, Copy / Move's Move) removes that card and re-renders only the destination section plus the header counts out of band (`_moved_response` in `routes/papers.py`), so the reader keeps their place. "Date added" in My Collection means `collected_at`. Responses are compressed (Flask-Compress): a big project page was ~5 MB raw.
- `partials/paper_card.html` — single paper with tag buttons, notes, notify bell, and the chat slot (`partials/chat_slot.html` → `chat_panel.html`, `chat_head.html`, `chat_log.html`, `chat_message.html`, `chat_live.html`, `chat_context.html`, `chat_notice.html`). Chat drafts are kept in `sessionStorage`; answers render markdown (raw HTML escaped) with math drawn by KaTeX from jsdelivr
- `partials/crawl_banner.html` — "Nightly crawl is off" banner with a setup checklist; redraws itself on the `crawlSetupChanged` event (HX-Trigger from the email/toggle/card-action routes, fired by hand after the page's `fetch()` saves of research interest, keywords and emails)
- `partials/crawl_status.html` — live crawl progress, auto-polls via HTMX

### Logs

`LOGS_DIR` (default `logs/` locally): `app.log` (web server requests, LLM calls from manual UI crawls) and `crawl.log` (nightly crawl runs only). Set `LOG_LLM=1` to additionally log every LLM prompt + response at INFO level (dev only — do not enable in a production deployment).

## Deployment

`gunicorn` runs with `--workers 1 --threads 4 --timeout 120`. **Do not increase `--workers` beyond 1** — the entire app depends on being a single process:
- `_arxiv_lock` / `_arxiv_waiters` (in `app/utils.py`, re-exported from `crawl.py`) enforce a global arXiv rate limit (one request in flight at a time, ≥5s gap) covering the search API *and* ar5iv/arxiv.org HTML+PDF fetches — all outbound arXiv-domain requests go through it. Each worker process would get its own copy of these; they can't see each other, so concurrent arXiv requests from different workers would bypass the lock entirely.
- `_crawl_current_papers`, `_arxiv_count_tasks`, `crawl._enriching`, and the paper chat's `_live`/`_running` (in-memory state) — UI polling could hit a different worker than the one running the task, returning stale or missing data.
- SQLite write contention — WAL mode helps reads, but concurrent writers across processes still cause `database is locked` errors.

The 4 threads are sufficient: they share all in-process state correctly and handle concurrent web requests during long LLM/crawl operations.

Every new SQLite connection gets `PRAGMA busy_timeout=30000` set via a SQLAlchemy `connect` event listener in `create_app()` (not the connection string — `busy_timeout` is per-connection and doesn't survive there) — this is what makes `database is locked` rare in practice despite concurrent threads, by having a writer wait up to 30s for a lock instead of failing immediately.

The Docker image is built `FROM --platform=linux/amd64` explicitly (see `Dockerfile`) since the cluster is amd64 — building locally on Apple Silicon works but runs under emulation (slower) unless you override the platform for local-only testing.

See `README.md` for the full deploy walkthrough and `k8s/*.yaml.example` for manifest templates.

## Backups

`scripts/backup_db.py` runs on its own daily CronJob (`k8s/backup-cronjob.yaml.example`), separate from the crawl CronJob — deliberately, since the crawl script fires up to 4x/day (once per possible `crawl_hour`) and folding backups into it would need its own once-per-day guard anyway. It snapshots the live DB via SQLite's [online backup API](https://www.sqlite.org/backup.html) (`sqlite3.Connection.backup`), not a plain file copy — the app runs `PRAGMA journal_mode=WAL`, and a raw `cp` of the `.db` file can miss data still sitting in the `-wal` file or catch it mid-checkpoint. The source is opened `mode=ro`, so the backup process can never write to the live DB. Snapshots land on a separate PVC (`k8s/backup-pvc.yaml`) so a lost/corrupted main volume doesn't take the backups with it, are verified (`PRAGMA integrity_check` + row count) right after writing, and are named `related_work_YYYYMMDD.db` — a second run on the same day is a no-op. Pruned once older than `BACKUP_RETENTION_DAYS` (default 30).

## Manual verification scripts

- `scripts/check_email.py` — sends a real test email via your configured notifier. `TEST_NOTIFY_EMAIL=you@example.com python scripts/check_email.py`
- `scripts/check_figure_fetch.py` — reports what figure (if any) would be extracted for given arXiv IDs, or a few papers from your DB if none are passed
- `scripts/check_prompt_rendering.py` — renders every collection-aware prompt template (`KEYWORD_EXTRACTION`, `RELEVANCE_QUICK`, `RESEARCH_INTEREST_FROM_COLLECTION`, `RESEARCH_INTEREST_IMPROVE`) through the real `_select_reference_papers`/`_collection_context_text` code against a synthetic collection, and prints the fully-substituted text for visual review. No LLM call, no network, no DB.
- `scripts/backfill_relevance_reasons.py` — one-off: fills `ProjectPaper.relevance_reason` for papers from before it existed, only in projects with the nightly crawl on and only for papers not in Trash (`RELEVANCE_REASON`, title + abstract, one LLM call per paper). `--dry-run` prints counts; re-runs skip filled papers.
