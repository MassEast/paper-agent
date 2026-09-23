# AGENTS.md

Conventions for anyone (human or AI agent) working on this codebase.

## What this is

A Flask-based literature review agent. It crawls arXiv and Semantic Scholar for new papers, screens them with an LLM against your research interest, generates summaries, and sends email notifications. Multiple research projects are tracked independently in one SQLite DB.

## Dev commands

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
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

### Crawl pipeline (`app/crawl.py`)

`run_crawl(project_id, date_from, date_to, ...)` is the main entrypoint, always called via `start_crawl_async(...)` which wraps it in a daemon thread. Flow:

1. Extract/reuse keywords from research interest via LLM — if `project.saved_keywords` is set, uses those directly (no LLM call). Auto-saves keywords after first successful generation. The LLM prompt also gets titles+abstract excerpts from up to `REFERENCE_PAPERS_LIMIT` "My Collection" papers (`_select_reference_papers`) as domain-vocabulary context.
2. Search arXiv by keyword (`abs:"phrase"` for multi-word, up to 200 results/keyword). Separately, if the project has any "My Collection" papers, seed Semantic Scholar's `/recommendations` API with their arXiv IDs (`get_semantic_scholar_recommendations`) — **not** keyword-driven; skipped entirely if the Collection is empty. Uses all Collection papers, not the capped selection below.
3. For each candidate paper: quick relevance check (abstract only) → deep check (full PDF content) — both LLM calls in parallel across up to 10 workers, both also given titles+abstract excerpts from up to `REFERENCE_PAPERS_LIMIT` "My Collection" papers as calibration context
4. For accepted papers: fetch figure URL/caption from arXiv HTML `<figcaption>` and citation count, then generate summary + contributions via LLM
5. After all workers finish: batch-enrich institutions via the Semantic Scholar graph API, fall back to LLM per-paper
6. LLM picks the "featured paper" for the crawl-complete email (`select_featured_paper_llm`)
7. Send notification email (if configured)

`use_arxiv` and `use_ss` flags (default True) can disable either source for manual crawls from the UI; the nightly script always uses both. Default `paper_limit=500`.

Previously-rejected candidates aren't re-screened on every crawl: `ScreenedPaper` caches each paper's relevance verdict under a `screening_hash` (`compute_screening_hash`, over `research_interest` + sorted My Collection titles). A paper found `is_relevant=False` under the current hash is skipped outright; changing the research interest or the Collection's contents changes the hash and makes everything eligible for re-screening again.

`_select_reference_papers` (used by the keyword-extraction and screening-context steps above, and by `generate_research_interest_from_collection` for the `RESEARCH_INTEREST_FROM_COLLECTION`/`RESEARCH_INTEREST_IMPROVE` prompts) fills by tag tier — `important` > `to_discuss` > `to_read` > untagged `related` — and only within whichever tier runs out of slots does it alternate newest/oldest by `collected_at` (newest, oldest, 2nd-newest, ...) rather than taking an arbitrary subset. Limit is `REFERENCE_PAPERS_LIMIT` (env var, default 15) — this same text block is repeated in every relevance-check call for a crawl, so it's cheap under prefix caching but still adds to KV-cache memory pressure; don't set it arbitrarily high.

### Schema migrations

**No Alembic.** Migrations are plain `ALTER TABLE` statements in the `_migrations` list inside `create_app()` in `app/__init__.py`. They run on every startup and errors (column already exists) are silently swallowed. Add new columns there and to the model simultaneously.

### Paper classification

Two `ProjectPaper` fields together decide where a paper appears:
- `manual_tag` — only ever `None` (untagged → "New Papers", any crawl epoch, not just the latest) or `"related"` (in "My Collection")
- `paper_tags` — a JSON list of visible label badges (`"important"` / `"to_read"` / `"to_discuss"`) shown on a My Collection paper; toggling one on auto-sets `manual_tag = "related"` if it wasn't already set, but toggling the last one off does **not** move the paper back to New Papers — only the explicit "Move to New Papers" action clears both fields (`manual_tag = None`, `paper_tags = []`)
- `trashed_at` set → "Trash" (independent of the above)

Papers do **not** automatically move between sections — only explicit user actions do.

`Paper.arxiv_id` doubles as the identity key for non-arXiv papers too: anything added via "paste any URL" that isn't an arXiv paper gets a synthetic `"web:..."`-prefixed id (`Paper.is_arxiv` is `False` for these). Code that builds arXiv URLs/links from `arxiv_id`, or does citation/Scholar lookups keyed on it, needs to branch on `is_arxiv` first — this has broken UI targeting before (a literal `:` in the id collided with card-selector parsing).

### Nightly crawl scheduling

Each `Project` has a `crawl_hour` (integer 1–4 UTC, or `None` = disabled). A CronJob (see `k8s/cronjob.yaml.example`) runs `scripts/nightly_crawl.py` on a schedule. The script:
- Filters to projects whose `crawl_hour` matches `datetime.now(UTC).hour` (trashed projects excluded)
- Uses `project.saved_keywords` if set, otherwise generates via LLM and auto-saves them
- Date range: yesterday UTC to today UTC (`CRAWL_DAYS_BACK` env var overrides, default 1)

Toggle is in the project sidebar (UI POST to `/projects/<slug>/toggle-nightly-crawl`).

### Blueprints

| Blueprint     | Prefix      | File                     |
| ------------- | ----------- | ------------------------ |
| `projects_bp` | `/projects` | `app/routes/projects.py` |
| `papers_bp`   | (none)      | `app/routes/papers.py`   |
| `crawl_bp`    | (none)      | `app/routes/crawl.py`    |
| `auth_bp`     | (none)      | `app/auth.py`            |

`/login` is rate-limited (Flask-Limiter, 20/hour + 5/minute per IP) — nothing else in the app is.

### Templates

- `base.html` — layout, nav, CSS tokens
- `project_detail.html` — the main working page; large file with inline `<script>` blocks for HTMX orchestration of the crawl count → crawl start flow
- `partials/paper_list.html` — HTMX-swappable paper list (New Papers + My Collection + Trash)
- `partials/paper_card.html` — single paper with tag buttons, notes, notify bell
- `partials/crawl_status.html` — live crawl progress, auto-polls via HTMX

### Logs

`LOGS_DIR` (default `logs/` locally): `app.log` (web server requests, LLM calls from manual UI crawls) and `crawl.log` (nightly crawl runs only). Set `LOG_LLM=1` to additionally log every LLM prompt + response at INFO level (dev only — do not enable in a production deployment).

## Deployment

`gunicorn` runs with `--workers 1 --threads 4 --timeout 120`. **Do not increase `--workers` beyond 1** — the entire app depends on being a single process:
- `_arxiv_lock` / `_arxiv_waiters` in `crawl.py` enforce a global arXiv rate limit (1 request / 3 s, single connection). Each worker process would get its own copy of these; they can't see each other, so concurrent arXiv requests from different workers would bypass the lock entirely.
- `_crawl_current_papers`, `_arxiv_count_tasks` (in-memory crawl/count state) — UI polling could hit a different worker than the one running the crawl, returning stale or missing data.
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
