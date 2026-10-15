import json
import os
import random
import threading
import uuid
from datetime import date, timedelta, datetime

from flask import current_app, make_response, render_template, request, redirect, url_for, jsonify

from app import db
from app.auth import login_required
from app.usage import counted, project_usage, usage_by_project, KINDS as USAGE_KINDS
from app.models import Project, ProjectPaper, Paper, CrawlLog, ScreenedPaper
from app import paper_lists
from app.routes import projects_bp
from app.crawl import (
    start_crawl_async,
    extract_paper_info_from_url,
    generate_research_interest_from_collection,
    extract_keywords_with_usage,
    gather_candidates,
    collection_seeds,
    _collection_tag_tier,
    arxiv_queue_depth,
    start_collection_add_async,
    start_web_paper_add_async,
    start_rescreen_async,
    rescreen_state,
    REFERENCE_PAPERS_LIMIT,
)
from app.llm import LLMUnavailableError, MODELS as LLM_MODELS
from app.models import slugify, compute_screening_hash
from app.utils import base_arxiv_id

# In-memory store for async arXiv count tasks (dev only — single process)
_arxiv_count_tasks: dict[str, dict] = {}


@projects_bp.route("")
@login_required
def index():
    all_projects = Project.query.order_by(Project.created_at.desc()).all()
    projects = [p for p in all_projects if not p.trashed_at]
    trashed_projects = [p for p in all_projects if p.trashed_at]
    return render_template(
        "projects.html", projects=projects, trashed_projects=trashed_projects, llm_model=LLM_MODELS[0],
        usage_totals=usage_by_project(),
        cost_comparison_model=os.environ.get("LLM_COST_COMPARISON_MODEL", "Claude Sonnet 4.6"),
    )


@projects_bp.route("", methods=["POST"])
@login_required
def create():
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(url_for("projects.index"))

    slug = slugify(name)
    counter = 1
    base_slug = slug
    while Project.query.filter_by(slug=slug).first():
        slug = f"{base_slug}-{counter}"
        counter += 1

    project = Project(name=name, slug=slug, crawl_hour=None)
    db.session.add(project)
    db.session.commit()

    return redirect(url_for("projects.detail", slug=slug))


@projects_bp.route("/<slug>/trash", methods=["POST"])
@login_required
def trash_project(slug):
    project = Project.query.filter_by(slug=slug).first_or_404()
    project.trashed_at = datetime.utcnow()
    db.session.commit()
    return redirect(url_for("projects.index"))


@projects_bp.route("/<slug>/restore", methods=["POST"])
@login_required
def restore_project(slug):
    project = Project.query.filter_by(slug=slug).first_or_404()
    project.trashed_at = None
    db.session.commit()
    return redirect(url_for("projects.index"))


@projects_bp.route("/<slug>")
@login_required
def detail(slug):
    """Main project page: splits papers into New/Curated/Trashed, plus crawl history and
    the prompt-preview panel (for the "show me what gets sent to the LLM" UI)."""
    project = Project.query.filter_by(slug=slug).first_or_404()
    if project.trashed_at:
        return redirect(url_for("projects.index"))

    state = paper_lists.list_state(request.args)
    sections = paper_lists.sections(project, state)
    papers = ProjectPaper.query.filter_by(project_id=project.id).all()  # every paper: calendar dots

    recent_crawl = (
        CrawlLog.query.filter_by(project_id=project.id).order_by(CrawlLog.started_at.desc()).first()
    )

    # Context-hash epochs: first date each unique screening context was used
    from sqlalchemy import func as _func
    _epoch_rows = (
        db.session.query(
            ScreenedPaper.screening_hash,
            _func.min(ScreenedPaper.screened_at).label("first_seen"),
        )
        .filter(ScreenedPaper.project_id == project.id)
        .group_by(ScreenedPaper.screening_hash)
        .all()
    )
    context_epochs = sorted(
        [{"date": r.first_seen.date().isoformat(), "hash": r.screening_hash[:6]}
         for r in _epoch_rows if r.first_seen],
        key=lambda e: e["date"],
    )

    # Pre-serialise crawl logs server-side so the template avoids bare {{ }} in JS object literals
    # (Prettier reformats those, breaking the Jinja expressions).
    crawl_logs_js = [
        {
            "started_at": str(log.started_at),
            "status": log.status or "",
            "papers_found": log.papers_found or 0,
            "papers_checked": log.papers_checked or 0,
            "papers_added": log.papers_added or 0,
            "date_from": log.date_from.isoformat() if log.date_from else "",
            "date_to": log.date_to.isoformat() if log.date_to else "",
            "sources_used": log.sources_used or "arxiv,ss",
            "triggered_by": log.triggered_by or "manual",
        }
        for log in project.crawl_logs
    ]

    import app.prompts as _prompts
    return render_template(
        "project_detail.html",
        project=project,
        papers=papers,
        new_papers=sections["new"],
        curated_papers=sections["col"],
        s=sections,
        state=state,
        page_size=paper_lists.PAGE_SIZE,
        recent_crawl=recent_crawl,
        search_query=state["search"],
        context_epochs=context_epochs,
        crawl_logs_js=crawl_logs_js,
        prompt_relevance_quick=_prompts.RELEVANCE_QUICK,
        prompt_relevance_deep=_prompts.RELEVANCE_DEEP,
        prompt_summary=_prompts.SUMMARY_WITH_FULL_CONTENT,
        reference_papers_limit=REFERENCE_PAPERS_LIMIT,
        rescreen=rescreen_state(project.id),
        usage=project_usage(project.id),
        usage_kinds=USAGE_KINDS,
        prompt_keyword_extraction=_prompts.KEYWORD_EXTRACTION,
        prompt_research_interest=_prompts.RESEARCH_INTEREST_IMPROVE,
        llm_model=LLM_MODELS[0],
        llm_api_base=os.environ.get("LLM_API_BASE", ""),
        cost_comparison_model=os.environ.get("LLM_COST_COMPARISON_MODEL", "Claude Sonnet 4.6"),
        notification_sender=os.environ.get("RESEND_FROM", "Related Work Agent <notifications@example.com>"),
    )


@projects_bp.route("/<slug>/research-interest", methods=["POST"])
@login_required
def update_research_interest(slug):
    project = Project.query.filter_by(slug=slug).first_or_404()
    project.research_interest = request.form.get("research_interest", "").strip()
    db.session.commit()
    if request.headers.get("X-Requested-With") == "fetch":
        return ("", 204)
    return redirect(url_for("projects.detail", slug=slug))


@projects_bp.route("/<slug>/collection-add", methods=["POST"])
@login_required
def add_to_collection(slug):
    """Bypass-add a paper directly to My Collection — no LLM screening, full enrichment in background.

    Accepts arXiv URLs (full enrichment) and any other URL (web-paper enrichment, with
    automatic arXiv detection if the page links to a matching arXiv paper).
    """
    from flask import current_app
    import re as _re
    import hashlib as _hashlib

    project = Project.query.filter_by(slug=slug).first_or_404()
    url = request.form.get("url", "").strip()
    if not url:
        return redirect(url_for("projects.detail", slug=slug))

    arxiv_match = _re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)

    _app = current_app._get_current_object()

    if arxiv_match:
        # --- arXiv URL path ---
        arxiv_id = arxiv_match.group(1)
        existing_paper = Paper.query.filter_by(arxiv_id=arxiv_id).first()
        if existing_paper:
            pp = ProjectPaper.query.filter_by(project_id=project.id, paper_id=existing_paper.id).first()
            if pp:
                if pp.trashed_at:
                    # Restore from trash — treat like a fresh add
                    pp.trashed_at = None
                    pp.manual_tag = "related"
                    pp.paper_tags = '["important"]'
                    pp.collected_at = datetime.utcnow()
                    db.session.commit()
                    if not existing_paper.abstract:
                        start_collection_add_async(project.id, arxiv_id, existing_paper.id, _app)
                    return redirect(url_for("projects.detail", slug=slug))
                # Active in collection already
                return redirect(url_for("projects.detail", slug=slug, add_info="already_exists"))
            # Paper exists in DB but not linked to this project
            pp = ProjectPaper(project_id=project.id, paper_id=existing_paper.id, origin="manual",
                              manual_tag="related", paper_tags='["important"]', collected_at=datetime.utcnow())
            db.session.add(pp)
            db.session.commit()
            if not existing_paper.abstract:
                start_collection_add_async(project.id, arxiv_id, existing_paper.id, _app)
            return redirect(url_for("projects.detail", slug=slug))

        paper = Paper(arxiv_id=arxiv_id, title=f"arXiv:{arxiv_id}", source="manual", authors="[]")
        db.session.add(paper)
        db.session.flush()
        pp = ProjectPaper(project_id=project.id, paper_id=paper.id, origin="manual",
                          manual_tag="related", paper_tags='["important"]', collected_at=datetime.utcnow())
        db.session.add(pp)
        db.session.commit()
        start_collection_add_async(project.id, arxiv_id, paper.id, _app)

    else:
        # --- Non-arXiv URL path ---
        synthetic_id = f"web:{_hashlib.sha256(url.encode()).hexdigest()[:12]}"
        existing_paper = Paper.query.filter_by(arxiv_id=synthetic_id).first()
        if existing_paper:
            pp = ProjectPaper.query.filter_by(project_id=project.id, paper_id=existing_paper.id).first()
            if pp:
                if pp.trashed_at:
                    pp.trashed_at = None
                    pp.manual_tag = "related"
                    pp.paper_tags = '["important"]'
                    pp.collected_at = datetime.utcnow()
                    db.session.commit()
                    return redirect(url_for("projects.detail", slug=slug))
                return redirect(url_for("projects.detail", slug=slug, add_info="already_exists"))
            pp = ProjectPaper(project_id=project.id, paper_id=existing_paper.id, origin="manual",
                              manual_tag="related", paper_tags='["important"]', collected_at=datetime.utcnow())
            db.session.add(pp)
            db.session.commit()
            return redirect(url_for("projects.detail", slug=slug))

        paper = Paper(arxiv_id=synthetic_id, title="Fetching…", pdf_url=url, source="web", authors="[]")
        db.session.add(paper)
        db.session.flush()
        pp = ProjectPaper(project_id=project.id, paper_id=paper.id, origin="manual",
                          manual_tag="related", paper_tags='["important"]', collected_at=datetime.utcnow())
        db.session.add(pp)
        db.session.commit()
        start_web_paper_add_async(project.id, url, paper.id, _app)

    return redirect(url_for("projects.detail", slug=slug))


@projects_bp.route("/extract-from-url", methods=["GET"])
@login_required
def extract_from_url():
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    info = extract_paper_info_from_url(url)
    return jsonify(info)


@projects_bp.route("/<slug>/generate-research-interest", methods=["POST"])
@login_required
def generate_research_interest(slug):
    project = Project.query.filter_by(slug=slug).first_or_404()
    collection_pps = ProjectPaper.query.filter_by(project_id=project.id).filter(
        ProjectPaper.manual_tag.isnot(None),
        ProjectPaper.trashed_at.is_(None),
    ).all()

    existing_text = ""
    if request.is_json and request.json:
        existing_text = (request.json.get("existing_text") or "").strip()

    if not collection_pps and not existing_text:
        return jsonify({"error": "Add papers to My Collection or write a research interest first"}), 400

    try:
        with counted(project.id, "research_interest"):
            generated = generate_research_interest_from_collection(collection_pps, existing_text=existing_text)
        return jsonify({"research_interest": generated})
    except LLMUnavailableError as e:
        return jsonify({"error": f"LLM unavailable: {str(e)}"}), 503


@projects_bp.route("/<slug>/arxiv-count", methods=["POST"])
@login_required
def arxiv_count(slug):
    """Start async arXiv count. Returns immediately with a task_id to poll."""
    keywords_json = request.form.get("keywords_json")
    if not keywords_json:
        return jsonify({"error": "No keywords provided"}), 400

    try:
        keywords = [k.strip() for k in json.loads(keywords_json) if str(k).strip()]
    except (json.JSONDecodeError, ValueError):
        return jsonify({"error": "Invalid keywords"}), 400

    date_from_str = request.form.get("date_from")
    date_to_str = request.form.get("date_to")
    date_from = date.fromisoformat(date_from_str) if date_from_str else (date.today() - timedelta(days=7))
    date_to = date.fromisoformat(date_to_str) if date_to_str else date.today()

    use_arxiv = request.form.get("use_arxiv", "1") != "0"
    use_scholar = request.form.get("use_scholar", "1") != "0"
    use_citations = request.form.get("use_citations", "1") != "0"

    task_id = uuid.uuid4().hex[:12]
    _arxiv_count_tasks[task_id] = {"status": "running"}

    # My Collection (seeds for recommendations and citations) and existing paper IDs before the thread starts
    project_obj = Project.query.filter_by(slug=slug).first()
    seeds = []
    existing_arxiv_ids = set()
    if project_obj:
        for pp in project_obj.papers:
            if pp.paper and pp.paper.arxiv_id:
                existing_arxiv_ids.add(base_arxiv_id(pp.paper.arxiv_id))
        seeds = collection_seeds(sorted([pp for pp in project_obj.papers if not pp.trashed_at and pp.manual_tag],
                                        key=_collection_tag_tier))

    # Papers rejected earlier under the *current* screening context — re-screening them is an instant
    # cache hit (no LLM call), so the UI reports them separately from genuinely new papers.
    rejected_base_ids = set()
    if project_obj:
        _collection = [pp for pp in project_obj.papers if pp.manual_tag and not pp.trashed_at and pp.paper]
        _hash = compute_screening_hash(project_obj.research_interest or "", _collection)
        rejected_base_ids = {
            sp.arxiv_id for sp in ScreenedPaper.query.filter_by(
                project_id=project_obj.id, screening_hash=_hash, is_relevant=False
            ).all()
        }

    search_warnings: list[str] = []

    def _run():
        try:
            all_papers = gather_candidates(keywords, seeds, date_from, date_to, use_arxiv=use_arxiv,
                                           use_scholar=use_scholar, use_citations=use_citations,
                                           warnings=search_warnings, status=_arxiv_count_tasks[task_id])
            # Subtract papers already in this project (any state: new/curated/trash)
            already_known = sum(1 for p in all_papers if base_arxiv_id(p["arxiv_id"]) in existing_arxiv_ids)
            new_papers = [p for p in all_papers if base_arxiv_id(p["arxiv_id"]) not in existing_arxiv_ids]
            previously_rejected = sum(1 for p in new_papers if base_arxiv_id(p["arxiv_id"]) in rejected_base_ids)
            new_arxiv = sum(1 for p in new_papers if p.get("source") != "semantic_scholar")
            new_scholar = sum(1 for p in new_papers if p.get("source") == "semantic_scholar")
            new_citing = sum(1 for p in new_papers if any(v["via"] == "cites" for v in p["found_via"]))
            _arxiv_count_tasks[task_id] = {
                "status": "done",
                "count": len(new_papers),
                "papers": new_papers,
                "arxiv_count": new_arxiv,
                "scholar_count": new_scholar,
                "citing_count": new_citing,
                "already_known": already_known,
                "previously_rejected": previously_rejected,
                "warnings": search_warnings,
            }
        except Exception as e:
            _arxiv_count_tasks[task_id] = {"status": "error", "error": str(e)[:200]}

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"task_id": task_id})


@projects_bp.route("/<slug>/arxiv-count-status/<task_id>", methods=["GET"])
@login_required
def arxiv_count_status(slug, task_id):
    result = _arxiv_count_tasks.get(task_id)
    if result is None:
        return jsonify({"status": "error", "error": "Task not found"}), 404
    return jsonify({**result, "arxiv_waiters": arxiv_queue_depth()})


@projects_bp.route("/<slug>/crawl", methods=["POST"])
@login_required
def trigger_crawl(slug):
    """Start a manual crawl. Reuses cached arXiv/Scholar results from a prior /arxiv-count
    call if arxiv_task_id is provided, so the count-preview step doesn't get re-searched."""
    project = Project.query.filter_by(slug=slug).first_or_404()

    # Guard against a double-click (or a second browser tab) starting a
    # duplicate crawl for the same project — two concurrent crawls race to
    # insert the same arxiv_id into the shared `papers` table, and the loser
    # hits an IntegrityError. See docs/PROGRESS.md 2026-09-22.
    existing_running = CrawlLog.query.filter_by(project_id=project.id, status="running").first()
    if existing_running:
        if request.headers.get("HX-Request"):
            return render_template("partials/crawl_status.html", project=project, crawl=existing_running)
        return redirect(url_for("projects.detail", slug=slug))

    keywords_override = None
    keywords_json = request.form.get("keywords_json")
    if keywords_json:
        try:
            keywords_override = [k.strip() for k in json.loads(keywords_json) if str(k).strip()]
        except json.JSONDecodeError:
            keywords_override = None

    has_collection = ProjectPaper.query.filter_by(project_id=project.id).filter(
        ProjectPaper.manual_tag.isnot(None), ProjectPaper.trashed_at.is_(None)
    ).first() is not None

    if not keywords_override and not project.research_interest and not has_collection:
        error_msg = "Please set a research interest or add papers to My Collection first."
        if request.headers.get("HX-Request"):
            return f'<div class="mt-4 p-4 bg-red-900/30 border border-red-700 rounded-lg"><span class="text-red-400">Cannot crawl: {error_msg}</span></div>'
        return redirect(url_for("projects.detail", slug=slug, error=error_msg))

    date_from_str = request.form.get("date_from")
    date_to_str = request.form.get("date_to")
    paper_limit = request.form.get("paper_limit", type=int, default=50)

    date_from = (
        date.fromisoformat(date_from_str) if date_from_str else (date.today() - timedelta(days=7))
    )
    date_to = date.fromisoformat(date_to_str) if date_to_str else date.today()

    from flask import current_app

    # Retrieve cached arXiv papers from the count step to avoid re-searching
    cached_papers = None
    arxiv_task_id = request.form.get("arxiv_task_id")
    if arxiv_task_id:
        task = _arxiv_count_tasks.get(arxiv_task_id, {})
        if task.get("status") == "done" and "papers" in task:
            cached_papers = task["papers"]

    # Persist the selected keywords so nightly crawl re-uses them
    if keywords_override:
        project.saved_keywords = json.dumps(keywords_override)
        db.session.commit()

    use_arxiv = request.form.get("use_arxiv", "1") != "0"
    use_scholar = request.form.get("use_scholar", "1") != "0"
    use_citations = request.form.get("use_citations", "1") != "0"

    _thread, crawl_log_id = start_crawl_async(
        project.id,
        date_from,
        date_to,
        triggered_by="manual",
        app=current_app._get_current_object(),
        paper_limit=paper_limit,
        keywords_override=keywords_override,
        cached_papers=cached_papers,
        use_arxiv=use_arxiv,
        use_scholar=use_scholar,
        use_citations=use_citations,
    )

    if request.headers.get("HX-Request"):
        crawl = CrawlLog.query.get(crawl_log_id)
        return render_template("partials/crawl_status.html", project=project, crawl=crawl)

    return redirect(url_for("projects.detail", slug=slug))


@projects_bp.route("/<slug>/rescreen", methods=["POST"])
@login_required
def rescreen(slug):
    """Re-check all New Papers against the current research interest (crawl.run_rescreen)."""
    project = Project.query.filter_by(slug=slug).first_or_404()
    if not (project.research_interest or "").strip():
        return ("Set a research interest first", 400)
    start_rescreen_async(project.id)  # no-op if one is already running; the status shows it
    return render_template("partials/rescreen_status.html", project=project, state=rescreen_state(project.id))


@projects_bp.route("/<slug>/rescreen-status")
@login_required
def rescreen_status(slug):
    project = Project.query.filter_by(slug=slug).first_or_404()
    state = rescreen_state(project.id)
    # Refresh the paper list exactly once, on the first poll that sees the re-screen finished
    refresh = bool(state) and state["status"] != "running" and not state.get("list_refreshed")
    if refresh:
        state["list_refreshed"] = True
    return render_template("partials/rescreen_status.html", project=project, state=state, refresh=refresh)


@projects_bp.route("/<slug>/papers")
@login_required
def papers_partial(slug):
    """All three sections (first page each) — after a crawl, a re-screen, or a search."""
    project = Project.query.filter_by(slug=slug).first_or_404()
    state = paper_lists.list_state(request.args)
    return render_template("partials/paper_list.html", project=project, s=paper_lists.sections(project, state),
                           state=state, page_size=paper_lists.PAGE_SIZE)


@projects_bp.route("/<slug>/papers/section")
@login_required
def papers_section(slug):
    """One section re-rendered from its first page — after the sort or the tag filter changed."""
    project = Project.query.filter_by(slug=slug).first_or_404()
    section = request.args.get("section")
    if section not in paper_lists.SECTIONS:
        return ("unknown section", 400)
    state = paper_lists.list_state(request.args)
    return render_template(f"partials/section_{section}.html", project=project, oob=False,
                           s=paper_lists.sections(project, state), state=state, page_size=paper_lists.PAGE_SIZE)


@projects_bp.route("/<slug>/papers/more")
@login_required
def papers_more(slug):
    """The next page of one section's list (requested by the sentinel at the end of the list)."""
    project = Project.query.filter_by(slug=slug).first_or_404()
    section = request.args.get("section")
    offset = request.args.get("offset", type=int)
    if section not in paper_lists.SECTIONS or offset is None or offset < 0:
        return ("bad section or offset", 400)
    state = paper_lists.list_state(request.args)
    s = paper_lists.sections(project, state)
    return render_template("partials/paper_page.html", project=project, section=section, papers=s[section],
                           offset=offset, page_size=paper_lists.PAGE_SIZE)


@projects_bp.route("/<slug>/generate-keywords", methods=["POST"])
@login_required
def generate_keywords(slug):
    """Generate search keywords via LLM from research interest + My Collection."""
    project = Project.query.filter_by(slug=slug).first_or_404()
    collection_pps = ProjectPaper.query.filter_by(project_id=project.id).filter(
        ProjectPaper.manual_tag.isnot(None), ProjectPaper.trashed_at.is_(None)
    ).all()

    if not project.research_interest and not collection_pps:
        return jsonify({"error": "Please set a research interest or add papers to My Collection first."}), 400

    try:
        with counted(project.id, "keywords"):
            keywords, _ = extract_keywords_with_usage(project.research_interest, collection_pps)
    except LLMUnavailableError as e:
        return jsonify({"error": f"LLM unavailable: {str(e)}"}), 503

    if not keywords:
        return jsonify({"error": "LLM returned no keywords — try again."}), 502

    return jsonify({"keywords": keywords})


@projects_bp.route("/<slug>/save-keywords", methods=["POST"])
@login_required
def save_keywords(slug):
    project = Project.query.filter_by(slug=slug).first_or_404()
    keywords_json = request.form.get("keywords_json")
    if keywords_json:
        try:
            keywords = [k.strip() for k in json.loads(keywords_json) if str(k).strip()]
            project.saved_keywords = json.dumps(keywords) if keywords else None  # all removed: none saved
            db.session.commit()
        except (json.JSONDecodeError, ValueError):
            return jsonify({"error": "Invalid keywords"}), 400
    return jsonify({"ok": True})


@projects_bp.route("/<slug>/stop-crawl", methods=["POST"])
@login_required
def stop_crawl(slug):
    project = Project.query.filter_by(slug=slug).first_or_404()

    running_crawl = CrawlLog.query.filter_by(project_id=project.id, status="running").first()

    if running_crawl:
        running_crawl.status = "cancelled"
        running_crawl.finished_at = datetime.utcnow()
        db.session.commit()

        message = "Crawl stopped"
    else:
        message = "No running crawl to stop"

    if request.headers.get("HX-Request"):
        crawl = (
            CrawlLog.query.filter_by(project_id=project.id)
            .order_by(CrawlLog.started_at.desc())
            .first()
        )
        return render_template("partials/crawl_status.html", project=project, crawl=crawl)

    return redirect(url_for("projects.detail", slug=slug, message=message))


def _has_collection(project) -> bool:
    return ProjectPaper.query.filter_by(project_id=project.id, trashed_at=None).filter(
        ProjectPaper.manual_tag.isnot(None)).first() is not None


@projects_bp.route("/<slug>/crawl-banner")
@login_required
def crawl_banner(slug):
    """The "Nightly crawl is off" banner, redrawn when the setup it lists changes (HX-Trigger: crawlSetupChanged)."""
    project = Project.query.filter_by(slug=slug).first_or_404()
    return render_template("partials/crawl_banner.html", project=project, curated_papers=_has_collection(project))


def setup_changed(body):
    """A response to something the nightly-crawl checklist shows: the banner redraws itself."""
    resp = make_response(body)
    resp.headers["HX-Trigger"] = "crawlSetupChanged"
    return resp


@projects_bp.route("/<slug>/notification-emails", methods=["POST"])
@login_required
def update_notification_emails(slug):
    project = Project.query.filter_by(slug=slug).first_or_404()

    import re as _re
    _email_re = _re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')

    emails_raw = request.form.get("notification_emails", "").strip()
    if not emails_raw:
        project.notification_emails = None
        db.session.commit()
        if request.headers.get("HX-Request"):
            return setup_changed('<span style="color: var(--success); font-size: 0.75rem;">Cleared</span>')
        return redirect(url_for("projects.detail", slug=slug))

    candidates = [e.strip() for e in emails_raw.split(",") if e.strip()]
    invalid = [e for e in candidates if not _email_re.match(e)]
    if invalid:
        if request.headers.get("HX-Request"):
            return '<span style="color: var(--error); font-size: 0.75rem;">Invalid input. Use comma-separated emails only</span>'
        return redirect(url_for("projects.detail", slug=slug))

    emails = ",".join(candidates)
    project.notification_emails = emails
    db.session.commit()

    if request.headers.get("HX-Request"):
        return setup_changed('<span style="color: var(--success); font-size: 0.75rem;">Saved</span>')

    return redirect(url_for("projects.detail", slug=slug))


@projects_bp.route("/<slug>/toggle-nightly-crawl", methods=["POST"])
@login_required
def toggle_nightly_crawl(slug):
    """Enable/disable the nightly cron crawl for this project. Enabling requires a research
    interest + saved keywords, and auto-assigns a crawl_hour (1-4 UTC) — preferring an hour
    no other project already uses, to spread nightly load."""
    project = Project.query.filter_by(slug=slug).first_or_404()

    if project.crawl_hour is not None:
        project.crawl_hour = None
    else:
        if not project.research_interest or not project.saved_keywords:
            missing = []
            if not project.research_interest:
                missing.append("a research interest")
            if not project.saved_keywords:
                missing.append("keywords")
            msg = "Set " + " and ".join(missing) + " first"
            return f'<span id="crawl-schedule-text" data-error="1" style="color:var(--error);">{msg}</span>'
        taken = {p.crawl_hour for p in Project.query.filter_by(trashed_at=None).all() if p.crawl_hour is not None}
        available = [h for h in range(1, 5) if h not in taken]
        project.crawl_hour = random.choice(available) if available else random.randint(1, 4)

    db.session.commit()
    current_app.logger.info(f"[nightly] toggle for {project.name!r}: crawl_hour now {project.crawl_hour}")

    if project.crawl_hour is not None:
        schedule_html = (
            f'<span id="crawl-schedule-text">Daily at '
            f'{project.crawl_hour:02d}:00 UTC</span>'
        )
    else:
        schedule_html = '<span id="crawl-schedule-text" style="color:var(--ink-4);">Off</span>'

    return setup_changed(schedule_html)


@projects_bp.route("/health-check", methods=["GET"])
@login_required
def health_check():
    """Run live smoke tests against external services in parallel. Returns JSON with pass/fail per service."""
    import time
    import arxiv as arxiv_lib
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def check_llm():
        t0 = time.time()
        try:
            from app.llm import _llm_json
            messages = [
                {"role": "system", "content": "You are a JSON API. Reply only with valid JSON."},
                {"role": "user", "content": 'Reply with exactly: {"ok": true}'},
            ]
            payload, model, elapsed_ms, _ = _llm_json(messages, temperature=0.0, max_tokens=1024)
            ok = payload.get("ok") is True
            return "llm", {"ok": ok, "model": model, "elapsed_ms": round(elapsed_ms), "error": None if ok else f"Unexpected payload: {payload}"}
        except Exception as e:
            return "llm", {"ok": False, "model": None, "elapsed_ms": round((time.time() - t0) * 1000), "error": str(e)}

    def check_arxiv():
        t0 = time.time()
        from app.crawl import _ArxivLockCtx
        import time as _time
        with _ArxivLockCtx(timeout=5.0) as acquired:
            if not acquired:
                depth = arxiv_queue_depth()
                return "arxiv", {"ok": None, "papers_returned": 0,
                                 "elapsed_ms": round((time.time() - t0) * 1000),
                                 "error": f"arXiv locked — crawl/search in progress ({depth} queued)"}
            try:
                # page_size=2 keeps the request small; num_retries=0 + delay_seconds=0 fails fast on 429.
                # NOTE: a 429 here does NOT mean our code is misbehaving. arXiv rate-limits by IP, and the
                # cluster uses a shared egress IP. Other workloads on the same nodes can exhaust the quota
                # even when our app is idle. The 20–30s hang before the 429 arrives is arXiv deliberately
                # slow-responding at the TCP level under heavy throttling — not a retry loop on our side.
                # Nothing to fix; self-resolves as cluster traffic eases.
                client = arxiv_lib.Client(page_size=2, delay_seconds=0, num_retries=0)
                search = arxiv_lib.Search(id_list=["1706.03762"])  # fetch one known paper by ID
                results_list = list(client.results(search))
                ok = len(results_list) > 0
                return "arxiv", {"ok": ok, "papers_returned": len(results_list), "elapsed_ms": round((time.time() - t0) * 1000), "error": None if ok else "No papers returned"}
            except arxiv_lib.HTTPError as e:
                code = getattr(e, "status", "?")
                msg = f"HTTP {code} — arXiv rate limit" if "429" in str(e) else f"HTTP {code} — arXiv temporarily unavailable" if "503" in str(e) else str(e)
                return "arxiv", {"ok": False, "papers_returned": 0, "elapsed_ms": round((time.time() - t0) * 1000), "error": msg}
            except Exception as e:
                return "arxiv", {"ok": False, "papers_returned": 0, "elapsed_ms": round((time.time() - t0) * 1000), "error": str(e)}
            finally:
                _time.sleep(3.5)  # hold the gap before releasing

    def check_scholar():
        t0 = time.time()
        try:
            from app.crawl import get_semantic_scholar_recommendations
            recs = get_semantic_scholar_recommendations(["2307.09288"], limit=3)  # Llama 2 — recent, well-indexed
            ok = len(recs) > 0
            return "semantic_scholar", {"ok": ok, "papers_returned": len(recs), "elapsed_ms": round((time.time() - t0) * 1000), "error": None if ok else "No recommendations returned"}
        except Exception as e:
            return "semantic_scholar", {"ok": False, "papers_returned": 0, "elapsed_ms": round((time.time() - t0) * 1000), "error": str(e)}

    def check_figures():
        t0 = time.time()
        try:
            from app.crawl import get_paper_figure
            test_id = "2605.30322v1"
            url, caption = get_paper_figure(test_id)
            ok = url is not None and url.startswith("http")
            doubled = bool(url and f"{test_id}/{test_id}" in url)
            return "figure_fetch", {"ok": ok and not doubled, "url": url, "caption": caption, "elapsed_ms": round((time.time() - t0) * 1000), "error": "doubled ID in URL" if doubled else (None if ok else "No figure URL returned")}
        except Exception as e:
            return "figure_fetch", {"ok": False, "url": None, "elapsed_ms": round((time.time() - t0) * 1000), "error": str(e)}

    import logging as _logging
    _hc_log = _logging.getLogger(__name__)

    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(f) for f in (check_llm, check_arxiv, check_scholar, check_figures)]
        for future in as_completed(futures):
            key, val = future.result()
            results[key] = val

    all_ok = all(v["ok"] for v in results.values())
    status = "✓ all OK" if all_ok else "✗ FAILURES"
    for svc, val in sorted(results.items()):
        icon = "✓" if val["ok"] else "✗"
        extra = f" — {val['error']}" if val.get("error") else ""
        _hc_log.info("[health-check] %s %s (%dms)%s", icon, svc, val.get("elapsed_ms", 0), extra)
    _hc_log.info("[health-check] %s", status)
    return jsonify({"all_ok": all_ok, "services": results})
