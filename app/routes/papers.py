import csv
import io
import json
import logging
from datetime import datetime

from flask import render_template, request, jsonify, Response

_log = logging.getLogger(__name__)

from app import db
from app.auth import login_required
from app.models import Paper, ProjectPaper, Project, CrawlLog
from app.routes import papers_bp
from app.crawl import (
    generate_summary_with_full_content,
    get_paper_figure,
    get_citation_count,
    get_citation_count_by_scholar_id,
    enrich_institutions_from_scholar,
)
from app.llm import LLMUnavailableError

MULTI_TAGS = {"important", "to_read", "to_discuss"}


def _get_paper_lists(project):
    """Build new_papers / curated_papers / trashed_papers for a project (same logic as detail view)."""
    recent_crawl = (
        CrawlLog.query.filter_by(project_id=project.id)
        .order_by(CrawlLog.started_at.desc())
        .first()
    )
    last_crawl_time = recent_crawl.started_at if recent_crawl and recent_crawl.started_at else None

    all_pp = (
        ProjectPaper.query.filter_by(project_id=project.id)
        .order_by(ProjectPaper.added_at.desc())
        .all()
    )

    new_papers, curated_papers, trashed_papers = [], [], []
    for pp in all_pp:
        if pp.trashed_at:
            trashed_papers.append(pp)
        elif pp.manual_tag is None:
            new_papers.append(pp)
        else:
            curated_papers.append(pp)

    return new_papers, curated_papers, trashed_papers


def _paper_list_response(project_id):
    project = Project.query.get(project_id)
    if not project:
        return ("", 204)
    new_papers, curated_papers, trashed_papers = _get_paper_lists(project)
    return render_template(
        "partials/paper_list.html",
        project=project,
        new_papers=new_papers,
        curated_papers=curated_papers,
        trashed_papers=trashed_papers,
    )


@papers_bp.route("/project/<slug>/papers")
@login_required
def list_papers(slug):
    project = Project.query.filter_by(slug=slug).first_or_404()

    last_crawl = (
        CrawlLog.query.filter_by(project_id=project.id, status="success")
        .order_by(CrawlLog.finished_at.desc())
        .first()
    )

    last_crawl_time = last_crawl.finished_at if last_crawl else None

    new_papers_query = (
        ProjectPaper.query.filter_by(project_id=project.id, trashed_at=None)
        .filter(ProjectPaper.manual_tag.is_(None))
        .join(Paper)
    )

    if last_crawl_time:
        new_papers_query = new_papers_query.filter(ProjectPaper.added_at >= last_crawl_time)

    search_term = request.args.get("search", "")
    if search_term:
        new_papers_query = new_papers_query.filter(
            Paper.title.ilike(f"%{search_term}%") | Paper.abstract.ilike(f"%{search_term}%")
        )

    new_papers = new_papers_query.order_by(ProjectPaper.added_at.desc()).all()

    curated_papers = (
        ProjectPaper.query.filter_by(project_id=project.id, trashed_at=None)
        .filter(ProjectPaper.manual_tag.isnot(None))
        .order_by(ProjectPaper.added_at.desc())
        .all()
    )

    if search_term:
        curated_papers = [
            p
            for p in curated_papers
            if search_term.lower() in p.paper.title.lower()
            or search_term.lower() in (p.paper.abstract or "").lower()
        ]

    trashed_papers = (
        ProjectPaper.query.filter_by(project_id=project.id)
        .filter(ProjectPaper.trashed_at.isnot(None))
        .order_by(ProjectPaper.trashed_at.desc())
        .all()
    )

    search_query = request.args.get("search", "").strip()
    if search_query:
        new_papers = [
            p
            for p in new_papers
            if search_query.lower() in p.paper.title.lower()
            or search_query.lower() in (p.paper.abstract or "").lower()
        ]
        curated_papers = [
            p
            for p in curated_papers
            if search_query.lower() in p.paper.title.lower()
            or search_query.lower() in (p.paper.abstract or "").lower()
        ]

    return render_template(
        "partials/paper_list.html",
        project=project,
        new_papers=new_papers,
        curated_papers=curated_papers,
        trashed_papers=trashed_papers,
    )


@papers_bp.route("/paper/<arxiv_id>/tag", methods=["POST"])
@login_required
def tag_paper(arxiv_id):
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    project_id = request.form.get("project_id", type=int)
    tag = request.form.get("tag")

    if project_id:
        pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first()
        if pp:
            if tag in MULTI_TAGS:
                tags = pp.tags_list
                if tag in tags:
                    tags.remove(tag)
                else:
                    tags.append(tag)
                pp.paper_tags = json.dumps(tags)
                if pp.manual_tag is None:
                    pp.manual_tag = "related"
                    pp.collected_at = datetime.utcnow()
            elif tag == "":
                pp.manual_tag = None
                pp.paper_tags = "[]"
                pp.collected_at = None
            elif tag == "related":
                pp.manual_tag = "related"
                pp.collected_at = datetime.utcnow()
                tags = pp.tags_list
                if "to_read" not in tags:
                    tags.append("to_read")
                    pp.paper_tags = json.dumps(tags)
            db.session.commit()

    if request.headers.get("HX-Request") and project_id:
        hx_target = request.headers.get("HX-Target", "")
        if hx_target.startswith("paper-card-") and pp is not None:
            project = Project.query.get(project_id)
            if project:
                return render_template("partials/paper_card.html", pp=pp, project=project, show_curate=False)
        return _paper_list_response(project_id)
    return ("", 204)


@papers_bp.route("/paper/<arxiv_id>/notes", methods=["POST"])
@login_required
def update_notes(arxiv_id):
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    project_id = request.form.get("project_id", type=int)
    notes = request.form.get("notes", "")
    notes_height = request.form.get("notes_height", type=int)

    if project_id:
        pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first()
        if pp:
            pp.notes = notes
            if notes_height:
                pp.notes_height = notes_height
            db.session.commit()

    return ("", 204)


@papers_bp.route("/paper/<arxiv_id>/trash", methods=["POST"])
@login_required
def trash_paper(arxiv_id):
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    project_id = request.form.get("project_id", type=int)

    if project_id:
        pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first()
        if pp:
            pp.trashed_at = datetime.utcnow()
            db.session.commit()

    if request.headers.get("HX-Request") and project_id:
        return _paper_list_response(project_id)
    return ("", 204)


@papers_bp.route("/paper/<arxiv_id>/restore", methods=["POST"])
@login_required
def restore_paper(arxiv_id):
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    project_id = request.form.get("project_id", type=int)

    if project_id:
        pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first()
        if pp:
            pp.trashed_at = None
            pp.manual_tag = None
            pp.collected_at = None
            db.session.commit()

    if request.headers.get("HX-Request") and project_id:
        return _paper_list_response(project_id)
    return ("", 204)


@papers_bp.route("/paper/<arxiv_id>/tag-and-notify", methods=["POST"])
@login_required
def tag_and_notify(arxiv_id):
    """Toggle a MULTI_TAGS label (important/to_read/to_discuss) and, only when the tag was
    newly added (not removed) and the project has notification emails set, send the
    "paper flagged" email. Removing a tag never triggers a notification."""
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    project_id = request.form.get("project_id", type=int)
    tag = request.form.get("tag")
    added = False

    if project_id:
        pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first()
        if pp and tag in MULTI_TAGS:
            tags = pp.tags_list
            if tag in tags:
                tags.remove(tag)
                added = False
            else:
                tags.append(tag)
                added = True
            pp.paper_tags = json.dumps(tags)
            if pp.manual_tag is None:
                pp.manual_tag = "related"
                pp.collected_at = datetime.utcnow()
            db.session.commit()

        project = Project.query.get(project_id)
        if added and pp and project and project.notification_emails:
            from app.crawl import notify_important_paper

            notify_important_paper(
                paper=paper,
                project_name=project.name,
                project_slug=project.slug,
                figure_url=paper.figure_url,
                figure_caption=paper.figure_caption,
                notes=pp.notes if pp else None,
            )

    if request.headers.get("HX-Request") and project_id:
        hx_target = request.headers.get("HX-Target", "")
        if hx_target.startswith("paper-card-") and pp is not None and project is not None:
            return render_template("partials/paper_card.html", pp=pp, project=project, show_curate=False)
        return _paper_list_response(project_id)
    return ("", 204)


@papers_bp.route("/paper/<arxiv_id>/regenerate-summary", methods=["POST"])
@login_required
def regenerate_summary(arxiv_id):
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    project_id = request.form.get("project_id", type=int)
    project = Project.query.get(project_id) if project_id else None

    try:
        summary_data, model, elapsed, tokens = generate_summary_with_full_content(
            title=paper.title,
            abstract=paper.abstract or "",
            paper_content=paper.abstract,
            research_interest=project.research_interest if project else "",
        )
    except LLMUnavailableError as e:
        return (f"LLM unavailable: {str(e)}", 503)

    if paper.summary:
        paper.summary.summary_text = summary_data.get("summary", "")
        paper.summary.model_used = model
        paper.summary.introduces_dataset = summary_data.get("introduces_dataset", False)
        paper.summary.introduces_architecture = summary_data.get("introduces_architecture", False)
        paper.summary.introduces_method = summary_data.get("introduces_method", False)
        paper.summary.is_survey = summary_data.get("is_survey", False)
        paper.summary.is_benchmark = summary_data.get("is_benchmark", False)
        paper.summary.elapsed_ms = elapsed
        paper.summary.tokens_used = (paper.summary.tokens_used or 0) + tokens
    else:
        from app.models import PaperSummary

        summary = PaperSummary(
            paper_id=paper.id,
            summary_text=summary_data.get("summary", ""),
            model_used=model,
            introduces_dataset=summary_data.get("introduces_dataset", False),
            introduces_architecture=summary_data.get("introduces_architecture", False),
            introduces_method=summary_data.get("introduces_method", False),
            is_survey=summary_data.get("is_survey", False),
            is_benchmark=summary_data.get("is_benchmark", False),
            elapsed_ms=elapsed,
            tokens_used=tokens,
        )
        db.session.add(summary)

    db.session.commit()

    if request.headers.get("HX-Request") and project_id:
        pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first()
        if pp:
            show_curate = pp.manual_tag is None
            return render_template("partials/paper_card.html", pp=pp, project=project, show_curate=show_curate)

    return ("", 204)


@papers_bp.route("/paper/<arxiv_id>/regenerate", methods=["POST"])
@login_required
def regenerate_paper(arxiv_id):
    """Re-fetch figure, citations, institutions and regenerate summary for a single paper."""
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    project_id = request.form.get("project_id", type=int)
    project = Project.query.get(project_id) if project_id else None

    if paper.is_arxiv:
        # Figure
        try:
            figure_url, figure_caption = get_paper_figure(arxiv_id)
            if figure_url:
                paper.figure_url = figure_url
                paper.figure_caption = figure_caption
        except Exception as e:
            _log.warning("[regenerate] figure fetch failed for %s: %s", arxiv_id, e)

        # Citations
        try:
            count = get_citation_count(arxiv_id)
            if count is not None:
                if not paper.citation_manual or count >= paper.citation_count:
                    paper.citation_count = count
                    paper.citation_manual = False
                paper.citation_fetched_at = datetime.utcnow()
        except Exception as e:
            _log.warning("[regenerate] citation fetch failed for %s: %s", arxiv_id, e)

    elif paper.semantic_scholar_id:
        # Non-arXiv paper with a Scholar ID — fetch citations only
        try:
            count = get_citation_count_by_scholar_id(paper.semantic_scholar_id)
            if count is not None:
                if not paper.citation_manual or count >= paper.citation_count:
                    paper.citation_count = count
                    paper.citation_manual = False
                paper.citation_fetched_at = datetime.utcnow()
        except Exception as e:
            _log.warning("[regenerate] citation fetch failed for scholar:%s: %s", paper.semantic_scholar_id, e)

        # Institutions (batch function takes a list; skips if already set — force by clearing)
        paper.institutions = None
        try:
            enrich_institutions_from_scholar([paper])
        except Exception as e:
            _log.warning("[regenerate] institution enrich failed for %s: %s", arxiv_id, e)

    # Summary
    try:
        summary_data, model, elapsed, tokens = generate_summary_with_full_content(
            title=paper.title,
            abstract=paper.abstract or "",
            paper_content=paper.abstract,
            research_interest=project.research_interest if project else "",
        )
        from app.models import PaperSummary
        if paper.summary:
            paper.summary.summary_text = summary_data.get("summary", "")
            paper.summary.model_used = model
            paper.summary.introduces_dataset = summary_data.get("introduces_dataset", False)
            paper.summary.introduces_architecture = summary_data.get("introduces_architecture", False)
            paper.summary.introduces_method = summary_data.get("introduces_method", False)
            paper.summary.is_survey = summary_data.get("is_survey", False)
            paper.summary.is_benchmark = summary_data.get("is_benchmark", False)
            paper.summary.elapsed_ms = elapsed
            paper.summary.tokens_used = (paper.summary.tokens_used or 0) + tokens
        else:
            db.session.add(PaperSummary(
                paper_id=paper.id,
                summary_text=summary_data.get("summary", ""),
                model_used=model,
                introduces_dataset=summary_data.get("introduces_dataset", False),
                introduces_architecture=summary_data.get("introduces_architecture", False),
                introduces_method=summary_data.get("introduces_method", False),
                is_survey=summary_data.get("is_survey", False),
                is_benchmark=summary_data.get("is_benchmark", False),
                elapsed_ms=elapsed,
                tokens_used=tokens,
            ))
    except LLMUnavailableError as e:
        _log.warning("[regenerate] LLM unavailable for %s: %s", arxiv_id, e)

    db.session.commit()

    if request.headers.get("HX-Request") and project_id:
        pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first()
        if pp:
            show_curate = pp.manual_tag is None
            return render_template("partials/paper_card.html", pp=pp, project=project, show_curate=show_curate)
    return ("", 204)


@papers_bp.route("/paper/<arxiv_id>/notify", methods=["POST"])
@login_required
def notify_paper(arxiv_id):
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    project_id = request.form.get("project_id", type=int)

    emails_used = None
    error = None
    sent = False
    if project_id:
        project = Project.query.get(project_id)
        if project:
            if not project.notification_emails:
                error = "No notification emails configured for this project"
            else:
                from app.crawl import notify_important_paper
                pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first()
                sent = notify_important_paper(
                    paper=paper,
                    project_name=project.name,
                    project_slug=project.slug,
                    figure_url=paper.figure_url,
                    figure_caption=paper.figure_caption,
                    project_emails=project.notification_emails,
                    notes=pp.notes if pp else None,
                )
                if sent:
                    emails_used = project.notification_emails
                else:
                    error = "Email send failed — check server logs for details"
        else:
            error = "Project not found"
    else:
        error = "No project context"

    return jsonify({"ok": sent, "emails": emails_used, "error": error})


@papers_bp.route("/paper/<arxiv_id>/citations", methods=["POST"])
@login_required
def update_citation_count(arxiv_id):
    count = request.form.get("count", type=int)
    if count is None or count < 0:
        return jsonify({"ok": False, "error": "Invalid count"}), 400
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    paper.citation_count = count
    paper.citation_fetched_at = datetime.utcnow()
    paper.citation_manual = True
    db.session.commit()
    return jsonify({"ok": True, "count": count, "as_of": datetime.utcnow().strftime("%b %Y")})


@papers_bp.route("/project/<slug>/export.csv")
@login_required
def export_csv(slug):
    project = Project.query.filter_by(slug=slug).first_or_404()
    new_papers, curated_papers, _ = _get_paper_lists(project)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["title", "authors", "url", "tag", "abstract"])

    for pp in new_papers + curated_papers:
        p = pp.paper
        try:
            authors = "; ".join(json.loads(p.authors)) if p.authors else ""
        except (json.JSONDecodeError, TypeError):
            authors = p.authors or ""
        base_id = p.arxiv_id.split("v")[0] if p.arxiv_id and not p.arxiv_id.startswith("web:") else ""
        arxiv_url = f"https://arxiv.org/abs/{base_id}" if base_id else (p.arxiv_id or "")
        writer.writerow([p.title or "", authors, arxiv_url, pp.manual_tag or "untagged", p.abstract or ""])

    csv_bytes = buf.getvalue().encode("utf-8")
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={slug}-papers.csv"},
    )
