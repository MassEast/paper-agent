import csv
import io
import json
import logging
from datetime import datetime

from flask import current_app, render_template, request, jsonify, Response

_log = logging.getLogger(__name__)

from app import db
from app.auth import login_required
from app.models import ChatMessage, Paper, ProjectPaper, Project
from app import chat
from app import paper_lists
from app.routes import papers_bp
from app.utils import base_arxiv_id
from app.crawl import (
    generate_summary_with_full_content,
    get_paper_figure,
    get_citation_count,
    get_citation_count_by_scholar_id,
    enrich_institutions_from_scholar,
    _enrich_one_web_paper,
    start_relevance_reason_async,
    generate_relevance_reason,
    get_paper_content,
    extract_main_contributions,
    summary_source,
)
from app.llm import LLMUnavailableError

MULTI_TAGS = {"important", "to_read", "to_discuss"}


def _moved_response(project_id, dest):
    """A card left its section (the request targets that card or trash row, outerHTML): the empty main
    response removes it, and out-of-band swaps update the header counts, re-render the section it moved
    into (so it shows up there at once, in the current sort), and any section it left empty.
    Only the destination is re-rendered, so the reader keeps their place in the list they were working in."""
    project = Project.query.get(project_id)
    if not project:
        return ("", 204)
    state = paper_lists.list_state(request.values)
    s = paper_lists.sections(project, state)
    ctx = dict(project=project, s=s, state=state, page_size=paper_lists.PAGE_SIZE, oob=True)
    redraw = [dest] + [sec for sec in paper_lists.SECTIONS if sec != dest and not s[sec]]
    resp = current_app.make_response(render_template("partials/counts_oob.html", **ctx) + "".join(
        render_template(f"partials/section_{sec}.html", **ctx) for sec in redraw))
    resp.headers["HX-Trigger"] = "crawlSetupChanged"  # My Collection changed: the nightly-crawl checklist
    return resp


def _card_response(pp, project):
    """A card changed in place (a tag toggled): the card itself, plus the tag counts out of band."""
    state = paper_lists.list_state(request.values)
    s = paper_lists.sections(project, state)
    resp = current_app.make_response(
        render_template("partials/paper_card.html", pp=pp, project=project, show_curate=False)
        + render_template("partials/counts_oob.html", project=project, s=s, state=state))
    resp.headers["HX-Trigger"] = "crawlSetupChanged"  # a tag can put a paper into My Collection
    return resp


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
        if tag in MULTI_TAGS and pp is not None:
            return _card_response(pp, Project.query.get(project_id))
        return _moved_response(project_id, "col" if tag == "related" else "new")
    return ("", 204)


def _chat_pp(arxiv_id):
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    project_id = request.values.get("project_id", type=int)
    return ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first_or_404()


def _chat_panel(pp, error=None, focus=False):
    return render_template("partials/chat_panel.html", pp=pp, project=pp.project, error=error, focus=focus)


@papers_bp.route("/paper/<arxiv_id>/chat")
@login_required
def chat_panel(arxiv_id):
    """The chat panel for a card that has no chat yet (the card renders an existing one itself)."""
    return _chat_panel(_chat_pp(arxiv_id), focus=True)


@papers_bp.route("/paper/<arxiv_id>/chat", methods=["POST"])
@login_required
def chat_ask(arxiv_id):
    pp = _chat_pp(arxiv_id)
    question = (request.form.get("question") or "").strip()
    error = chat.ask(pp, question, current_app._get_current_object()) if question else None
    return _chat_panel(pp, error=error, focus=True)


@papers_bp.route("/paper/<arxiv_id>/chat/clear", methods=["POST"])
@login_required
def chat_clear(arxiv_id):
    pp = _chat_pp(arxiv_id)
    chat.clear(pp)
    db.session.expire(pp, ["chat_messages"])
    return render_template("partials/chat_slot.html", pp=pp, project=pp.project)


@papers_bp.route("/chat/message/<int:msg_id>")
@login_required
def chat_message(msg_id):
    """Polled while an answer is written: the answer so far; once it has ended, the conversation (which
    may already show the next queued question being answered) plus the head and notice out of band."""
    live = chat.live_state(msg_id)
    if live:
        m = ChatMessage.query.get_or_404(msg_id)
        html = render_template("partials/chat_live.html", m=m, live=live)
        ctx = chat.live_context(msg_id)
        if ctx:  # the meter counts up while the answer is written
            html += render_template("partials/chat_context.html", ctx=ctx, pp_id=m.project_paper_id, oob=True)
        return html
    msg = ChatMessage.query.get(msg_id)
    if msg is None:  # the chat was cleared meanwhile
        return ("", 286)  # htmx: stop polling
    return _chat_redraw(msg.project_paper)


def _chat_redraw(pp):
    """The log (into #chat-log-<pp.id>) with head and notice out of band; the form is left alone."""
    ctx = dict(pp=pp, project=pp.project, cv=chat.view(pp, current_app._get_current_object()), oob=True)
    resp = current_app.make_response(render_template("partials/chat_log.html", **ctx)
                                      + render_template("partials/chat_head.html", **ctx)
                                      + render_template("partials/chat_notice.html", **ctx))
    resp.headers["HX-Retarget"] = f"#chat-log-{pp.id}"
    resp.headers["HX-Reswap"] = "outerHTML"
    return resp


@papers_bp.route("/chat/message/<int:msg_id>/stop", methods=["POST"])
@login_required
def chat_stop(msg_id):
    chat.stop(msg_id)
    return ("", 204)


@papers_bp.route("/chat/message/<int:msg_id>/remove", methods=["POST"])
@login_required
def chat_remove_queued(msg_id):
    """Take a queued question back. Too late if it has been sent meanwhile: then the conversation is redrawn."""
    msg = ChatMessage.query.get_or_404(msg_id)
    pp = msg.project_paper
    if chat.remove_queued(msg_id):
        return ""
    return _chat_redraw(pp)


@papers_bp.route("/project-paper/<int:pp_id>/card")
@login_required
def paper_card_refresh(pp_id):
    """Polled by a card whose details are still being fetched in the background: nothing while the step
    it shows is current; the whole card once the step changed or the work is done. Keyed by the
    ProjectPaper, which stays the same when a web add turns out to be an arXiv paper."""
    from app.crawl import enrich_step
    pp = ProjectPaper.query.get(pp_id)
    if pp is None:
        return ("", 286)  # htmx: stop polling
    step = enrich_step(pp.paper_id)
    if step is not None and step == request.args.get("step"):
        return ("", 204)
    resp = current_app.make_response(render_template(
        "partials/paper_card.html", pp=pp, project=pp.project, show_curate=pp.manual_tag is None))
    resp.headers["HX-Retarget"] = f'[data-pp-card="{pp.id}"]'
    resp.headers["HX-Reswap"] = "outerHTML"
    return resp


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
        return _moved_response(project_id, "trash")
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
        return _moved_response(project_id, "new")
    return ("", 204)


@papers_bp.route("/paper/<arxiv_id>/transfer-menu")
@login_required
def transfer_menu(arxiv_id):
    """The card's Copy / Move panel: every other (non-trashed) project, with where the paper
    already is there."""
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    project = Project.query.get_or_404(request.args.get("project_id", type=int))
    targets = []
    for other in Project.query.filter(Project.id != project.id, Project.trashed_at.is_(None)).order_by(Project.name):
        link = ProjectPaper.query.filter_by(project_id=other.id, paper_id=paper.id).first()
        in_collection = link is not None and link.manual_tag is not None and not link.trashed_at
        targets.append((other, in_collection))
    return render_template("partials/transfer_menu.html", paper=paper, project=project, targets=targets)


@papers_bp.route("/paper/<arxiv_id>/transfer", methods=["POST"])
@login_required
def transfer_paper(arxiv_id):
    """Copy or move a paper into another project's My Collection. Tags, notes and the paper chat come
    along (notes and chat only if the target has none). Move also puts the paper in this project's Trash, not
    a hard delete, so this project's crawls don't add it again. The target gets its own
    "why it's relevant" line in the background if it has none yet."""
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    project_id = request.form.get("project_id", type=int)
    target_id = request.form.get("target_project_id", type=int)
    mode = request.form.get("mode")
    if mode not in ("copy", "move") or target_id == project_id:
        return ("Invalid transfer", 400)
    source = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first_or_404()
    target_project = Project.query.get_or_404(target_id)

    now = datetime.utcnow()
    target = ProjectPaper.query.filter_by(project_id=target_id, paper_id=paper.id).first()
    if target is None:
        target = ProjectPaper(project_id=target_id, paper_id=paper.id, added_at=now,
                              origin="moved" if mode == "move" else "copied", origin_project_id=project_id)
        db.session.add(target)
    target.trashed_at = None
    if target.manual_tag is None:
        target.manual_tag = "related"
        target.collected_at = now
    tags = target.tags_list
    for tag in source.tags_list:
        if tag not in tags:
            tags.append(tag)
    target.paper_tags = json.dumps(tags)
    if not target.notes and source.notes:
        target.notes = source.notes
    db.session.flush()  # a new target needs its id for the chat
    chat.copy_chat(source, target)
    if mode == "move":
        source.trashed_at = now
    db.session.commit()
    _log.info("[transfer] %s %s from project %s to %s", mode, arxiv_id, project_id, target_id)

    if not target.relevance_reason:
        start_relevance_reason_async(target_id, paper.id)

    if mode == "move":
        return _moved_response(project_id, "trash")
    return render_template("partials/transfer_menu.html", paper=paper, project=Project.query.get(project_id),
                           copied_to=target_project)


@papers_bp.route("/paper/<arxiv_id>/regenerate", methods=["POST"])
@login_required
def regenerate_paper(arxiv_id):
    """Re-fetch figure + citations (arXiv) or citations + institutions (Scholar-only paper), then
    regenerate the shared summary + main contributions (full text, like a crawl) and this project's
    why-relevant line (abstract)."""
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

    # Summary + main contributions, from the full text like a crawl (abstract if there is none)
    content = (get_paper_content(arxiv_id) if paper.is_arxiv else None) or paper.abstract or ""
    try:
        summary_data, model, elapsed, tokens = generate_summary_with_full_content(
            title=paper.title,
            abstract=paper.abstract or "",
            paper_content=content,
        )
        contributions, _, contrib_elapsed, contrib_tokens = extract_main_contributions(
            title=paper.title, paper_content=content, abstract=paper.abstract or "",
        )
        elapsed, tokens = elapsed + contrib_elapsed, tokens + contrib_tokens
        source = summary_source(content, paper.abstract or "")
        from app.models import PaperSummary
        if paper.summary:
            paper.summary.content_source = source
            if contributions:
                paper.summary.main_contributions = contributions
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
                content_source=source,
                summary_text=summary_data.get("summary", ""),
                main_contributions=contributions,
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

    # This project's why-relevant line (the summary above is shared by all projects)
    pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first() if project_id else None
    if pp and project and (project.research_interest or "").strip():
        try:
            why = generate_relevance_reason(paper.title, paper.abstract or "", project.research_interest)
            if why:
                pp.relevance_reason = why
        except LLMUnavailableError as e:
            _log.warning("[regenerate] LLM unavailable for why-line of %s: %s", arxiv_id, e)

    db.session.commit()

    if request.headers.get("HX-Request") and project_id:
        pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first()
        if pp:
            show_curate = pp.manual_tag is None
            return render_template("partials/paper_card.html", pp=pp, project=project, show_curate=show_curate)
    return ("", 204)


@papers_bp.route("/paper/<arxiv_id>/retry-web-enrich", methods=["POST"])
@login_required
def retry_web_enrich(arxiv_id):
    """Manually reset a stuck 'web:' paper's attempt counter and retry PDF metadata
    enrichment in the background — the nightly Pass-3 backfill otherwise gives up on it
    permanently after MAX_WEB_ENRICH_ATTEMPTS failed attempts."""
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    if not paper.arxiv_id.startswith("web:"):
        return jsonify({"ok": False, "error": "Not a web paper"}), 400

    paper.web_enrich_attempts = 0
    paper.web_enrich_last_attempt_at = None
    db.session.commit()

    from flask import current_app
    _app = current_app._get_current_object()
    paper_id = paper.id

    def _run():
        with _app.app_context():
            p = Paper.query.get(paper_id)
            if not p:
                return
            try:
                _enrich_one_web_paper(p)
                db.session.commit()
                _log.info("[retry-web-enrich] %s: page_count=%s institutions=%s",
                          p.arxiv_id, p.page_count, p.institutions)
            except Exception as e:
                db.session.rollback()
                _log.warning("[retry-web-enrich] failed for %s: %s", p.arxiv_id, e)

    import threading
    threading.Thread(target=_run, daemon=True).start()

    return jsonify({"ok": True})


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
    s = paper_lists.sections(project, paper_lists.list_state({}))
    new_papers, curated_papers = s["new"], s["col"]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["title", "authors", "url", "tag", "abstract"])

    for pp in new_papers + curated_papers:
        p = pp.paper
        try:
            authors = "; ".join(json.loads(p.authors)) if p.authors else ""
        except (json.JSONDecodeError, TypeError):
            authors = p.authors or ""
        # Same link as the card: arXiv abs page, or the stored URL of a web-added paper
        url = f"https://arxiv.org/abs/{base_arxiv_id(p.arxiv_id)}" if p.is_arxiv else (p.pdf_url or "")
        writer.writerow([p.title or "", authors, url, pp.manual_tag or "untagged", p.abstract or ""])

    csv_bytes = buf.getvalue().encode("utf-8")
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={slug}-papers.csv"},
    )
