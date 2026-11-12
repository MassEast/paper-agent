import csv
import io
import json
import logging
import re
from datetime import datetime

from flask import current_app, render_template, request, jsonify, Response, send_from_directory

_log = logging.getLogger(__name__)

from app import db
from app.auth import login_required
from app.usage import counted
from app.models import ChatMessage, Paper, ProjectPaper, Project
from app import arxiv_meta, chat, resources
from app import figures, members, note_locks, paper_lists
from app.routes import papers_bp
from app.utils import base_arxiv_id
from app.crawl import (
    get_paper_figure,
    get_citation_count,
    get_citation_count_by_scholar_id,
    enrich_institutions_from_scholar,
    _enrich_one_web_paper,
    start_relevance_reason_async,
    generate_relevance_reason,
    get_paper_content,
    write_summary,
)
from app.llm import LLMUnavailableError

MULTI_TAGS = {"important", "to_read", "to_discuss"}
NOTIFY_MESSAGE_MAX_CHARS = 2000  # maxlength in paper_card.html


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


def _project_paper(arxiv_id):
    paper = Paper.query.filter_by(arxiv_id=arxiv_id).first_or_404()
    project_id = request.values.get("project_id", type=int)
    return ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first_or_404()


def _act(pp, change):
    """Apply a card action and answer with what changed for this person: the card in place, or (when the
    paper left the section it is in for them) the card removed and its new section redrawn.
    In a per_person project a change of the shared section is recorded as this person's move (so it shows up
    in the others' New Papers); any other action only marks the paper as seen by them."""
    project = pp.project
    me = members.current(project)
    before, shared_before = members.section_for(pp, me), members.shared_section(pp)
    change(me)
    if me is not None:
        if members.shared_section(pp) != shared_before:
            members.moved(pp, me)
        else:
            members.seen(pp, me)
    elif not project.per_person and members.shared_section(pp) != shared_before:
        pp.moved_by_id = None  # moved with per-person mode off: nobody's move, everyone saw it
    db.session.commit()
    if not request.headers.get("HX-Request"):
        return ("", 204)
    after = members.section_for(pp, me)
    if after == before:
        return _card_response(pp, project)
    return _moved_response(project.id, after)


def _collect(pp):
    if pp.manual_tag is None:
        pp.manual_tag = "related"
        pp.collected_at = datetime.utcnow()
    if pp.trashed_at:  # collecting a paper someone trashed takes it out of Trash
        pp.trashed_at = None
        pp.trashed_reason = None


@papers_bp.route("/figures/<name>")
@login_required
def figure_file(name):
    """Figure 1 cut from a paper's PDF (app/figures.py). send_from_directory refuses paths outside the folder."""
    return send_from_directory(figures.figures_dir(), name, mimetype="image/png", max_age=24 * 3600)  # a day: Regenerate rewrites the same file


@papers_bp.route("/paper/<arxiv_id>/tag", methods=["POST"])
@login_required
def tag_paper(arxiv_id):
    """tag: one of MULTI_TAGS (toggle; collects the paper), "related" (Move to My Collection, tagged to read)
    or "" (Move to New Papers). Tags are per person in per_person projects."""
    pp = _project_paper(arxiv_id)
    tag = request.form.get("tag")

    def change(me):
        tags = members.tags_for(pp, me)
        if tag in MULTI_TAGS:
            tags = [t for t in tags if t != tag] if tag in tags else tags + [tag]
            _collect(pp)
        elif tag == "related":
            _collect(pp)
            if "to_read" not in tags:
                tags = tags + ["to_read"]
        elif tag == "":
            pp.manual_tag = None
            pp.collected_at = None
            tags = []
        else:
            return
        if me is None:
            pp.paper_tags = json.dumps(tags)
        else:
            members.set_tags(pp, me, tags)

    return _act(pp, change)


@papers_bp.route("/paper/<arxiv_id>/seen", methods=["POST"])
@login_required
def seen_paper(arxiv_id):
    """Someone else moved this paper (collected or trashed it) and this person agrees: it leaves their
    New Papers for the section it is in."""
    return _act(_project_paper(arxiv_id), lambda me: None)


@papers_bp.route("/paper/<arxiv_id>/vote", methods=["POST"])
@login_required
def vote_paper(arxiv_id):
    """1-5; the same vote again takes it back. Everyone sees everyone's votes."""
    pp = _project_paper(arxiv_id)
    me = members.current(pp.project)
    vote = request.form.get("vote", type=int)
    if me is None or vote not in range(1, 6):
        return ("pick who you are first", 400)
    st = members.state(pp, me, create=True)
    st.vote = None if st.vote == vote else vote
    db.session.commit()
    return _card_response(pp, pp.project)


@papers_bp.route("/paper/<arxiv_id>/notes-runs", methods=["POST"])
@login_required
def update_notes_runs(arxiv_id):
    """Per-person projects: the whole note as runs [[member id or null, text, day or null], ...] (who wrote what,
    and on which day; the day is the writer's browser's, "YYYY-MM-DD").
    ProjectPaper.notes keeps the plain text (export, Copy / Move, the paper chat)."""
    pp = _project_paper(arxiv_id)
    ids = {m.id for m in members.members(pp.project)}
    runs = json.loads(request.form.get("runs", "[]"))
    if not all(isinstance(r, list) and len(r) in (2, 3) and (r[0] is None or r[0] in ids) and isinstance(r[1], str)
               and (len(r) == 2 or r[2] is None or (isinstance(r[2], str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", r[2])))
               for r in runs):
        return ("bad runs", 400)
    if _note_is_stale("".join(r[1] for r in runs), pp):
        return jsonify(runs=members.runs(pp)), 409
    runs = members.merge_runs(runs)
    pp.notes_runs = json.dumps(runs)
    pp.notes = "".join(r[1] for r in runs)
    db.session.commit()
    return ("", 204)


def _same_note(a, b) -> bool:
    # A textarea hands back \n for \r\n and drops a leading newline
    return (a or "").replace("\r\n", "\n").strip() == (b or "").replace("\r\n", "\n").strip()


def _note_is_stale(text, pp) -> bool:
    """The save started from a note that was changed since. A resend of what is already saved (the answer to the
    first try got lost) is fine."""
    base = request.form.get("base")
    return base is not None and not _same_note(base, pp.notes) and not _same_note(text, pp.notes)


@papers_bp.route("/paper/<arxiv_id>/note-lock", methods=["POST"])
@login_required
def note_lock(arxiv_id):
    """Claim, renew or (release=1) give back the note. Answers who else has it: by is null if nobody does."""
    pp = _project_paper(arxiv_id)
    client = request.form["client"]
    if request.form.get("release"):
        note_locks.release(pp.project_id, pp.paper_id, client)
        return jsonify(by=None)
    me = members.current(pp.project)
    return jsonify(by=note_locks.claim(pp.project_id, pp.paper_id, client, me.name if me else ""))


@papers_bp.route("/paper/<arxiv_id>/note-state")
@login_required
def note_state(arxiv_id):
    """The open card asks every few seconds: is someone writing, and what does the note say now."""
    pp = _project_paper(arxiv_id)
    return jsonify(by=note_locks.holder(pp.project_id, pp.paper_id, request.args["client"]),
                   notes=pp.notes or "", runs=members.runs(pp))


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


@papers_bp.route("/papers/titles")
@login_required
def paper_titles():
    """{paper id: title} for the paper links in notes, which show a paper of this app by its title
    (notes_format.js). At most 50 ids per request."""
    ids = [int(i) for i in request.args.get("ids", "").split(",") if i.strip().isdigit()][:50]
    rows = Paper.query.with_entities(Paper.id, Paper.title).filter(Paper.id.in_(ids)).all() if ids else []
    return jsonify({str(pid): title for pid, title in rows})


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
    notes_height = request.form.get("notes_height", type=int)

    if project_id:
        pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first()
        if pp:
            if "notes" in request.form:  # a resize sends only the height
                if _note_is_stale(request.form["notes"], pp):
                    return jsonify(notes=pp.notes or ""), 409
                pp.notes = request.form["notes"]
            if notes_height:
                pp.notes_height = notes_height
            db.session.commit()

    return ("", 204)


@papers_bp.route("/paper/<arxiv_id>/trash", methods=["POST"])
@login_required
def trash_paper(arxiv_id):
    pp = _project_paper(arxiv_id)

    def change(me):
        pp.trashed_at = datetime.utcnow()

    return _act(pp, change)


@papers_bp.route("/paper/<arxiv_id>/restore", methods=["POST"])
@login_required
def restore_paper(arxiv_id):
    pp = _project_paper(arxiv_id)

    def change(me):
        pp.trashed_at = None
        pp.trashed_reason = None
        pp.manual_tag = None
        pp.collected_at = None

    return _act(pp, change)


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
    target.trashed_reason = None
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

    with counted(project_id, "regenerate"):
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

            # Version, comment and journal ref (the card's Version and Venue)
            try:
                meta = arxiv_meta._fetch([paper.arxiv_id]).get(paper.arxiv_id)
                if meta:
                    arxiv_meta.apply(paper, meta)
            except Exception as e:
                _log.warning("[regenerate] arXiv data fetch failed for %s: %s", arxiv_id, e)

        # Summary + main contributions, from the full text like a crawl (abstract if there is none)
        full_text = get_paper_content(arxiv_id) if paper.is_arxiv else None
        content = full_text or paper.abstract or ""
        try:
            write_summary(paper, content)
            if full_text:  # code, project page, corresponding authors, venue
                resources.apply(paper, resources.extract_for(paper, full_text)[0])
        except LLMUnavailableError as e:
            _log.warning("[regenerate] LLM unavailable for %s: %s", arxiv_id, e)

        # This project's why-relevant line (the summary above is shared by all projects)
        pp = ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper.id).first() if project_id else None
        if pp and project and (project.research_interest or "").strip():
            try:
                why = generate_relevance_reason(paper.title, paper.abstract or "", project.research_interest, full_text)
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
        with _app.app_context(), counted(None, "metadata"):
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
                    note_parts=members.note_parts(pp) if pp else [],
                    project=project,
                    flagged_by=getattr(members.current(project), "name", None),
                    message=request.form.get("message", "").strip()[:NOTIFY_MESSAGE_MAX_CHARS],
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
        url = p.landing_url  # same link as the card title
        writer.writerow([p.title or "", authors, url, pp.manual_tag or "untagged", p.abstract or ""])

    csv_bytes = buf.getvalue().encode("utf-8")
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={slug}-papers.csv"},
    )
