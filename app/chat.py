"""The "Ask about this paper" chat on a paper card.

One conversation per paper per project (ChatMessage rows on a ProjectPaper, like notes), kept until
someone clears it. Every turn sends the model the paper's text (fetched once and kept in PaperText),
the project's research interest, and the earlier questions and answers (never the model's thinking:
reasoning models are trained to see only earlier final answers).

Answers are written by one background worker thread per chat, which works through the questions in
order: a question asked while an answer is still coming is stored as "queued" and answered next. The
worker streams from the model; the visible part (nothing while the model thinks, then the answer as it
grows) lives in _live, which the card polls via /chat/message/<id>. Closing the page doesn't matter: the
worker keeps going and the card shows the result when it is opened again. A server restart ends the
worker: the answer it was writing is marked interrupted and the queue resumes when the chat is next shown.
All of this needs the single gunicorn process the app already relies on (see CLAUDE.md).

The worker never holds a SQLite write transaction while it streams, and writes its results with
UPDATE ... WHERE id, so a chat cleared mid-answer just makes those writes no-ops.
"""

import io
import logging
import os
import re
import threading

import httpx
from markupsafe import Markup
from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

from app import db, prompts
from app.llm import MAX_TOKENS_CEILING, MODELS, LLMUnavailableError, _llm_stream
from app.models import ChatMessage, PaperText, ProjectPaper
from app.usage import counted
from app.utils import _ArxivLockCtx, _http_client, fetch_paper_full_text

_log = logging.getLogger(__name__)

# The first model's context window: set CHAT_CONTEXT_TOKENS to what your endpoint reports (a vLLM server
# names it in the "maximum context length" error of a too-long request).
CONTEXT_LIMIT = int(os.environ.get("CHAT_CONTEXT_TOKENS", 196000))
MAX_ANSWER_TOKENS = MAX_TOKENS_CEILING  # thinking + answer; the server counts it against the context
MAX_PAPER_CHARS = 150_000  # ~40k tokens, the same cap the deep relevance check uses
MIN_TEXT_CHARS = 2000  # anything shorter is an error page or a stub, not the paper
MAX_QUESTION_CHARS = 20_000

# Label on the chat and the tooltip explaining it
SOURCES = {
    "html": ("Full text (arXiv HTML)",
             "The model reads the whole paper, from arXiv's HTML version. Figures are not included; "
             "tables and math can come through garbled."),
    "pdf": ("Full text (PDF)",
            "There is no HTML version, so the model reads text extracted from the PDF. Figures are not "
            "included; layout, tables and math can come through garbled."),
    "abstract": ("Abstract only",
                 "No full text could be fetched (no arXiv HTML or PDF, the download failed, or the PDF is on "
                 "a site outside the open-access allowlist), so the model sees only the title and abstract. "
                 "The download is tried again with the next question."),
}

CONTEXT_FULL = "The chat has used up the model's context. Clear the chat to start a new one."

# assistant message id -> {"phase": "reading"|"thinking"|"answering", "text": str, "stop": bool}
_live: dict[int, dict] = {}
_running: set[int] = set()  # ProjectPaper ids whose worker is alive
_lock = threading.Lock()  # guards _running and taking the next question off a queue

# $...$ / $$...$$ become <span|div class="math ..."> with the TeX left as text, which KaTeX renders in the
# page (base.html); "$5 vs $6" stays money (no space or digit right inside/after the dollars)
_md = (MarkdownIt("commonmark", {"html": False, "breaks": True}).enable("table")
       .use(dollarmath_plugin, allow_space=False, allow_digits=False, double_inline=True))


def render_markdown(text: str) -> Markup:
    """Model answers as HTML. Raw HTML in the answer is escaped (html=False), and markdown-it refuses
    javascript:/data: links. \\( \\) and \\[ \\] math (models use both styles) is turned into dollars first."""
    text = re.sub(r"\\\((.+?)\\\)", lambda m: f"${m.group(1).strip()}$", text or "")
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: f"$${m.group(1).strip()}$$", text, flags=re.S)
    return Markup(_md.render(text))


# ── Paper text ───────────────────────────────────────────────────────────────

def _pdf_text(url: str, arxiv: bool) -> str | None:
    from pypdf import PdfReader

    def _get():
        with _http_client(timeout=httpx.Timeout(60.0), follow_redirects=True) as http:
            return http.get(url)

    try:
        if arxiv:
            with _ArxivLockCtx():
                r = _get()
        else:
            r = _get()
    except httpx.HTTPError as e:  # the chat then says "abstract only" and tries again next question
        _log.warning("[chat] PDF download failed for %s: %s", url, e)
        return None
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        _log.info("[chat] no PDF at %s (HTTP %s)", url, r.status_code)
        return None
    reader = PdfReader(io.BytesIO(r.content), strict=False)
    text = " ".join(page.extract_text() or "" for page in reader.pages)
    return re.sub(r"\s+", " ", text).strip()[:MAX_PAPER_CHARS]


def _fetch_text(paper) -> tuple[str | None, str | None]:
    from app.crawl import _is_open_access_pdf_domain

    if paper.is_arxiv:
        text = fetch_paper_full_text(paper.arxiv_id, max_chars=MAX_PAPER_CHARS)
        if text and len(text) >= MIN_TEXT_CHARS:
            return text, "html"
        text = _pdf_text(f"https://arxiv.org/pdf/{paper.arxiv_id}", arxiv=True)
        if text and len(text) >= MIN_TEXT_CHARS:
            return text, "pdf"
    elif paper.pdf_url and _is_open_access_pdf_domain(paper.pdf_url):
        text = _pdf_text(paper.pdf_url, arxiv=False)
        if text and len(text) >= MIN_TEXT_CHARS:
            return text, "pdf"
    return None, None


def _nothing_to_download(paper) -> bool:
    """A web paper without an open-access PDF: the chat is abstract-only before anyone asks."""
    from app.crawl import _is_open_access_pdf_domain
    return not paper.is_arxiv and not (paper.pdf_url and _is_open_access_pdf_domain(paper.pdf_url))


def paper_text(paper) -> tuple[str, str]:
    """(text, source) for the chat: the stored full text, fetched on first use. Falls back to
    ("", "abstract") without storing anything, so a later question tries the download again."""
    stored = PaperText.query.filter_by(paper_id=paper.id).first()
    if stored:
        return stored.text, stored.source
    if _nothing_to_download(paper):
        return "", "abstract"
    text, source = _fetch_text(paper)
    if not text:
        return "", "abstract"
    db.session.add(PaperText(paper_id=paper.id, text=text, source=source))
    db.session.commit()
    return text, source


# ── Conversation ─────────────────────────────────────────────────────────────

def ordered(msgs: list) -> list:
    """Messages in conversation order: each answer right after its question (an answer row is created
    only when its question is sent, so ids alone would put it after questions queued meanwhile)."""
    return sorted(msgs, key=lambda m: (m.reply_to_id, 1) if m.role == "assistant" and m.reply_to_id else (m.id, 0))


def _answered_pairs(msgs: list, before_id: int) -> list[tuple]:
    """(question, answer) for every question before `before_id` whose answer can go back to the model:
    finished ones, and stopped ones that got some text out."""
    questions = {m.id: m for m in msgs if m.role == "user"}
    pairs = [(questions[a.reply_to_id], a) for a in msgs
             if a.role == "assistant" and a.reply_to_id in questions and a.reply_to_id < before_id
             and (a.status == "done" or (a.status == "stopped" and a.content.strip()))]
    return sorted(pairs, key=lambda qa: qa[0].id)


def build_messages(pp, text: str, source: str, question) -> list[dict]:
    """System message with the paper, then every earlier answered turn, then the new question."""
    paper, project = pp.paper, pp.project
    system = prompts.PAPER_CHAT_SYSTEM.format(
        research_interest=(project.research_interest or "(not set)").strip(),
        keywords_line=f"\nSearch keywords of the project: {', '.join(project.saved_keywords_list)}\n"
        if project.saved_keywords_list else "",
        notes_line=f"\nThe researcher's notes on this paper: {pp.notes.strip()}\n" if (pp.notes or "").strip() else "",
        relevance_line=f"Why this paper was flagged for the project: {pp.relevance_reason}\n" if pp.relevance_reason else "",
        title=paper.title,
        authors=", ".join(paper.authors_list) or "unknown",
        source_note=prompts.PAPER_CHAT_SOURCE_NOTES[source],
        text=text if source != "abstract" else (paper.abstract or "(no abstract)"),
    )
    messages = [{"role": "system", "content": system}]
    msgs = ChatMessage.query.filter_by(project_paper_id=pp.id).all()
    for q, a in _answered_pairs(msgs, question.id):
        answer = a.content if a.status == "done" else a.content + "\n\n[The user stopped this answer here.]"
        messages += [{"role": "user", "content": q.content}, {"role": "assistant", "content": answer}]
    messages.append({"role": "user", "content": question.content})
    return messages


def _context_used(msgs: list) -> int | None:
    """Tokens the conversation fills, as the model counted them for the latest finished answer."""
    counted = [m for m in msgs if m.role == "assistant" and m.context_tokens]
    return max(counted, key=lambda m: m.id).context_tokens if counted else None


def _would_overflow(used: int | None, question: str) -> bool:
    return used is not None and used + len(question) // 3 + MAX_ANSWER_TOKENS > CONTEXT_LIMIT


def view(pp, app=None) -> dict:
    """What the chat panel shows: messages, which text the model reads, context usage, and whether
    a question is being answered. Also repairs what a server restart left behind (see module doc)."""
    msgs = ChatMessage.query.filter_by(project_paper_id=pp.id).all()
    stale = [m for m in msgs if m.status == "streaming" and m.id not in _live]
    if stale and pp.id not in _running:
        db.session.commit()  # fresh snapshot for the write; the status filter skips one finished meanwhile
        ChatMessage.query.filter(ChatMessage.id.in_([m.id for m in stale]), ChatMessage.status == "streaming").update(
            {"status": "error", "content": ChatMessage.content + "\n\n*(Interrupted: the server restarted.)*"},
            synchronize_session=False)
        db.session.commit()
        msgs = ChatMessage.query.filter_by(project_paper_id=pp.id).all()
    if app is not None and pp.id not in _running and any(m.status == "queued" for m in msgs):
        _kick(app, pp.id)  # a queue left over from before a restart
        msgs = ChatMessage.query.filter_by(project_paper_id=pp.id).all()
    used = _context_used(msgs)
    sources = [m for m in msgs if m.context_source]
    stored = PaperText.query.filter_by(paper_id=pp.paper_id).first()
    if sources:
        source = max(sources, key=lambda m: m.id).context_source
    elif stored:
        source = stored.source
    else:
        source = "abstract" if _nothing_to_download(pp.paper) else None
    streaming = [m for m in msgs if m.status == "streaming"]
    ctx = live_context(streaming[0].id) if streaming else None
    return {
        "ctx": ctx or _ctx(used, approx=False),
        "messages": ordered(msgs), "busy": bool(streaming),
        "queued": sum(m.status == "queued" for m in msgs),
        "full": _ctx(used, approx=False)["full"],
        "source": source, "source_label": SOURCES[source][0] if source else None,
        "source_tip": SOURCES[source][1] if source else None,
    }


CHARS_PER_TOKEN = 3.2  # measured on our model: 134.5k chars of a paper's text -> 42k prompt tokens


def _ctx(used: int | None, approx: bool) -> dict:
    return {"used": used, "limit": CONTEXT_LIMIT, "approx": approx,
            "pct": min(100, round(100 * (used or 0) / CONTEXT_LIMIT)),
            "full": used is not None and used + MAX_ANSWER_TOKENS > CONTEXT_LIMIT}


def live_context(answer_id: int) -> dict | None:
    """Context usage while an answer streams: the prompt (counted by the model for the previous answer,
    else estimated from its length) plus the answer so far. None until the prompt is known."""
    live = _live.get(answer_id)
    if not live or live.get("prompt_tokens") is None:
        return None
    return _ctx(live["prompt_tokens"] + round(len(live["text"]) / CHARS_PER_TOKEN), approx=True)


def ask(pp, question: str, app) -> str | None:
    """Queue a question; it is sent to the model now if nothing is being answered, else after the
    questions before it. Returns an error to show instead, if it can't be queued."""
    if len(question) > MAX_QUESTION_CHARS:
        return f"That question is too long ({len(question):,} characters, at most {MAX_QUESTION_CHARS:,})."
    if _would_overflow(_context_used(ChatMessage.query.filter_by(project_paper_id=pp.id).all()), question):
        return CONTEXT_FULL
    db.session.add(ChatMessage(project_paper_id=pp.id, role="user", content=question, status="queued"))
    db.session.commit()
    _kick(app, pp.id)
    return None


def _take_next(pp_id: int) -> int | None:
    """Under _lock: move the oldest queued question to "sent", create its (streaming) answer, and
    return the answer's id; or, with nothing queued, mark the worker finished."""
    q = (ChatMessage.query.filter_by(project_paper_id=pp_id, role="user", status="queued")
         .order_by(ChatMessage.id).first())
    if q is None:
        _running.discard(pp_id)
        return None
    q.status = "done"
    a = ChatMessage(project_paper_id=pp_id, role="assistant", content="", status="streaming", reply_to_id=q.id)
    db.session.add(a)
    db.session.flush()
    _live[a.id] = {"phase": "reading", "text": "", "stop": False}  # before commit: never seen as stale
    db.session.commit()
    return a.id


def _kick(app, pp_id: int) -> None:
    """Start the chat's worker unless it is running. The first question is taken off the queue here,
    in the caller's request, so the page it renders already shows the answer being written."""
    with _lock:
        if pp_id in _running:
            return
        _running.add(pp_id)
        answer_id = _take_next(pp_id)
    if answer_id is not None:
        threading.Thread(target=_work, args=(app, pp_id, answer_id), daemon=True).start()


def _work(app, pp_id: int, answer_id: int) -> None:
    with app.app_context():
        project_id = db.session.get(ProjectPaper, pp_id).project_id
        try:
            while answer_id is not None:
                try:
                    with counted(project_id, "chat"):
                        _answer(answer_id)
                except Exception as e:  # never leave an answer "streaming" with nothing writing it
                    _log.exception("[chat] answer %d: worker failed", answer_id)
                    db.session.rollback()
                    _finish(answer_id, status="error", content=f"*(Error: {str(e)[:300]})*")
                db.session.expire_all()
                with _lock:
                    # take the next question before this answer leaves _live, so a page polling it
                    # finds the next answer already started when it re-renders the conversation
                    next_id = _take_next(pp_id)
                    _live.pop(answer_id, None)
                answer_id = next_id
        finally:
            with _lock:
                _running.discard(pp_id)
                if answer_id is not None:
                    _live.pop(answer_id, None)


def _finish(answer_id: int, **values) -> None:
    ChatMessage.query.filter_by(id=answer_id).update(values, synchronize_session=False)
    db.session.commit()


def _answer(answer_id: int) -> None:
    live = _live[answer_id]
    msg = ChatMessage.query.get(answer_id)
    if msg is None:  # cleared before it started
        return
    question = ChatMessage.query.get(msg.reply_to_id)
    pp = ProjectPaper.query.get(msg.project_paper_id)
    if question is None or pp is None:  # chat cleared (or paper removed) in between
        return
    try:
        used = _context_used(ChatMessage.query.filter_by(project_paper_id=pp.id).all())
        if _would_overflow(used, question.content):
            _finish(answer_id, status="error", content=f"*({CONTEXT_FULL})*")
            return
        text, source = paper_text(pp.paper)
        _finish(answer_id, context_source=source)
        if live["stop"]:
            _finish(answer_id, status="stopped")
            return
        messages = build_messages(pp, text, source, question)
        db.session.commit()  # end the read transaction: nothing open on SQLite while the answer streams
        live["prompt_tokens"] = (used + round(len(question.content) / CHARS_PER_TOKEN) if used is not None
                                 else round(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN))
        live["phase"] = "thinking"
        final = None
        stream = _llm_stream(messages, temperature=0.3, max_tokens=MAX_ANSWER_TOKENS)
        for event in stream:
            if live["stop"]:
                stream.close()  # closes the HTTP stream, so the server stops generating too
                break
            if isinstance(event, dict):
                final = event
            elif isinstance(event, tuple):  # ("retry", model, attempt)
                live["phase"], live["retry"] = "retrying", f"{event[1]}, attempt {event[2]}"
            elif event:
                live["phase"], live["text"] = "answering", event
            elif live["phase"] == "retrying":  # a retry got through and the model is thinking
                live["phase"] = "thinking"
        answer = live["text"]
        if final is None:
            _finish(answer_id, status="stopped", content=answer)
            _log.info("[chat] answer %d stopped by the user after %d chars", answer_id, len(answer))
            return
        if final["finish_reason"] == "length":
            answer += "\n\n*(Cut off: the answer hit the output limit.)*"
        # The answer's own tokens are estimated: completion_tokens also counts the thinking, which is
        # never sent back
        _finish(answer_id, status="done", content=answer, model=final["model"],
                context_tokens=final["usage"]["prompt_tokens"] + round(len(answer) / CHARS_PER_TOKEN))
        _log.info("[chat] answer %d done (%s, %d chars, context %d)", answer_id, final["model"], len(answer),
                  final["usage"]["prompt_tokens"])
    except Exception as e:  # the card must stop waiting and say what went wrong, not spin forever
        _log.exception("[chat] answer %d failed", answer_id)
        db.session.rollback()
        partial = live["text"] + "\n\n" if live["text"] else ""
        if isinstance(e, LLMUnavailableError) and not live["text"]:
            msg = (f"The model is not available right now ({', '.join(MODELS)} all failed). "
                   f"Try again later; your question is kept above. Last error: {str(e)[:200]}")
        else:
            msg = f"Error: {str(e)[:300]}"
        _finish(answer_id, status="error", content=f"{partial}*({msg})*")


def stop(answer_id: int) -> bool:
    live = _live.get(answer_id)
    if live:
        live["stop"] = True
    return bool(live)


def remove_queued(question_id: int) -> bool:
    """Take a question off the queue; False if it was already sent (or doesn't exist)."""
    with _lock:
        n = ChatMessage.query.filter_by(id=question_id, role="user", status="queued").delete()
        db.session.commit()
    return bool(n)


def clear(pp) -> None:
    """Delete the whole chat; an answer being written is stopped (its final write then hits no row)."""
    with _lock:
        msgs = ChatMessage.query.filter_by(project_paper_id=pp.id).all()
        for m in msgs:
            stop(m.id)
        ChatMessage.query.filter_by(project_paper_id=pp.id).delete()
        db.session.commit()


def copy_chat(source, target) -> int:
    """Copy / Move of a paper to another project: its chat comes along like its notes, and only if the
    target has no chat of its own. Only finished turns are copied (an answer still being written and
    queued questions stay behind). Caller commits. Returns the number of messages copied."""
    if ChatMessage.query.filter_by(project_paper_id=target.id).count():
        return 0
    msgs = ChatMessage.query.filter_by(project_paper_id=source.id).all()
    answers = {m.reply_to_id: m for m in msgs if m.role == "assistant" and m.status != "streaming"}
    copied = 0
    for q in sorted((m for m in msgs if m.role == "user" and m.id in answers), key=lambda m: m.id):
        a = answers[q.id]
        new_q = ChatMessage(project_paper_id=target.id, role="user", content=q.content, status="done",
                            created_at=q.created_at)
        db.session.add(new_q)
        db.session.flush()
        db.session.add(ChatMessage(project_paper_id=target.id, role="assistant", content=a.content, status=a.status,
                                   reply_to_id=new_q.id, context_source=a.context_source, model=a.model,
                                   context_tokens=a.context_tokens, created_at=a.created_at))
        copied += 2
    return copied


def live_state(msg_id: int) -> dict | None:
    return _live.get(msg_id)


def init_app(app) -> None:
    app.jinja_env.filters["chat_md"] = render_markdown
    app.jinja_env.globals["chat_live"] = live_state
    app.jinja_env.globals["chat_view"] = lambda pp: view(pp, app)
    app.jinja_env.globals["chat_main_model"] = MODELS[0]
