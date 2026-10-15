"""
"Ask about this paper" chat on a card (app/chat.py).

- The answer streams in without the model's thinking; the conversation is saved and sent back as context.
- The paper text is fetched once: HTML, else PDF, else abstract only, which the chat says.
- An answer keeps going with nobody watching (page closed), and a restart leaves no stuck "Thinking...".
- Questions sent meanwhile are queued and answered in order. Stop keeps the partial answer, and
  clearing the chat mid-answer is safe.
- A full context window blocks new questions instead of silently dropping old turns.

The model is a scripted fake (FakeModel). Nothing here calls the real LLM or arXiv.
"""

import contextlib
import json
import re
import threading
import time

import httpx
import pytest

from app import chat, db
from app import llm
from app.models import ChatMessage, Paper, PaperText, Project, ProjectPaper

PAPER_TEXT = "Section 1 Introduction. " * 200  # > MIN_TEXT_CHARS
ARXIV = "2609.11111"


# ── Fixtures and helpers ─────────────────────────────────────────────────────

@pytest.fixture
def chat_pp(app):
    """A project with one arXiv paper in My Collection and one web paper without an open-access PDF."""
    with app.app_context():
        project = Project(name="Chat Proj", slug="chat-proj", research_interest="sparse attention")
        paper = Paper(arxiv_id=ARXIV, title="Sparse Things", source="arxiv", abstract="We sparsify things.",
                      authors=json.dumps(["A. Author"]))
        web = Paper(arxiv_id="web:chat-blog", title="A Blog", source="web", abstract="Blog abstract.",
                    pdf_url="https://example.com/post")
        db.session.add_all([project, paper, web])
        db.session.flush()
        pp = ProjectPaper(project_id=project.id, paper_id=paper.id, manual_tag="related",
                          relevance_reason="It sparsifies attention.")
        wpp = ProjectPaper(project_id=project.id, paper_id=web.id, manual_tag="related")
        db.session.add_all([pp, wpp])
        db.session.commit()
        ids = {"project": project.id, "paper": paper.id, "pp": pp.id, "web": web.id, "wpp": wpp.id}
    yield ids
    for pp_id in (ids["pp"], ids["wpp"]):
        _idle(pp_id)
    with app.app_context():
        ChatMessage.query.filter(ChatMessage.project_paper_id.in_([ids["pp"], ids["wpp"]])).delete()
        PaperText.query.filter(PaperText.paper_id.in_([ids["paper"], ids["web"]])).delete()
        ProjectPaper.query.filter(ProjectPaper.id.in_([ids["pp"], ids["wpp"]])).delete()
        Paper.query.filter(Paper.id.in_([ids["paper"], ids["web"]])).delete()
        Project.query.filter_by(id=ids["project"]).delete()
        db.session.commit()


@pytest.fixture
def c(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    return client


@pytest.fixture
def full_text(monkeypatch):
    """arXiv HTML is available; counts fetches."""
    fetches = []
    monkeypatch.setattr(chat, "fetch_paper_full_text", lambda arxiv_id, max_chars: fetches.append(arxiv_id) or PAPER_TEXT)
    return fetches


class FakeModel:
    """Stands in for llm._llm_stream. Each call plays the next script: a str is yielded (the visible
    answer so far, "" while thinking), WAIT blocks until the test calls release(), and an Exception is
    raised. Records the messages of every call and which calls were closed early (Stop)."""
    WAIT = object()

    def __init__(self, *scripts, prompt_tokens=42000, finish_reason="stop"):
        self.scripts = list(scripts)
        self.calls, self.closed = [], []
        self.prompt_tokens, self.finish_reason = prompt_tokens, finish_reason
        self._gate = threading.Semaphore(0)
        self.waiting = threading.Event()

    def release(self):
        self._gate.release()

    def __call__(self, messages, temperature, max_tokens):
        self.calls.append(messages)
        n = len(self.calls)
        script = self.scripts.pop(0) if self.scripts else ["Ok."]
        try:
            for step in script:
                if step is FakeModel.WAIT:
                    self.waiting.set()
                    assert self._gate.acquire(timeout=5), "test never released the model"
                    self.waiting.clear()
                elif isinstance(step, Exception):
                    raise step
                else:
                    yield step
            yield {"model": llm.MODELS[0], "usage": {"prompt_tokens": self.prompt_tokens},
                   "finish_reason": self.finish_reason}
        except GeneratorExit:
            self.closed.append(n)
            raise


def _wait(cond, what="condition", timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def _idle(pp_id):
    _wait(lambda: pp_id not in chat._running, "worker to end")


def _ask(c, ids, question, pp="pp"):
    arxiv = ARXIV if pp == "pp" else "web:chat-blog"
    return c.post(f"/paper/{arxiv}/chat", data={"project_id": ids["project"], "question": question}).data.decode()


def _msgs(app, pp_id):
    with app.app_context():
        return [(m.role, m.status, m.content) for m in chat.ordered(ChatMessage.query.filter_by(project_paper_id=pp_id).all())]


def _latest_answer_id(app, pp_id):
    with app.app_context():
        return ChatMessage.query.filter_by(project_paper_id=pp_id, role="assistant").order_by(ChatMessage.id.desc()).first().id


# ── llm._llm_stream ──────────────────────────────────────────────────────────

def test_visible_answer_hides_the_thinking_block():
    assert llm._visible_answer("<thi") == ""
    assert llm._visible_answer("<think>hmm, the user") == ""
    assert llm._visible_answer("<think>hmm</think>\n\nThe answer") == "The answer"
    assert llm._visible_answer("No thinking at all") == "No thinking at all"


def _sse(events, usage=None, finish="stop"):
    body = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': e}, 'finish_reason': None}]})}\n\n" for e in events)
    body += f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': finish}]})}\n\n"
    if usage:
        body += f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
    return body + "data: [DONE]\n\n"


def _mock_http(monkeypatch, handler):
    real_client = httpx.Client
    monkeypatch.setattr(llm.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def test_llm_stream_yields_only_the_answer_and_the_usage(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, text=_sse(["<think>", "let me see", "</think>", "\n\nIt ", "works."],
                                             usage={"prompt_tokens": 10, "completion_tokens": 9}))

    _mock_http(monkeypatch, handler)
    out = list(llm._llm_stream([{"role": "user", "content": "hi"}]))
    assert out[:3] == ["", "", ""] and out[-2] == "It works."
    assert out[-1] == {"model": llm.MODELS[0], "usage": {"prompt_tokens": 10, "completion_tokens": 9}, "finish_reason": "stop"}
    assert seen["stream"] is True and seen["stream_options"] == {"include_usage": True}


def test_llm_stream_falls_back_to_the_next_model_before_the_first_chunk(monkeypatch):
    models = []

    def handler(request):
        model = json.loads(request.content)["model"]
        models.append(model)
        if model == llm.MODELS[0]:
            return httpx.Response(400, json={"error": {"message": "maximum context length is 196000 tokens"}})
        return httpx.Response(200, text=_sse(["Fine."]))

    _mock_http(monkeypatch, handler)
    out = list(llm._llm_stream([{"role": "user", "content": "hi"}]))
    assert models == [llm.MODELS[0], llm.MODELS[1]]  # a 400 (too long) is not retried on the same model
    assert out[-2] == "Fine." and out[-1]["model"] == llm.MODELS[1]


def test_llm_stream_gives_up_when_every_model_fails(monkeypatch):
    _mock_http(monkeypatch, lambda request: httpx.Response(503, text="down"))
    with pytest.raises(llm.LLMUnavailableError):
        list(llm._llm_stream([{"role": "user", "content": "hi"}]))


# ── Panel ────────────────────────────────────────────────────────────────────

def test_card_has_an_ask_button_until_there_is_a_chat(c, chat_pp):
    html = c.get("/projects/chat-proj").data.decode()
    assert "Ask about this paper" in html and '<div class="chat-panel">' not in html
    html = c.get(f"/paper/{ARXIV}/chat?project_id={chat_pp['project']}").data.decode()
    assert '<div class="chat-panel">' in html and "visible to everyone on this project" in html
    assert "Full text on first question" in html  # not fetched yet


def test_web_paper_without_open_access_pdf_says_abstract_only_up_front(c, chat_pp):
    html = c.get(f"/paper/web:chat-blog/chat?project_id={chat_pp['project']}").data.decode()
    assert "Abstract only" in html and "chat-badge-warn" in html


def test_empty_question_does_nothing(c, app, chat_pp):
    _ask(c, chat_pp, "   ")
    assert _msgs(app, chat_pp["pp"]) == []


def test_too_long_question_is_refused(c, app, chat_pp):
    html = _ask(c, chat_pp, "x" * (chat.MAX_QUESTION_CHARS + 1))
    assert "too long" in html and _msgs(app, chat_pp["pp"]) == []


def test_unknown_project_or_paper_is_404(c, chat_pp):
    assert c.post(f"/paper/{ARXIV}/chat", data={"project_id": 99999, "question": "q"}).status_code == 404
    assert c.post("/paper/0000.00000/chat", data={"project_id": chat_pp["project"], "question": "q"}).status_code == 404


def test_chat_routes_need_login(client, chat_pp):
    assert client.get("/chat/message/1").status_code in (302, 401)
    assert client.post(f"/paper/{ARXIV}/chat", data={"project_id": chat_pp["project"], "question": "q"}).status_code in (302, 401)


def test_answer_markdown_escapes_html():
    out = str(chat.render_markdown("**bold** <script>alert(1)</script> [x](javascript:alert(1))"))
    assert "<strong>bold</strong>" in out and "<script>" not in out and 'href="javascript' not in out


# ── Answering ────────────────────────────────────────────────────────────────

def test_answer_streams_without_thinking_then_is_saved(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["", FakeModel.WAIT, "Sparse ", FakeModel.WAIT, "Sparse attention helps."])
    monkeypatch.setattr(chat, "_llm_stream", model)

    html = _ask(c, chat_pp, "Why sparse?")
    answer_id = _latest_answer_id(app, chat_pp["pp"])
    assert "Why sparse?" in html and f'hx-get="/chat/message/{answer_id}"' in html and ">Stop<" in html
    assert not re.search(r'class="chat-input"[^>]*\sdisabled', html)  # stays usable: more questions queue

    _wait(model.waiting.is_set, "model to think")
    html = c.get(f"/chat/message/{answer_id}").data.decode()
    assert "Thinking" in html and 'class="chat-poll"' in html
    model.release()
    _wait(lambda: chat.live_state(answer_id)["text"] == "Sparse ", "partial text")
    html = c.get(f"/chat/message/{answer_id}").data.decode()
    assert "Sparse" in html and "Thinking" not in html
    model.release()
    _idle(chat_pp["pp"])

    resp = c.get(f"/chat/message/{answer_id}")
    html = resp.data.decode()
    assert resp.headers["HX-Retarget"] == f"#chat-log-{chat_pp['pp']}"  # the log, not the form (drafts survive)
    assert "chat-form" not in html and 'id="chat-head-' in html and 'hx-swap-oob="true"' in html
    assert "Sparse attention helps." in html and "Full text (arXiv HTML)" in html and "Context 42.0k / 196k" in html
    assert 'class="chat-poll"' not in html  # polling ends
    assert _msgs(app, chat_pp["pp"]) == [("user", "done", "Why sparse?"), ("assistant", "done", "Sparse attention helps.")]

    system = model.calls[0][0]["content"]
    assert PAPER_TEXT[:100] in system and "sparse attention" in system and "It sparsifies attention." in system


def test_answer_keeps_going_with_the_page_closed(c, app, chat_pp, monkeypatch, full_text):
    """Nobody polls (the page was closed right after sending); the answer is there when the page is opened again."""
    monkeypatch.setattr(chat, "_llm_stream", FakeModel(["", "Done without anyone watching."]))
    _ask(c, chat_pp, "Q?")
    _idle(chat_pp["pp"])
    html = c.get("/projects/chat-proj").data.decode()
    assert '<div class="chat-panel">' in html and "Done without anyone watching." in html and 'class="chat-poll"' not in html


def test_reopening_the_page_mid_answer_resumes_polling(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["", FakeModel.WAIT, "Back again."])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q?")
    _wait(model.waiting.is_set, "model to think")
    html = c.get("/projects/chat-proj").data.decode()
    answer_id = _latest_answer_id(app, chat_pp["pp"])
    assert f'hx-get="/chat/message/{answer_id}"' in html and "Thinking" in html
    model.release()
    _idle(chat_pp["pp"])
    assert "Back again." in c.get(f"/chat/message/{answer_id}").data.decode()


def test_history_goes_back_as_context_and_the_text_is_fetched_once(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["", "First answer."], ["Second answer."])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "First?")
    _idle(chat_pp["pp"])
    _ask(c, chat_pp, "Second?")
    _idle(chat_pp["pp"])
    assert full_text == [ARXIV]  # stored after the first question
    assert [(m["role"], m["content"]) for m in model.calls[1][1:]] == [
        ("user", "First?"), ("assistant", "First answer."), ("user", "Second?")]


def test_cut_off_answer_is_marked(c, app, chat_pp, monkeypatch, full_text):
    monkeypatch.setattr(chat, "_llm_stream", FakeModel(["Half an ans"], finish_reason="length"))
    _ask(c, chat_pp, "Long?")
    _idle(chat_pp["pp"])
    assert "Cut off" in _msgs(app, chat_pp["pp"])[1][2]


def test_model_error_is_shown_and_left_out_of_the_next_context(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["Partial", llm.LLMUnavailableError("all models down")], ["Recovered."])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q1")
    _idle(chat_pp["pp"])
    role, status, content = _msgs(app, chat_pp["pp"])[1]
    assert status == "error" and "Partial" in content and "all models down" in content
    _ask(c, chat_pp, "Q2")
    _idle(chat_pp["pp"])
    assert [m["content"] for m in model.calls[1][1:]] == ["Q2"]  # the failed turn isn't context
    assert _msgs(app, chat_pp["pp"])[-1] == ("assistant", "done", "Recovered.")


def test_two_papers_answer_at_the_same_time(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["", FakeModel.WAIT, "A"], ["B"])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "About the arXiv paper")
    _wait(model.waiting.is_set, "first chat to think")
    _ask(c, chat_pp, "About the blog", pp="wpp")
    _idle(chat_pp["wpp"])  # not held up by the other chat
    assert _msgs(app, chat_pp["wpp"])[-1] == ("assistant", "done", "B")
    model.release()
    _idle(chat_pp["pp"])


# ── Queue ────────────────────────────────────────────────────────────────────

def test_questions_sent_meanwhile_are_queued_and_answered_in_order(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["", FakeModel.WAIT, "A1"], ["A2"], ["A3"])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q1")
    _wait(model.waiting.is_set, "model to think")
    _ask(c, chat_pp, "Q2")
    html = _ask(c, chat_pp, "Q3")
    assert html.count("Queued: sent when the answer before it is done") == 2 and html.count(">Remove<") == 2

    model.release()
    _idle(chat_pp["pp"])
    assert _msgs(app, chat_pp["pp"]) == [
        ("user", "done", "Q1"), ("assistant", "done", "A1"),
        ("user", "done", "Q2"), ("assistant", "done", "A2"),
        ("user", "done", "Q3"), ("assistant", "done", "A3"),
    ]
    assert [m["content"] for m in model.calls[2][1:]] == ["Q1", "A1", "Q2", "A2", "Q3"]
    log = c.get("/projects/chat-proj").data.decode()
    assert re.findall(r"\b([QA][123])\b", log.split('class="chat-log"')[1].split("chat-form")[0]) == ["Q1", "A1", "Q2", "A2", "Q3", "A3"]


def test_finished_answer_poll_shows_the_next_queued_one_already_running(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["", FakeModel.WAIT, "A1"], ["", FakeModel.WAIT, "A2"])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q1")
    first = _latest_answer_id(app, chat_pp["pp"])
    _wait(model.waiting.is_set, "model to think")
    _ask(c, chat_pp, "Q2")
    model.release()
    _wait(lambda: chat.live_state(first) is None, "first answer to end")
    html = c.get(f"/chat/message/{first}").data.decode()
    second = _latest_answer_id(app, chat_pp["pp"])
    assert second != first and "A1" in html and f'hx-get="/chat/message/{second}"' in html  # page polls the next one
    _wait(model.waiting.is_set, "second answer to think")
    model.release()
    _idle(chat_pp["pp"])


def test_a_queued_question_can_be_removed_until_it_is_sent(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["", FakeModel.WAIT, "A1"], ["A2"])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q1")
    _wait(model.waiting.is_set, "model to think")
    _ask(c, chat_pp, "Q2")
    _ask(c, chat_pp, "Q3")
    with app.app_context():
        q2, q3 = [m.id for m in ChatMessage.query.filter_by(project_paper_id=chat_pp["pp"], status="queued").order_by(ChatMessage.id)]
    assert c.post(f"/chat/message/{q3}/remove").data.decode() == ""  # gone from the page
    model.release()
    _idle(chat_pp["pp"])
    assert [m[2] for m in _msgs(app, chat_pp["pp"])] == ["Q1", "A1", "Q2", "A2"]
    resp = c.post(f"/chat/message/{q2}/remove")  # already answered: not removed, the log is redrawn
    assert resp.headers["HX-Retarget"] == f"#chat-log-{chat_pp['pp']}" and "A2" in resp.data.decode()
    assert len(_msgs(app, chat_pp["pp"])) == 4


# ── Stop and clear ───────────────────────────────────────────────────────────

def test_stop_while_thinking_closes_the_stream(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["", FakeModel.WAIT, "", "never shown"])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q")
    answer_id = _latest_answer_id(app, chat_pp["pp"])
    _wait(model.waiting.is_set, "model to think")
    assert c.post(f"/chat/message/{answer_id}/stop").status_code == 204
    assert "Stopping" in c.get(f"/chat/message/{answer_id}").data.decode()
    model.release()
    _idle(chat_pp["pp"])
    assert model.closed == [1]  # the HTTP stream was closed, so the server stops generating
    assert _msgs(app, chat_pp["pp"])[1] == ("assistant", "stopped", "")
    assert "Stopped before it wrote anything" in c.get(f"/chat/message/{answer_id}").data.decode()


def test_stopped_partial_answer_is_kept_and_queue_continues(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["", "Half of it", FakeModel.WAIT, "Half of it and the rest"], ["A2"])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q1")
    answer_id = _latest_answer_id(app, chat_pp["pp"])
    _wait(model.waiting.is_set, "model mid-answer")
    _ask(c, chat_pp, "Q2")
    c.post(f"/chat/message/{answer_id}/stop")
    model.release()
    _idle(chat_pp["pp"])
    assert _msgs(app, chat_pp["pp"]) == [("user", "done", "Q1"), ("assistant", "stopped", "Half of it"),
                                         ("user", "done", "Q2"), ("assistant", "done", "A2")]
    assert model.calls[1][2]["content"] == "Half of it\n\n[The user stopped this answer here.]"
    assert "Stopped." in c.get(f"/chat/message/{answer_id}").data.decode()


def test_stop_during_paper_download_skips_the_model(c, app, chat_pp, monkeypatch):
    downloading, done = threading.Event(), threading.Event()

    def slow_fetch(arxiv_id, max_chars):
        downloading.set()
        done.wait(5)
        return PAPER_TEXT

    monkeypatch.setattr(chat, "fetch_paper_full_text", slow_fetch)
    model = FakeModel(["never"])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q")
    answer_id = _latest_answer_id(app, chat_pp["pp"])
    _wait(downloading.is_set, "download to start")
    assert "Reading the paper" in c.get(f"/chat/message/{answer_id}").data.decode()
    c.post(f"/chat/message/{answer_id}/stop")
    done.set()
    _idle(chat_pp["pp"])
    assert model.calls == [] and _msgs(app, chat_pp["pp"])[1][1] == "stopped"


def test_stop_of_a_finished_answer_is_harmless(c, app, chat_pp, monkeypatch, full_text):
    monkeypatch.setattr(chat, "_llm_stream", FakeModel(["Done."]))
    _ask(c, chat_pp, "Q")
    _idle(chat_pp["pp"])
    assert c.post(f"/chat/message/{_latest_answer_id(app, chat_pp['pp'])}/stop").status_code == 204
    assert _msgs(app, chat_pp["pp"])[1] == ("assistant", "done", "Done.")


def test_clear_mid_answer_stops_it_and_a_new_chat_works(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["", FakeModel.WAIT, "", "old"], ["New answer."])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Old question")
    _wait(model.waiting.is_set, "model to think")
    _ask(c, chat_pp, "Queued old question")
    html = c.post(f"/paper/{ARXIV}/chat/clear", data={"project_id": chat_pp["project"]}).data.decode()
    assert "Ask about this paper" in html and _msgs(app, chat_pp["pp"]) == []
    _ask(c, chat_pp, "Fresh question")  # queued behind the stopping worker, then answered
    model.release()
    _idle(chat_pp["pp"])
    assert _msgs(app, chat_pp["pp"]) == [("user", "done", "Fresh question"), ("assistant", "done", "New answer.")]
    assert [m["content"] for m in model.calls[1][1:]] == ["Fresh question"]


def test_poll_of_a_cleared_answer_stops_polling(c, app, chat_pp):
    assert c.get("/chat/message/987654").status_code == 286


def test_message_ids_are_never_reused_after_a_clear(c, app, chat_pp, monkeypatch, full_text):
    """A page still polling an answer of a cleared chat must not be handed a new message with the same id."""
    monkeypatch.setattr(chat, "_llm_stream", FakeModel(["A"], ["B"]))
    _ask(c, chat_pp, "Q")
    _idle(chat_pp["pp"])
    old = _latest_answer_id(app, chat_pp["pp"])
    c.post(f"/paper/{ARXIV}/chat/clear", data={"project_id": chat_pp["project"]})
    _ask(c, chat_pp, "Q again")
    _idle(chat_pp["pp"])
    assert _latest_answer_id(app, chat_pp["pp"]) > old + 1


# ── Server restart ───────────────────────────────────────────────────────────

def test_restart_marks_the_cut_answer_and_resumes_the_queue(c, app, chat_pp, monkeypatch, full_text):
    """What a restart leaves behind: an answer still "streaming" with no worker, and a queued question."""
    with app.app_context():
        q1 = ChatMessage(project_paper_id=chat_pp["pp"], role="user", content="Q1", status="done")
        db.session.add(q1)
        db.session.flush()
        db.session.add_all([
            ChatMessage(project_paper_id=chat_pp["pp"], role="assistant", content="Half", status="streaming", reply_to_id=q1.id),
            ChatMessage(project_paper_id=chat_pp["pp"], role="user", content="Q2", status="queued"),
        ])
        db.session.commit()
    model = FakeModel(["A2"])
    monkeypatch.setattr(chat, "_llm_stream", model)
    html = c.get("/projects/chat-proj").data.decode()
    assert "Interrupted: the server restarted" in html
    _idle(chat_pp["pp"])
    msgs = _msgs(app, chat_pp["pp"])
    assert msgs[1][:2] == ("assistant", "error") and msgs[2:] == [("user", "done", "Q2"), ("assistant", "done", "A2")]
    assert [m["content"] for m in model.calls[0][1:]] == ["Q2"]  # the cut answer isn't context


# ── Context window ───────────────────────────────────────────────────────────

def test_full_context_blocks_the_question_instead_of_dropping_turns(c, app, chat_pp, monkeypatch):
    with app.app_context():
        q = ChatMessage(project_paper_id=chat_pp["pp"], role="user", content="Q1")
        db.session.add(q)
        db.session.flush()
        db.session.add(ChatMessage(project_paper_id=chat_pp["pp"], role="assistant", content="A1", reply_to_id=q.id,
                                   context_source="html", context_tokens=chat.CONTEXT_LIMIT - 1000))
        db.session.commit()
    monkeypatch.setattr(chat, "_llm_stream", lambda *a, **k: pytest.fail("must not call the model"))
    html = _ask(c, chat_pp, "Q2")
    assert "used up the model" in html and "Clear the chat" in html
    assert re.search(r'class="chat-input"[^>]*\sdisabled', html)
    assert len(_msgs(app, chat_pp["pp"])) == 2

    html = c.post(f"/paper/{ARXIV}/chat/clear", data={"project_id": chat_pp["project"]}).data.decode()
    assert "Ask about this paper" in html and _msgs(app, chat_pp["pp"]) == []


def test_queued_question_that_no_longer_fits_gets_an_error_not_a_dropped_turn(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["", FakeModel.WAIT, "A1"], prompt_tokens=chat.CONTEXT_LIMIT - chat.MAX_ANSWER_TOKENS - 10)
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q1")
    _wait(model.waiting.is_set, "model to think")
    _ask(c, chat_pp, "Q2 " * 50)
    model.release()
    _idle(chat_pp["pp"])
    msgs = _msgs(app, chat_pp["pp"])
    assert msgs[3][:2] == ("assistant", "error") and "used up the model" in msgs[3][2]
    assert len(model.calls) == 1


def test_context_meter_shows_usage(c, app, chat_pp, monkeypatch, full_text):
    monkeypatch.setattr(chat, "_llm_stream", FakeModel(["x" * 400], prompt_tokens=150_000))
    _ask(c, chat_pp, "Q")
    _idle(chat_pp["pp"])
    html = c.get("/projects/chat-proj").data.decode()
    assert "Context 150.1k / 196k" in html and "width: 77%; background-color: var(--warning)" in html


# ── Where the paper text comes from ──────────────────────────────────────────

def test_pdf_when_there_is_no_html(c, app, chat_pp, monkeypatch):
    monkeypatch.setattr(chat, "fetch_paper_full_text", lambda *a, **k: None)
    monkeypatch.setattr(chat, "_pdf_text", lambda url, arxiv: PAPER_TEXT if url == f"https://arxiv.org/pdf/{ARXIV}" and arxiv else None)
    model = FakeModel(["From the PDF."])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q")
    _idle(chat_pp["pp"])
    assert "extracted from the PDF" in model.calls[0][0]["content"]
    assert "Full text (PDF)" in c.get("/projects/chat-proj").data.decode()


def test_abstract_only_is_marked_and_the_download_retried_next_time(c, app, chat_pp, monkeypatch):
    monkeypatch.setattr(chat, "fetch_paper_full_text", lambda *a, **k: None)
    monkeypatch.setattr(chat, "_pdf_text", lambda url, arxiv: None)
    model = FakeModel(["From the abstract only."])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Results?")
    _idle(chat_pp["pp"])
    system = model.calls[0][0]["content"]
    assert "Only the abstract is available" in system and "We sparsify things." in system
    html = c.get("/projects/chat-proj").data.decode()
    assert "Abstract only" in html and "chat-badge-warn" in html and "only sees the title and abstract" not in html.split("chat-log")[0][:0]
    with app.app_context():
        assert PaperText.query.filter_by(paper_id=chat_pp["paper"]).count() == 0
    monkeypatch.setattr(chat, "_pdf_text", lambda url, arxiv: PAPER_TEXT)
    with app.app_context():
        assert chat.paper_text(Paper.query.get(chat_pp["paper"])) == (PAPER_TEXT, "pdf")


def test_failed_pdf_download_falls_back_to_the_abstract(monkeypatch):
    def boom(**kw):
        raise httpx.ConnectError("arxiv down")
    monkeypatch.setattr(chat, "_http_client", boom)
    monkeypatch.setattr(chat, "_ArxivLockCtx", contextlib.nullcontext)
    assert chat._pdf_text(f"https://arxiv.org/pdf/{ARXIV}", arxiv=True) is None


def test_html_error_page_is_not_taken_for_the_paper(app, chat_pp, monkeypatch):
    monkeypatch.setattr(chat, "fetch_paper_full_text", lambda *a, **k: "Conversion failed.")  # < MIN_TEXT_CHARS
    monkeypatch.setattr(chat, "_pdf_text", lambda url, arxiv: PAPER_TEXT)
    with app.app_context():
        assert chat.paper_text(Paper.query.get(chat_pp["paper"]))[1] == "pdf"


def test_web_paper_chat_never_downloads(c, app, chat_pp, monkeypatch):
    monkeypatch.setattr(chat, "_fetch_text", lambda paper: pytest.fail("nothing to download for this web paper"))
    model = FakeModel(["Blog answer."])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q", pp="wpp")
    _idle(chat_pp["wpp"])
    assert "Blog abstract." in model.calls[0][0]["content"]


# ── Browser (slow): the real page with a scripted model ──────────────────────

@pytest.mark.slow
def test_chat_in_the_browser_queue_stop_and_drafts(live_server, browser_session, chat_pp, monkeypatch, full_text):
    browser, state = browser_session
    model = FakeModel(["", FakeModel.WAIT, "**First** answer."],
                      ["", FakeModel.WAIT, "Second, partial", FakeModel.WAIT, "Second, partial and the rest"])
    monkeypatch.setattr(chat, "_llm_stream", model)
    context = browser.new_context(storage_state=state)
    pg = context.new_page()
    pg.goto(f"{live_server}/projects/chat-proj")
    card_sel = f'[id^="paper-card-{ARXIV.replace(".", "-")}-"]'
    card = pg.locator(card_sel)
    card.get_by_role("button", name="Ask about this paper").click()
    bound = "() => !!(document.querySelector('.chat-form') || {})['htmx-internal-data']"
    pg.wait_for_function(bound)
    card.locator(".chat-input").fill("First question")
    card.locator(".chat-input").press("Enter")
    pg.wait_for_selector("text=Thinking")
    assert card.locator(".chat-input").input_value() == ""

    # queue a second question while the first is being answered
    pg.wait_for_function(bound)
    card.locator(".chat-input").fill("Second question")
    card.locator(".chat-input").press("Enter")
    pg.wait_for_selector("text=Queued: sent when the answer before it is done")

    # a draft typed now survives the first answer finishing and the second starting
    pg.wait_for_function(bound)
    card.locator(".chat-input").fill("A draft I am typing")
    model.release()
    pg.wait_for_selector(".chat-md strong:text('First')")
    _wait(model.waiting.is_set, "second answer to think")
    model.release()
    pg.wait_for_selector("text=Second, partial")
    assert card.locator(".chat-input").input_value() == "A draft I am typing"

    # stop the second answer: the partial text stays, marked as stopped
    card.get_by_role("button", name="Stop").click()
    model.release()
    pg.wait_for_selector("text=Stopped.")
    log = card.locator(".chat-log").inner_text()
    assert "Second, partial" in log and "and the rest" not in log
    assert card.locator(".chat-poll").count() == 0

    # the draft also survives a reload
    pg.reload()
    assert pg.locator(f"{card_sel} .chat-input").input_value() == "A draft I am typing"
    context.close()


def test_math_is_marked_up_for_katex_and_money_is_left_alone():
    out = str(chat.render_markdown(r"the \(x^2\) and $N \times N$, $$\sum_i a_i$$, \[ O(N^2) \] and $5 vs $6"))
    assert out.count('class="math inline"') == 4 and r"N \times N" in out and "a_i" in out
    assert "$5 vs $6" in out and "<em>" not in out  # underscores in math are not emphasis


def test_answer_from_a_fallback_model_says_so(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["Small answer."])
    real_call = model.__call__

    def fallback(*a, **k):
        for event in real_call(*a, **k):
            yield {**event, "model": llm.MODELS[1]} if isinstance(event, dict) else event

    monkeypatch.setattr(chat, "_llm_stream", fallback)
    _ask(c, chat_pp, "Q")
    _idle(chat_pp["pp"])
    html = c.get("/projects/chat-proj").data.decode()
    assert f"Answered by the fallback model {llm.MODELS[1]}" in html


def test_context_meter_counts_up_while_the_answer_is_written(c, app, chat_pp, monkeypatch, full_text):
    """First answer: estimated from the prompt's length, growing with the answer, then the model's count."""
    model = FakeModel(["", "x" * 3200, FakeModel.WAIT, "x" * 32000, FakeModel.WAIT])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q")
    answer_id = _latest_answer_id(app, chat_pp["pp"])
    _wait(model.waiting.is_set, "first part of the answer")

    def meter():
        html = c.get(f"/chat/message/{answer_id}").data.decode()
        m = re.search(r'id="chat-ctx-\d+" hx-swap-oob="true"[^>]*>\s*Context ~([\d.]+)k', html)
        return float(m.group(1)) if m else None

    first = meter()
    assert first is not None and first > 1  # the paper text alone is a few k tokens
    model.release()
    _wait(lambda: len(chat.live_state(answer_id)["text"]) == 32000, "more text")
    assert meter() == pytest.approx(first + (32000 - 3200) / chat.CHARS_PER_TOKEN / 1000, abs=0.15)
    model.release()
    _idle(chat_pp["pp"])
    html = c.get(f"/chat/message/{answer_id}").data.decode()
    assert re.search(r"Context 52\.0k / 196k", html) and "Context ~" not in html  # 42000 + 32000/3.2, exact


def test_llm_stream_announces_retries(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="down") if len(calls) == 1 else httpx.Response(200, text=_sse(["Ok."]))

    _mock_http(monkeypatch, handler)
    out = list(llm._llm_stream([{"role": "user", "content": "hi"}]))
    assert out[0] == ("retry", llm.MODELS[0], 2) and out[-2] == "Ok."


def test_unavailable_model_says_so_instead_of_thinking_forever(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel([("retry", llm.MODELS[1], 1), FakeModel.WAIT, llm.LLMUnavailableError("503: down")])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Anyone there?")
    answer_id = _latest_answer_id(app, chat_pp["pp"])
    _wait(model.waiting.is_set, "the retry")
    html = c.get(f"/chat/message/{answer_id}").data.decode()
    assert f"trying again ({llm.MODELS[1]}, attempt 1)" in html and "answering" in html
    model.release()
    _idle(chat_pp["pp"])
    role, status, content = _msgs(app, chat_pp["pp"])[1]
    assert status == "error" and "The model is not available right now" in content and "Try again later" in content


def test_prompt_carries_the_project_keywords_and_the_card_notes(c, app, chat_pp, monkeypatch, full_text):
    """The model reads the researcher's own terms in the project's sense (2026-10-15: "programs" was taken
    for changed weights when only the research interest was there)."""
    with app.app_context():
        db.session.get(Project, chat_pp["project"]).saved_keywords = json.dumps(["layer looping", "PoLar"])
        db.session.get(ProjectPaper, chat_pp["pp"]).notes = "maybe not for polars"
        db.session.commit()
    model = FakeModel(["Answer."])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q?")
    _idle(chat_pp["pp"])
    system = model.calls[0][0]["content"]
    assert "in the sense of that project" in system
    assert "Search keywords of the project: layer looping, PoLar" in system
    assert "The researcher's notes on this paper: maybe not for polars" in system
    assert "sparse attention" in system and "It sparsifies attention." in system


def test_prompt_leaves_out_keywords_and_notes_when_there_are_none(c, app, chat_pp, monkeypatch, full_text):
    model = FakeModel(["Answer."])
    monkeypatch.setattr(chat, "_llm_stream", model)
    _ask(c, chat_pp, "Q?")
    _idle(chat_pp["pp"])
    system = model.calls[0][0]["content"]
    assert "Search keywords" not in system and "notes on this paper" not in system
