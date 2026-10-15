"""
Per-paper actions across projects and the New Papers re-screen:

- "Why it's relevant" (ProjectPaper.relevance_reason) from the deep relevance check
- Copy / Move a paper into another project's My Collection (/paper/<id>/transfer)
- Re-screen New Papers against the current research interest (crawl.run_rescreen)
- Card layout: New Papers actions sit together in the bottom row

All LLM / network calls are mocked.
"""

import json
from unittest.mock import patch

import pytest

from app import db
from app.models import ChatMessage, Paper, PaperSummary, Project, ProjectPaper, ScreenedPaper, compute_screening_hash
from tests.conftest import make_llm_json_response, FAKE_RELEVANCE_YES, FAKE_RELEVANCE_NO


@pytest.fixture
def two_projects(app):
    """Source project with one New Papers and one My Collection paper, plus a target project."""
    with app.app_context():
        src = Project(name="Source Proj", slug="pa-source", research_interest="sparse attention")
        dst = Project(name="Target Proj", slug="pa-target", research_interest="robot manipulation")
        db.session.add_all([src, dst])
        db.session.flush()
        new = Paper(arxiv_id="2601.00001", title="New One", abstract="About attention.", source="arxiv")
        coll = Paper(arxiv_id="2601.00002", title="Collected One", abstract="About robots.", source="arxiv")
        db.session.add_all([new, coll])
        db.session.flush()
        db.session.add(ProjectPaper(project_id=src.id, paper_id=new.id))
        db.session.add(ProjectPaper(project_id=src.id, paper_id=coll.id, manual_tag="related",
                                    paper_tags='["important"]', notes="my notes"))
        db.session.commit()
        ids = {"src": src.id, "dst": dst.id, "new": new.id, "coll": coll.id}
    yield ids
    with app.app_context():
        pp_ids = [pp.id for pp in ProjectPaper.query.filter(ProjectPaper.project_id.in_([ids["src"], ids["dst"]]))]
        ChatMessage.query.filter(ChatMessage.project_paper_id.in_(pp_ids)).delete()
        ProjectPaper.query.filter(ProjectPaper.project_id.in_([ids["src"], ids["dst"]])).delete()
        ScreenedPaper.query.filter(ScreenedPaper.project_id.in_([ids["src"], ids["dst"]])).delete()
        PaperSummary.query.filter(PaperSummary.paper_id.in_([ids["new"], ids["coll"]])).delete()
        Paper.query.filter(Paper.id.in_([ids["new"], ids["coll"]])).delete()
        Project.query.filter(Project.id.in_([ids["src"], ids["dst"]])).delete()
        db.session.commit()


@pytest.fixture
def logged_in(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    return client


def _link(project_id, paper_id):
    db.session.expire_all()
    return ProjectPaper.query.filter_by(project_id=project_id, paper_id=paper_id).first()


# ── Why it's relevant ─────────────────────────────────────────────────────────

def test_deep_check_returns_why_only_when_relevant(app):
    from app.crawl import _screen_paper_deep
    with app.app_context():
        with patch("app.crawl._llm_json", return_value=make_llm_json_response(FAKE_RELEVANCE_YES)):
            relevant, reason, _, _, _, why = _screen_paper_deep("T", "A", "full", "interest", [])
        assert (relevant, reason, why) == (True, "relevant", FAKE_RELEVANCE_YES["reason"])
        with patch("app.crawl._llm_json", return_value=make_llm_json_response(FAKE_RELEVANCE_NO)):
            relevant, _, _, _, _, why = _screen_paper_deep("T", "A", "full", "interest", [])
        assert (relevant, why) == (False, "")


def test_summary_prompt_does_not_get_the_research_interest(app):
    """The summary is shared across projects, so it must not be written for one project's interest."""
    from app.crawl import generate_summary_with_full_content
    with app.app_context():
        with patch("app.crawl._llm_json", return_value=make_llm_json_response({"summary": "S"})) as llm:
            generate_summary_with_full_content("Title", "Abstract", "Content")
    prompt = llm.call_args[0][0][0]["content"]
    assert "research project on" not in prompt
    assert "Title" in prompt


def test_relevance_reason_is_filled_in_background(app, two_projects):
    from app.crawl import start_relevance_reason_async
    with app.app_context():
        with patch("app.crawl._llm_json", return_value=make_llm_json_response({"reason": "Robots, like yours."})):
            start_relevance_reason_async(two_projects["src"], two_projects["coll"], app=app).join(5)
        assert _link(two_projects["src"], two_projects["coll"]).relevance_reason == "Robots, like yours."


# ── Copy / Move ───────────────────────────────────────────────────────────────

def test_copy_puts_paper_in_target_collection_and_keeps_source(app, logged_in, two_projects):
    ids = two_projects
    with patch("app.routes.papers.start_relevance_reason_async") as why:
        resp = logged_in.post("/paper/2601.00002/transfer", data={
            "project_id": ids["src"], "target_project_id": ids["dst"], "mode": "copy"})
    assert resp.status_code == 200
    assert "Copied to My Collection of" in resp.get_data(as_text=True)
    target = _link(ids["dst"], ids["coll"])
    assert target.manual_tag == "related" and target.collected_at is not None
    assert json.loads(target.paper_tags) == ["important"]
    assert target.notes == "my notes"
    source = _link(ids["src"], ids["coll"])
    assert source.trashed_at is None and source.manual_tag == "related"
    why.assert_called_once_with(ids["dst"], ids["coll"])


def test_move_trashes_paper_in_source(app, logged_in, two_projects):
    ids = two_projects
    with patch("app.routes.papers.start_relevance_reason_async"):
        resp = logged_in.post("/paper/2601.00001/transfer", data={
            "project_id": ids["src"], "target_project_id": ids["dst"], "mode": "move"},
            headers={"HX-Request": "true"})
    assert resp.status_code == 200
    assert "New One" in resp.get_data(as_text=True)  # paper list re-rendered (now in Trash)
    assert _link(ids["src"], ids["new"]).trashed_at is not None
    target = _link(ids["dst"], ids["new"])
    assert target.manual_tag == "related" and target.trashed_at is None
    assert json.loads(target.paper_tags) == []  # an untagged paper stays untagged, no "to read" default


def _chat(project_id, paper_id, turns):
    """turns: (question, answer, answer status) tuples; returns the ProjectPaper id."""
    pp = _link(project_id, paper_id)
    if pp is None:
        pp = ProjectPaper(project_id=project_id, paper_id=paper_id, manual_tag="related")
        db.session.add(pp)
        db.session.flush()
    for q, a, status in turns:
        qm = ChatMessage(project_paper_id=pp.id, role="user", content=q, status="done")
        db.session.add(qm)
        db.session.flush()
        db.session.add(ChatMessage(project_paper_id=pp.id, role="assistant", content=a, status=status,
                                   reply_to_id=qm.id, context_source="html", context_tokens=42000))
    db.session.commit()
    return pp.id


def _chat_contents(project_id, paper_id):
    from app import chat
    pp = _link(project_id, paper_id)
    return [(m.role, m.content, m.status) for m in chat.ordered(ChatMessage.query.filter_by(project_paper_id=pp.id).all())]


def test_copy_and_move_take_the_finished_chat_along(app, logged_in, two_projects):
    ids = two_projects
    with app.app_context():
        _chat(ids["src"], ids["coll"], [("Q1", "A1", "done"), ("Q2", "partial", "stopped"), ("Q3", "", "streaming")])
        src_pp = _link(ids["src"], ids["coll"]).id
        db.session.add(ChatMessage(project_paper_id=src_pp, role="user", content="Q4", status="queued"))
        db.session.commit()
    with patch("app.routes.papers.start_relevance_reason_async"):
        logged_in.post("/paper/2601.00002/transfer", data={
            "project_id": ids["src"], "target_project_id": ids["dst"], "mode": "copy"})
    with app.app_context():
        assert _chat_contents(ids["dst"], ids["coll"]) == [
            ("user", "Q1", "done"), ("assistant", "A1", "done"), ("user", "Q2", "done"), ("assistant", "partial", "stopped")]
        assert len(_chat_contents(ids["src"], ids["coll"])) == 7  # the source keeps its chat
        target = _link(ids["dst"], ids["coll"])
        assert ChatMessage.query.filter_by(project_paper_id=target.id, role="assistant").first().context_tokens == 42000


def test_transfer_keeps_the_targets_own_chat(app, logged_in, two_projects):
    ids = two_projects
    with app.app_context():
        _chat(ids["src"], ids["new"], [("Source Q", "Source A", "done")])
        _chat(ids["dst"], ids["new"], [("Target Q", "Target A", "done")])
    with patch("app.routes.papers.start_relevance_reason_async"):
        logged_in.post("/paper/2601.00001/transfer", data={
            "project_id": ids["src"], "target_project_id": ids["dst"], "mode": "move"}, headers={"HX-Request": "true"})
    with app.app_context():
        assert _chat_contents(ids["dst"], ids["new"]) == [("user", "Target Q", "done"), ("assistant", "Target A", "done")]


def test_transfer_into_project_where_paper_was_trashed_restores_it(app, logged_in, two_projects):
    ids = two_projects
    with app.app_context():
        db.session.add(ProjectPaper(project_id=ids["dst"], paper_id=ids["coll"], notes="kept there",
                                    trashed_at=db.func.now(), relevance_reason="already known"))
        db.session.commit()
    with patch("app.routes.papers.start_relevance_reason_async") as why:
        logged_in.post("/paper/2601.00002/transfer", data={
            "project_id": ids["src"], "target_project_id": ids["dst"], "mode": "copy"})
    target = _link(ids["dst"], ids["coll"])
    assert target.trashed_at is None and target.manual_tag == "related"
    assert target.notes == "kept there"  # target notes win
    why.assert_not_called()  # it already has a reason for that project


@pytest.mark.parametrize("mode,same", [("delete", False), ("copy", True)])
def test_transfer_rejects_bad_requests(app, logged_in, two_projects, mode, same):
    ids = two_projects
    resp = logged_in.post("/paper/2601.00002/transfer", data={
        "project_id": ids["src"], "target_project_id": ids["src"] if same else ids["dst"], "mode": mode})
    assert resp.status_code == 400
    assert _link(ids["dst"], ids["coll"]) is None


def test_transfer_menu_lists_other_projects(app, logged_in, two_projects):
    ids = two_projects
    with app.app_context():
        db.session.add(Project(name="Trashed Proj", slug="pa-trashed", trashed_at=db.func.now()))
        db.session.add(ProjectPaper(project_id=ids["dst"], paper_id=ids["coll"], manual_tag="related"))
        db.session.commit()
    try:
        html = logged_in.get(f"/paper/2601.00002/transfer-menu?project_id={ids['src']}").get_data(as_text=True)
        assert "Target Proj" in html
        assert "already in its My Collection" in html
        assert "Source Proj" not in html and "Trashed Proj" not in html
        html = logged_in.get(f"/paper/2601.00001/transfer-menu?project_id={ids['src']}").get_data(as_text=True)
        assert '"mode": "copy"' in html and '"mode": "move"' in html
    finally:
        with app.app_context():
            Project.query.filter_by(slug="pa-trashed").delete()
            db.session.commit()


# ── Re-screen ─────────────────────────────────────────────────────────────────

def _quick(title, abstract, research_interest, refs):
    verdict = {"Off topic": (False, "not-relevant"), "Garbled": (False, "parse-failed")}
    relevant, reason = verdict.get(title, (True, "relevant"))
    return relevant, reason, "m", 1, 1


def _deep(title, abstract, content, research_interest, refs):
    return True, "relevant", "m", 1, 1, f"Fresh reason for {title}"


def test_rescreen_trashes_clear_rejects_and_keeps_the_rest(app, two_projects):
    from app import crawl
    ids = two_projects
    with app.app_context():
        src = db.session.get(Project, ids["src"])
        papers = {}
        for i, title in enumerate(["Off topic", "Still fits", "Garbled", "Already screened"]):
            p = Paper(arxiv_id=f"2601.1000{i}", title=title, abstract="x", source="arxiv")
            db.session.add(p)
            db.session.flush()
            db.session.add(ProjectPaper(project_id=src.id, paper_id=p.id, relevance_reason="old"))
            papers[title] = p.id
        collection = [pp for pp in ProjectPaper.query.filter_by(project_id=src.id).all() if pp.manual_tag]
        db.session.add(ScreenedPaper(project_id=src.id, arxiv_id="2601.10003", is_relevant=True,
                                     screening_hash=compute_screening_hash(src.research_interest, collection)))
        db.session.commit()
        try:
            crawl._rescreen_tasks.pop(src.id, None)
            with patch("app.crawl._screen_paper_quick", side_effect=_quick) as quick, \
                 patch("app.crawl._screen_paper_deep", side_effect=_deep), \
                 patch("app.crawl.get_paper_content", return_value="full text"):
                state = crawl.run_rescreen(src.id)

            # "New One" from the fixture + 3 of the 4 above; "Already screened" skipped without an LLM call
            assert quick.call_count == 4
            assert (state["total"], state["checked"], state["skipped"]) == (4, 4, 1)
            assert (state["trashed"], state["kept"], state["status"]) == (1, 3, "done")
            assert _link(src.id, papers["Off topic"]).trashed_at is not None
            assert _link(src.id, papers["Garbled"]).trashed_at is None  # unsure → kept
            assert _link(src.id, papers["Garbled"]).relevance_reason == "old"
            assert _link(src.id, papers["Still fits"]).relevance_reason == "Fresh reason for Still fits"
            assert _link(src.id, papers["Already screened"]).relevance_reason == "old"
            # My Collection is never touched
            assert _link(src.id, ids["coll"]).trashed_at is None
            # Clear verdicts are cached for the next crawl/re-screen, parse failures aren't
            cached = {sp.arxiv_id: sp.is_relevant for sp in ScreenedPaper.query.filter_by(project_id=src.id)}
            assert cached["2601.10000"] is False and cached["2601.10001"] is True
            assert "2601.10002" not in cached
        finally:
            ProjectPaper.query.filter(ProjectPaper.paper_id.in_(papers.values())).delete()
            Paper.query.filter(Paper.id.in_(papers.values())).delete()
            db.session.commit()
            crawl._rescreen_tasks.pop(src.id, None)


def test_rescreen_leaves_papers_collected_meanwhile_alone(app, two_projects):
    from app import crawl
    ids = two_projects

    def quick_and_collect(title, *args):
        # The user moves the paper to My Collection (another request) while its screen is running
        with app.app_context():
            pp = ProjectPaper.query.filter_by(project_id=ids["src"], paper_id=ids["new"]).first()
            pp.manual_tag = "related"
            db.session.commit()
        return False, "not-relevant", "m", 1, 1

    with app.app_context():
        crawl._rescreen_tasks.pop(ids["src"], None)
        with patch("app.crawl._screen_paper_quick", side_effect=quick_and_collect):
            state = crawl.run_rescreen(ids["src"])
        assert state["trashed"] == 0
        assert _link(ids["src"], ids["new"]).trashed_at is None
        crawl._rescreen_tasks.pop(ids["src"], None)


def test_rescreen_route_needs_a_research_interest(app, logged_in, two_projects):
    with app.app_context():
        db.session.get(Project, two_projects["src"]).research_interest = ""
        db.session.commit()
    with patch("app.routes.projects.start_rescreen_async") as start:
        assert logged_in.post("/projects/pa-source/rescreen").status_code == 400
    start.assert_not_called()


def test_rescreen_status_refreshes_paper_list_once_when_done(app, logged_in, two_projects):
    from app import crawl
    crawl._rescreen_tasks[two_projects["src"]] = {
        "status": "running", "total": 3, "checked": 1, "kept": 1, "trashed": 0, "skipped": 0, "error": None}
    try:
        html = logged_in.get("/projects/pa-source/rescreen-status").get_data(as_text=True)
        assert "1 / 3 done" in html and 'hx-trigger="every 2s"' in html and "htmx.ajax" not in html
        crawl._rescreen_tasks[two_projects["src"]].update(status="done", checked=3, trashed=2)
        html = logged_in.get("/projects/pa-source/rescreen-status").get_data(as_text=True)
        assert "2 moved to Trash" in html and "htmx.ajax" in html and "every 2s" not in html
        html = logged_in.get("/projects/pa-source/rescreen-status").get_data(as_text=True)
        assert "htmx.ajax" not in html
    finally:
        crawl._rescreen_tasks.pop(two_projects["src"], None)


# ── Card ──────────────────────────────────────────────────────────────────────

def test_new_papers_card_has_its_actions_together_in_the_bottom_row(app, logged_in, two_projects):
    with app.app_context():
        _link(two_projects["src"], two_projects["new"]).relevance_reason = "It uses sparse attention."
        db.session.commit()
    html = logged_in.get("/projects/pa-source/papers").get_data(as_text=True)
    new_card = html[html.index('id="paper-card-2601-00001'):html.index('id="paper-card-2601-00002')]
    order = [new_card.index(s) for s in ("Move to My Collection", "Copy / Move", "Notify", '/trash"')]
    assert order == sorted(order)
    assert 'style="color: var(--warning);"' in new_card  # yellow star
    assert "Why it's relevant for this project" in new_card and "It uses sparse attention." in new_card
    # Not in the title row any more: it comes after the summary and the arXiv/PDF links
    assert new_card.index(">PDF</a>") < new_card.index("Move to My Collection")
    assert "stops the same papers from coming back" in html  # Trash explains why it is never emptied
    coll_card = html[html.index('id="paper-card-2601-00002'):]
    assert "Move to My Collection" not in coll_card[:coll_card.index("Notes...")]
    assert "Copy / Move" in coll_card


# ── Browser (slow): the htmx wiring of Copy / Move and Re-screen ──────────────

@pytest.fixture
def page(live_server, browser_session, two_projects):
    browser, state = browser_session
    context = browser.new_context(storage_state=state)
    pg = context.new_page()
    pg.goto(f"{live_server}/projects/pa-source")
    pg.wait_for_function("window.htmx !== undefined")
    yield pg
    context.close()


@pytest.mark.slow
def test_move_from_card_menu_in_browser(app, page, two_projects):
    ids = two_projects
    card = page.locator('[id="paper-card-2601-00001-%d"]' % ids["src"])
    with patch("app.routes.papers.start_relevance_reason_async"):
        card.get_by_role("button", name="Copy / Move").click()
        card.locator(".transfer-target", has_text="Target Proj").get_by_role("button", name="Move").click()
        page.wait_for_function("() => !document.querySelector('#new-papers-list [id^=\"paper-card-2601-00001\"]')")
    with app.app_context():
        assert _link(ids["dst"], ids["new"]).manual_tag == "related"
        assert _link(ids["src"], ids["new"]).trashed_at is not None


@pytest.mark.slow
def test_rescreen_button_in_browser(app, page, two_projects):
    ids = two_projects
    page.on("dialog", lambda d: d.accept())  # hx-confirm
    with patch("app.crawl._screen_paper_quick", return_value=(False, "not-relevant", "m", 1, 1)):
        page.get_by_role("button", name="Re-screen").click()
        page.locator("#rescreen-status", has_text="Re-screen done").wait_for(timeout=10_000)
    # The list refreshed by itself: New Papers is empty, the paper sits in Trash
    page.wait_for_function("() => !document.getElementById('new-papers-list')")
    with app.app_context():
        assert _link(ids["src"], ids["new"]).trashed_at is not None
    from app import crawl
    crawl._rescreen_tasks.pop(ids["src"], None)


# ── Refill script ─────────────────────────────────────────────────────────────

def test_backfill_only_nightly_projects_and_untrashed_papers(app, two_projects):
    from scripts.backfill_relevance_reasons import backfill
    ids = two_projects
    with app.app_context():
        src, dst = db.session.get(Project, ids["src"]), db.session.get(Project, ids["dst"])
        src.crawl_hour = 2          # nightly on
        dst.crawl_hour = None       # nightly off → skipped even though it has a paper
        db.session.add(ProjectPaper(project_id=dst.id, paper_id=ids["new"]))
        db.session.commit()
        _link(src.id, ids["coll"]).relevance_reason = "already there"
        db.session.commit()
        with patch("app.crawl._llm_json", return_value=make_llm_json_response({"reason": "Filled."})) as llm:
            result = backfill()
        assert (result["total"], result["filled"]) == (1, 1)
        assert llm.call_count == 1
        assert _link(src.id, ids["new"]).relevance_reason == "Filled."
        assert _link(src.id, ids["coll"]).relevance_reason == "already there"
        assert _link(dst.id, ids["new"]).relevance_reason is None
        # Trash is skipped, and a second run has nothing left to do
        link = _link(src.id, ids["new"])
        link.relevance_reason, link.trashed_at = None, db.func.now()
        db.session.commit()
        assert backfill(dry_run=False)["total"] == 0


# ── Regenerate ────────────────────────────────────────────────────────────────

def test_regenerate_refreshes_summary_and_this_projects_why_line(app, logged_in, two_projects):
    ids = two_projects
    with app.app_context():
        _link(ids["src"], ids["new"]).relevance_reason = "stale"
        db.session.add(ProjectPaper(project_id=ids["dst"], paper_id=ids["new"], relevance_reason="other project's"))
        db.session.commit()

    def llm(messages, **kw):
        prompt = messages[0]["content"]
        payload = {"reason": "Fresh why."} if '{"reason"' in prompt else {"summary": "Fresh summary."}
        return make_llm_json_response(payload)

    with patch("app.routes.papers.get_paper_figure", return_value=(None, None)), \
         patch("app.routes.papers.get_citation_count", return_value=None), \
         patch("app.routes.papers.get_paper_content", return_value="FULL TEXT OF THE PAPER"), \
         patch("app.crawl._llm_json", side_effect=llm) as llm_mock:
        resp = logged_in.post("/paper/2601.00001/regenerate", data={"project_id": ids["src"]},
                              headers={"HX-Request": "true"})
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Fresh summary." in html and "Fresh why." in html
    prompts_sent = [c[0][0][0]["content"] for c in llm_mock.call_args_list]
    summary_prompt = [p for p in prompts_sent if "introduces_dataset" in p]
    assert len(summary_prompt) == 1 and "FULL TEXT OF THE PAPER" in summary_prompt[0]  # not just the abstract
    assert any("main_contributions" in p and "FULL TEXT OF THE PAPER" in p for p in prompts_sent)
    why_prompt = [p for p in prompts_sent if '{"reason"' in p]
    assert len(why_prompt) == 1 and "sparse attention" in why_prompt[0]  # this project's interest
    assert _link(ids["src"], ids["new"]).relevance_reason == "Fresh why."
    assert _link(ids["dst"], ids["new"]).relevance_reason == "other project's"  # other projects untouched


@pytest.mark.slow
def test_tooltips_show_fast_on_hover(app, page, two_projects):
    tip = page.locator("#rw-tip")
    for button, text in ((page.locator("#rescreen-btn"), "Re-check every paper"),
                         (page.get_by_role("button", name="Regenerate").first, "Re-fetches the figure")):
        button.hover()
        page.wait_for_function("() => document.getElementById('rw-tip').classList.contains('show')", timeout=600)
        assert text in tip.inner_text()
        page.mouse.move(0, 0)
        page.wait_for_function("() => !document.getElementById('rw-tip').classList.contains('show')", timeout=600)


# ── Starred projects (projects page, per browser) ─────────────────────────────

@pytest.mark.slow
def test_starred_project_goes_on_top_and_stays_after_reload(live_server, browser_session, two_projects):
    browser, state = browser_session
    context = browser.new_context(storage_state=state)
    pg = context.new_page()
    pg.goto(f"{live_server}/projects")
    order = lambda: pg.eval_on_selector_all("#project-grid .project-card", "cs => cs.map(c => c.dataset.slug)")
    server_order = order()
    # Newest first from the server, so the source project sits below the target
    assert server_order.index("pa-source") > server_order.index("pa-target")

    pg.locator('.project-card[data-slug="pa-source"] .project-star-btn').click()
    assert order()[0] == "pa-source"
    pg.reload()
    assert order()[0] == "pa-source"
    star = pg.locator('.project-card[data-slug="pa-source"] .project-star-btn')
    assert star.get_attribute("aria-pressed") == "true"

    star.click()
    assert order() == server_order
    context.close()


def test_collection_sorts_by_when_a_paper_was_collected_not_when_it_was_crawled(app, logged_in, two_projects):
    """'Date added' in My Collection = date added to My Collection: a paper crawled long ago but
    collected just now goes to the top."""
    import re
    from datetime import datetime, timedelta
    ids = two_projects
    with app.app_context():
        old = Paper(arxiv_id="2601.00003", title="Crawled Long Ago", source="arxiv")
        db.session.add(old)
        db.session.flush()
        ids["old"] = old.id
        db.session.add(ProjectPaper(project_id=ids["src"], paper_id=old.id, added_at=datetime.utcnow() - timedelta(days=30)))
        _link(ids["src"], ids["coll"]).collected_at = datetime.utcnow() - timedelta(days=1)
        db.session.commit()
    try:
        logged_in.post("/paper/2601.00003/tag", data={"project_id": ids["src"], "tag": "related"})
        html = logged_in.get("/projects/pa-source").data.decode()
        keys = dict(re.findall(r'id="paper-card-(2601-0000[23])-\d+"\s+(?:\{#.*?#\}\s+)?data-added-at="([^"]+)"', html))
        assert keys["2601-00003"] > keys["2601-00002"]
    finally:
        with app.app_context():
            ProjectPaper.query.filter_by(paper_id=ids["old"]).delete()
            Paper.query.filter_by(id=ids["old"]).delete()
            db.session.commit()


def test_summary_source_follows_what_the_generators_read():
    from app.crawl import summary_source
    assert summary_source("A long full text " * 10, "Short abstract") == "full_text"
    assert summary_source(None, "Short abstract") == "abstract"
    assert summary_source("Short abstract", "Short abstract") == "abstract"  # content fell back to the abstract


@pytest.mark.parametrize("content, source, label", [
    ("FULL TEXT OF THE PAPER, much longer than the abstract", "full_text", "· from full text"),
    (None, "abstract", "⚠ from abstract only"),
])
def test_regenerate_records_and_shows_what_the_summary_was_made_from(app, logged_in, two_projects, content, source, label):
    ids = two_projects
    with patch("app.routes.papers.get_paper_figure", return_value=(None, None)), \
         patch("app.routes.papers.get_citation_count", return_value=None), \
         patch("app.routes.papers.get_paper_content", return_value=content), \
         patch("app.crawl._llm_json", return_value=make_llm_json_response({"summary": "S.", "reason": "R."})):
        html = logged_in.post("/paper/2601.00001/regenerate", data={"project_id": ids["src"]},
                              headers={"HX-Request": "true"}).get_data(as_text=True)
    with app.app_context():
        assert PaperSummary.query.filter_by(paper_id=ids["new"]).one().content_source == source
    assert label in html and "summary-src" in html


def test_summary_made_before_sources_were_recorded_has_no_label(app, logged_in, two_projects):
    ids = two_projects
    with app.app_context():
        db.session.add(PaperSummary(paper_id=ids["coll"], summary_text="Old summary."))
        db.session.commit()
    html = logged_in.get("/projects/pa-source").get_data(as_text=True)
    assert "Old summary." in html and 'class="summary-src' not in html
