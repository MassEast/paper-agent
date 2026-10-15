"""
Where the paper's full text comes from, and what happens when a paper was screened without it.

- fetch_paper_full_text: arXiv's own HTML first, then ar5iv; ar5iv's redirect to the abstract page for
  papers it hasn't converted yet must not pass for the full text (it did, and the chat, the deep check and
  the summary read the abstract page as "full text").
- get_paper_content falls back to the PDF.
- refresh_from_full_text (nightly): New Papers screened from the abstract are re-read once the full text
  can be fetched; rejects go to Trash with a reason, My Collection is never touched.

HTTP is a MockTransport; LLM calls are mocked.
"""

import contextlib
from datetime import datetime, timedelta
from unittest.mock import patch

import httpx
import pytest

from app import STALE_PAPER_TEXT_SQL, crawl, db, utils
from app.models import Paper, PaperSummary, PaperText, Project, ProjectPaper, ScreenedPaper

ARTICLE = "<article class='ltx_document'>" + "<p>Section 1. We study circuits.</p>" * 100 + "</article>"
PAGE = f"<html><nav>Report GitHub Issue</nav>{ARTICLE}<footer>About arXiv</footer></html>"
ABS_PAGE = "<html>Skip to main content " + "Abstract: circuits. " * 200 + "Submission history</html>"


def _web(monkeypatch, routes):
    """routes: url -> httpx.Response, or a str naming the url to redirect to. Records requested urls."""
    seen = []

    def handler(request):
        url = str(request.url)
        seen.append(url)
        target = routes.get(url)
        if target is None:
            return httpx.Response(404)
        if isinstance(target, str):
            return httpx.Response(307, headers={"location": target})
        return target

    real = httpx.Client
    monkeypatch.setattr(utils, "_http_client",
                        lambda public_only=False, **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(utils, "_ArxivLockCtx", contextlib.nullcontext)
    return seen


def test_arxiv_html_comes_first_and_only_the_article_is_kept(monkeypatch):
    seen = _web(monkeypatch, {"https://arxiv.org/html/2610.04017": httpx.Response(200, text=PAGE)})
    text = utils.fetch_paper_full_text("2610.04017")
    assert text.startswith("Section 1. We study circuits.") and "GitHub" not in text and "About arXiv" not in text
    assert seen == ["https://arxiv.org/html/2610.04017"]


def test_ar5iv_redirect_to_the_abstract_page_is_not_the_full_text(monkeypatch):
    _web(monkeypatch, {
        "https://ar5iv.labs.arxiv.org/html/2610.04017": "https://arxiv.org/abs/2610.04017",
        "https://arxiv.org/abs/2610.04017": httpx.Response(200, text=ABS_PAGE),
    })
    assert utils.fetch_paper_full_text("2610.04017") is None


def test_ar5iv_when_arxiv_has_no_html(monkeypatch):
    _web(monkeypatch, {"https://ar5iv.labs.arxiv.org/html/1706.03762": httpx.Response(200, text=PAGE)})
    assert utils.fetch_paper_full_text("1706.03762").startswith("Section 1.")


def test_paper_content_falls_back_to_the_pdf(monkeypatch):
    seen = _web(monkeypatch, {})
    monkeypatch.setattr(utils, "fetch_pdf_text", lambda url, arxiv, max_chars=80000: "PDF TEXT" if arxiv else None)
    assert crawl.get_paper_content("2610.14998") == "PDF TEXT"
    assert seen == ["https://arxiv.org/html/2610.14998", "https://ar5iv.labs.arxiv.org/html/2610.14998"]


def test_unreadable_pdf_is_no_text(monkeypatch):
    _web(monkeypatch, {"https://arxiv.org/pdf/x": httpx.Response(200, content=b"%PDF-1.4 garbage")})
    assert utils.fetch_pdf_text("https://arxiv.org/pdf/x", arxiv=True) is None


def test_stored_abstract_pages_are_dropped(app):
    with app.app_context():
        paper = Paper(arxiv_id="2610.09001", title="Stale", abstract="a", source="arxiv")
        good = Paper(arxiv_id="2610.09002", title="Good", abstract="a", source="arxiv")
        db.session.add_all([paper, good])
        db.session.flush()
        db.session.add_all([PaperText(paper_id=paper.id, text=ABS_PAGE, source="html"),
                            PaperText(paper_id=good.id, text="Section 1. " * 300, source="html")])
        db.session.commit()
        db.session.execute(db.text(STALE_PAPER_TEXT_SQL))
        db.session.commit()
        assert PaperText.query.filter_by(paper_id=paper.id).count() == 0
        assert PaperText.query.filter_by(paper_id=good.id).count() == 1
        PaperText.query.filter_by(paper_id=good.id).delete()
        for p in (paper, good):
            db.session.delete(p)
        db.session.commit()


# ── Nightly full-text re-read ────────────────────────────────────────────────

@pytest.fixture
def proj(app):
    with app.app_context():
        p = Project(name="Full Text Proj", slug="full-text-proj", research_interest="circuits")
        db.session.add(p)
        db.session.commit()
        pid = p.id
    yield pid
    with app.app_context():
        ScreenedPaper.query.filter_by(project_id=pid).delete()
        for pp in ProjectPaper.query.filter_by(project_id=pid):
            PaperSummary.query.filter_by(paper_id=pp.paper_id).delete()
            db.session.delete(pp)
            db.session.flush()
            Paper.query.filter_by(id=pp.paper_id).delete()
        Project.query.filter_by(id=pid).delete()
        db.session.commit()


def _link(pid, arxiv_id, screen_source, manual_tag=None, summary_source="abstract", age_days=0):
    paper = Paper(arxiv_id=arxiv_id, title=f"Paper {arxiv_id}", abstract="About circuits.", source="arxiv")
    db.session.add(paper)
    db.session.flush()
    db.session.add(PaperSummary(paper_id=paper.id, summary_text="old", content_source=summary_source,
                                model_used="m", tokens_used=5))
    pp = ProjectPaper(project_id=pid, paper_id=paper.id, manual_tag=manual_tag, screen_source=screen_source,
                      relevance_reason="old why", added_at=datetime.utcnow() - timedelta(days=age_days))
    db.session.add(pp)
    db.session.commit()
    return pp.id


def _refresh(pid, verdicts, texts=None, **kw):
    """verdicts: arxiv_id -> "relevant" / "not-relevant"; texts: arxiv_id -> full text (default: some)."""
    texts = texts or {}
    deep_calls = []

    def deep(title, abstract, content, ri, refs):
        arxiv_id = title.removeprefix("Paper ")
        deep_calls.append(arxiv_id)
        return verdicts[arxiv_id] == "relevant", verdicts[arxiv_id], "m", 1, 10, "new why"

    with patch("app.crawl.get_paper_content", side_effect=lambda a: texts.get(a, "FULL TEXT " * 500)), \
         patch("app.crawl._screen_paper_deep", side_effect=deep), \
         patch("app.crawl.generate_summary_with_full_content", return_value=({"summary": "new"}, "m", 1, 100)), \
         patch("app.crawl.extract_main_contributions", return_value=("c", "", 1, 50)):
        stats = crawl.refresh_from_full_text(pid, **kw)
    return stats, deep_calls


def test_new_papers_screened_from_the_abstract_are_reread(app, proj):
    with app.app_context():
        reject = _link(proj, "2610.10001", "abstract")
        keep = _link(proj, "2610.10002", "unknown")
        stats, _ = _refresh(proj, {"2610.10001": "not-relevant", "2610.10002": "relevant"})
        assert stats["trashed"] == 1 and stats["kept"] == 1 and stats["resummarized"] == 1

        r = db.session.get(ProjectPaper, reject)
        assert r.trashed_at and r.trashed_reason == "Not relevant after reading the full text"
        k = db.session.get(ProjectPaper, keep)
        assert not k.trashed_at and k.screen_source == "full_text" and k.relevance_reason == "new why"
        assert k.paper.summary.summary_text == "new" and k.paper.summary.content_source == "full_text"
        assert k.paper.summary.tokens_used == 150
        assert ScreenedPaper.query.filter_by(project_id=proj, arxiv_id="2610.10001", is_relevant=False).count() == 1

        # done: nothing left for the next night
        assert _refresh(proj, {})[0]["total"] == 0


def test_my_collection_full_text_and_unrecorded_papers_are_left_alone(app, proj):
    with app.app_context():
        _link(proj, "2610.10011", "abstract", manual_tag="related")
        _link(proj, "2610.10012", "full_text")
        _link(proj, "2610.10013", None)  # screened before this was recorded, not marked
        stats, deep_calls = _refresh(proj, {})
        assert stats["total"] == 0 and deep_calls == []


def test_no_full_text_yet_is_retried_and_given_up_after_three_nights(app, proj):
    with app.app_context():
        link = _link(proj, "2610.10021", "abstract")
        for night in range(1, 4):
            stats, _ = _refresh(proj, {}, texts={"2610.10021": None})
            assert stats["no_text"] == 1
            assert db.session.get(ProjectPaper, link).paper.full_text_attempts == night
        assert _refresh(proj, {}, texts={"2610.10021": None})[0]["total"] == 0
        assert not db.session.get(ProjectPaper, link).trashed_at


def test_summary_already_from_full_text_is_kept(app, proj):
    with app.app_context():
        link = _link(proj, "2610.10031", "abstract", summary_source="full_text")
        stats, _ = _refresh(proj, {"2610.10031": "relevant"})
        assert stats["resummarized"] == 0
        assert db.session.get(ProjectPaper, link).paper.summary.summary_text == "old"


def test_newest_first_and_capped_per_night(app, proj):
    with app.app_context():
        _link(proj, "2610.10041", "abstract", age_days=3)
        _link(proj, "2610.10042", "abstract", age_days=1)
        _link(proj, "2610.10043", "abstract", age_days=2)
        _, deep_calls = _refresh(proj, {a: "relevant" for a in ("2610.10041", "2610.10042", "2610.10043")}, limit=2)
        assert sorted(deep_calls) == ["2610.10042", "2610.10043"]


def test_trash_shows_the_reason_and_restore_clears_it(app, proj, client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    with app.app_context():
        link = _link(proj, "2610.10051", "abstract")
        _refresh(proj, {"2610.10051": "not-relevant"})
    html = client.get("/projects/full-text-proj").get_data(as_text=True)
    assert "Not relevant after reading the full text" in html
    client.post("/paper/2610.10051/restore", data={"project_id": proj})
    with app.app_context():
        pp = db.session.get(ProjectPaper, link)
        assert not pp.trashed_at and pp.trashed_reason is None
        # read with the full text already: not re-read (and re-trashed) the next night
        assert _refresh(proj, {})[0]["total"] == 0


def test_card_says_what_the_relevance_check_read(app, proj, client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    with app.app_context():
        _link(proj, "2610.10061", "abstract")
    html = client.get("/projects/full-text-proj").get_data(as_text=True)
    reason = html[html.index("Why it's relevant for this project"):]
    assert reason.index("from abstract only") < reason.index("old why")
