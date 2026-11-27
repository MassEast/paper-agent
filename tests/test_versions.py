"""New arXiv versions (app/versions.py): the summary knows its version, Update summary says what changed and keeps
the old summary, and in Collection a new version stays marked for each person until they mark it seen."""
import json
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from app import db, versions
from app.models import Paper, PaperSummary, ProjectPaper
from tests.conftest import make_llm_json_response
from tests.test_per_person import SLUG, _as, _post, team  # noqa: F401 (team is a fixture)


@pytest.fixture(autouse=True)
def _drop_summaries(app, team):
    """The team fixture removes its papers but not their summaries."""
    yield
    with app.app_context():
        PaperSummary.query.filter(PaperSummary.paper_id.in_(team["papers"].values())).delete()
        db.session.commit()


def _collect_with_new_version(app, team, version=3):
    """Paper 0: collected a month ago, v1 summary, arXiv now has v<version> from yesterday."""
    with app.app_context():
        pp = ProjectPaper.query.filter_by(project_id=team["id"], paper_id=team["papers"][0]).one()
        pp.manual_tag, pp.collected_at = "related", datetime.utcnow() - timedelta(days=30)
        p = pp.paper
        p.arxiv_version = 1
        db.session.add(PaperSummary(paper_id=p.id, summary_text="Old summary of v1.", main_contributions="- old"))
        db.session.commit()
        p.arxiv_version, p.arxiv_updated = version, date.today() - timedelta(days=1)
        p.arxiv_comment = "v3: added results on Llama; accepted at ICLR 2027"
        db.session.commit()


def _card(client, pid):
    return client.get(f"/projects/{SLUG}").get_data(as_text=True)


def test_a_summary_records_the_version_it_was_read_from(app, team):
    with app.app_context():
        p = db.session.get(Paper, team["papers"][1])
        p.arxiv_version = 2
        s = PaperSummary(paper_id=p.id, summary_text="From v2.")
        db.session.add(s)
        db.session.commit()
        assert s.summary_version == 2 and not versions.summary_outdated(p)
        p.arxiv_version = 3
        db.session.commit()
        assert versions.summary_outdated(p)
        s.summary_text = "From v3."  # a rewrite records the version again
        db.session.commit()
        assert s.summary_version == 3 and not versions.summary_outdated(p)


def test_text_changes_lists_removed_and_added_sentences_only():
    old = "We study looping in transformers.\nOur method uses a fixed depth of four layers.  Results hold on GPT-2 models."
    new = "We study looping in   transformers. Our method uses an adaptive depth chosen per token. Results hold on GPT-2 models."
    diff = versions.text_changes(old, new)
    assert diff == ("REMOVED: Our method uses a fixed depth of four layers.\n"
                    "ADDED: Our method uses an adaptive depth chosen per token.")
    assert versions.text_changes(old, old.replace("\n", " ")) == ""  # line breaks are no change


def test_version_history_is_read_from_the_abstract_page():
    html = ('<strong><a href="/abs/2611.00000v1">[v1]</a></strong> Mon, 3 Nov 2026 10:00:00 UTC (1,000 KB)<br/>'
            '<strong>[v2]</strong> Thu, 20 Nov 2026 18:30:05 UTC (1,200 KB)')
    resp = type("R", (), {"text": html, "raise_for_status": lambda self: None})()
    with patch("app.versions._http_client") as client, patch("app.versions._ArxivLockCtx"):
        client.return_value.__enter__.return_value.get.return_value = resp
        assert versions.history("2611.00000") == [(1, datetime(2026, 11, 3, 10)), (2, datetime(2026, 11, 20, 18, 30, 5))]


def test_new_version_is_marked_in_collection_until_each_person_marks_it_seen(app, client, team):
    _collect_with_new_version(app, team)
    pid = team["id"]
    for name in ("Justus", "Felix"):
        html = _card(_as(client, team, name), pid)
        assert 'data-new-version="3"' in html and "New version on arXiv: v3" in html
        assert "The summary below is from v1." in html and "Update summary" in html
        assert "accepted at ICLR 2027" in html  # the authors' comment
    _post(_as(client, team, "Justus"), "/paper/2611.00000/version-seen", project_id=pid)
    assert 'data-new-version' not in _card(_as(client, team, "Justus"), pid)
    assert 'data-new-version="3"' in _card(_as(client, team, "Felix"), pid)  # Felix hasn't seen it yet
    # The summary is still from v1: the offer to update stays for everyone
    assert "Update summary" in _card(_as(client, team, "Justus"), pid)


def test_a_version_out_before_the_paper_was_collected_is_not_news(app, client, team):
    _collect_with_new_version(app, team)
    with app.app_context():
        pp = ProjectPaper.query.filter_by(project_id=team["id"], paper_id=team["papers"][0]).one()
        pp.collected_at = datetime.utcnow()
        db.session.commit()
    assert 'data-new-version' not in _card(_as(client, team, "Justus"), team["id"])


def test_update_summary_says_what_changed_and_keeps_the_old_summary(app, client, team):
    _collect_with_new_version(app, team)
    pid = team["id"]
    texts = {"2611.00000": "We loop the middle layers. We also test on GPT-2 models. The results are good overall.",
             "2611.00000v1": "We loop the middle layers. The results are good overall."}

    def llm(messages, **kw):
        prompt = messages[0]["content"]
        if '"bullets"' in prompt:
            assert "ADDED: We also test on GPT-2 models." in prompt and "accepted at ICLR 2027" in prompt and "v1 -> v3" in prompt
            return make_llm_json_response({"bullets": ["Adds GPT-2 experiments.", "Accepted at ICLR 2027."]})
        if '{"reason"' in prompt:
            return make_llm_json_response({"reason": "Why."})
        return make_llm_json_response({"summary": "New summary of v3.", "main_contributions": ["new"], "code_url": None})

    c = _as(client, team, "Justus")
    with patch("app.routes.papers.get_paper_figure", return_value=(None, None)), \
         patch("app.routes.papers.get_citation_count", return_value=None), \
         patch("app.routes.papers.get_paper_content", side_effect=lambda i, *a: texts[i]), \
         patch("app.crawl._llm_json", side_effect=llm), patch("app.llm._llm_json", side_effect=llm), \
         patch("app.resources._llm_json", side_effect=llm):
        html = c.post("/paper/2611.00000/regenerate", data={"project_id": pid},
                      headers={"HX-Request": "true"}).get_data(as_text=True)
    assert "New summary of v3." in html
    assert "What changed from v1 to v3" in html and "Adds GPT-2 experiments." in html
    assert "Summary of v1" in html and "Old summary of v1." in html
    assert "The summary below is from" not in html  # up to date now; the mark stays until "Seen"
    assert 'data-new-version="3"' in html
    with app.app_context():
        s = PaperSummary.query.filter_by(paper_id=team["papers"][0]).one()
        assert s.summary_version == 3 and json.loads(s.version_changes)["from"] == 1


def test_card_links_the_published_version_and_shows_the_journal_ref(app, client, team):
    with app.app_context():
        p = db.session.get(Paper, team["papers"][0])
        p.doi, p.journal_ref = "10.1000/xyz.123", "Transactions on ML Research, 2026"
        q = db.session.get(Paper, team["papers"][1])
        q.doi = "10.48550/arXiv.2611.00001"  # arXiv's own DOI: no Published link
        db.session.commit()
    html = _card(_as(client, team, "Justus"), team["id"])
    assert html.count("Published ↗") == 1 and 'href="https://doi.org/10.1000/xyz.123"' in html
    assert "Transactions on ML Research, 2026" in html
