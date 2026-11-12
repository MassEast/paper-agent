"""Several people in one project (app/members.py, Project.per_person; built 2026-11-02 for LLM Anatomy).

My Collection and Trash are shared; New Papers, tags, votes and the paper chat are per person; a paper someone
else collected or trashed stays in your New Papers until you've acted on it (kept it there or moved it). Notes: one per person, or one shared
text whose runs remember their author. Each person gets their own email whose links say who they are."""

import json
import re
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from app import crawl, db, members
from app.models import ChatMessage, Paper, Project, ProjectMember, ProjectPaper

SLUG = "pp-proj"


@pytest.fixture
def team(app):
    """A per-person project with Justus and Felix (from the notification emails) and three New Papers."""
    with app.app_context():
        project = Project(name="Team Proj", slug=SLUG, per_person=True,
                          notification_emails="justus.westerhoff@example.org,FelixMaria.Doe@example.org")
        db.session.add(project)
        db.session.flush()
        ProjectMember.query.filter_by(project_id=project.id).delete()  # left by a deleted project with this id
        members.sync_from_emails(project)
        ids = {}
        for i in range(3):
            p = Paper(arxiv_id=f"2611.0000{i}", title=f"Paper {i}", source="arxiv")
            db.session.add(p)
            db.session.flush()
            ids[i] = p.id
            db.session.add(ProjectPaper(project_id=project.id, paper_id=p.id,
                                        added_at=datetime.utcnow() - timedelta(days=i)))
        db.session.commit()
        people = {m.name: m.id for m in members.members(project)}
        pid = project.id
    yield {"id": pid, "people": people, "papers": ids}
    with app.app_context():
        pps = ProjectPaper.query.filter_by(project_id=pid).all()
        ChatMessage.query.filter(ChatMessage.project_paper_id.in_([pp.id for pp in pps])).delete()
        for pp in pps:
            db.session.delete(pp)
        ProjectMember.query.filter_by(project_id=pid).delete()
        Paper.query.filter(Paper.id.in_(ids.values())).delete()
        Project.query.filter_by(id=pid).delete()
        db.session.commit()


def _as(client, team, name):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    client.set_cookie(f"rw_member_{team['id']}", str(team["people"][name]))
    return client


def _post(client, path, **data):
    return client.post(path, data={"project_id": data.pop("project_id"), **data}, headers={"HX-Request": "true"})


def _new_titles(client):
    html = client.get(f"/projects/{SLUG}").get_data(as_text=True)
    section = html.split('id="new-section"', 1)[1].split('id="col-section"', 1)[0]
    return re.findall(r">\s*(Paper \d)\s*</a>", section), html


def test_people_come_from_the_notification_emails(team):
    assert set(team["people"]) == {"Justus", "Felix"}
    assert members.name_from_email("stephan.mueller@example.org") == "Stephan"


def test_a_paper_collected_by_one_stays_in_the_others_new_papers_until_seen(client, team):
    pid = team["id"]
    c = _as(client, team, "Justus")
    _post(c, "/paper/2611.00000/tag", project_id=pid, tag="related")
    assert "Paper 0" not in _new_titles(c)[0]

    c = _as(client, team, "Felix")
    titles, html = _new_titles(c)
    assert titles[0] == "Paper 0"
    assert "added this to My Collection" in html and "Justus" in html
    _post(c, "/paper/2611.00000/seen", project_id=pid)  # Keep in My Collection
    assert "Paper 0" not in _new_titles(c)[0]

    # Felix's keep is not a move: it doesn't show up in Justus's New Papers
    assert "Paper 0" not in _new_titles(_as(client, team, "Justus"))[0]


def test_a_paper_trashed_by_one_can_be_put_back_by_the_other(client, team):
    pid = team["id"]
    _post(_as(client, team, "Justus"), "/paper/2611.00001/trash", project_id=pid)
    c = _as(client, team, "Felix")
    titles, html = _new_titles(c)
    assert "Paper 1" in titles and "moved this to Trash" in html
    _post(c, "/paper/2611.00001/restore", project_id=pid)  # Put back into New Papers
    titles, html = _new_titles(_as(client, team, "Justus"))
    assert "Paper 1" in titles and "put this back into New Papers" in html
    _post(c, "/paper/2611.00001/seen", project_id=pid)  # Justus: OK
    titles, html = _new_titles(c)
    assert "Paper 1" in titles and "put this back" not in html  # still a New Paper, the note is gone
    with client.application.app_context():
        pp = ProjectPaper.query.filter_by(project_id=pid, paper_id=team["papers"][1]).one()
        assert pp.trashed_at is None and pp.moved_by_id == team["people"]["Felix"]


def test_tags_are_per_person_and_the_crawl_counts_anyones(client, team, app):
    pid = team["id"]
    _post(_as(client, team, "Justus"), "/paper/2611.00002/tag", project_id=pid, tag="important")
    _post(_as(client, team, "Felix"), "/paper/2611.00002/tag", project_id=pid, tag="to_discuss")
    with app.app_context():
        pp = ProjectPaper.query.filter_by(project_id=pid, paper_id=team["papers"][2]).one()
        ju = db.session.get(ProjectMember, team["people"]["Justus"])
        fe = db.session.get(ProjectMember, team["people"]["Felix"])
        assert members.tags_for(pp, ju) == ["important"]
        assert members.tags_for(pp, fe) == ["to_discuss"]
        assert pp.any_tags == {"important", "to_discuss"}
        assert crawl._collection_tag_tier(pp) == 0  # "important" from anyone puts it in the top tier
    html = _as(client, team, "Felix").get(f"/projects/{SLUG}").get_data(as_text=True)
    assert "person-mark" in html and "Important" in html  # Felix sees Justus's tag


def test_tags_from_before_per_person_mode_count_for_everyone_until_they_change_them(team, app):
    with app.app_context():
        pp = ProjectPaper.query.filter_by(project_id=team["id"], paper_id=team["papers"][0]).one()
        pp.paper_tags = '["to_read"]'
        fe = db.session.get(ProjectMember, team["people"]["Felix"])
        assert members.tags_for(pp, fe) == ["to_read"]
        members.set_tags(pp, fe, [])
        assert members.tags_for(pp, fe) == []
        db.session.rollback()


def test_votes_are_per_person_and_everyone_sees_them(client, team, app):
    pid = team["id"]
    _post(_as(client, team, "Justus"), "/paper/2611.00000/vote", project_id=pid, vote=4)
    _post(_as(client, team, "Felix"), "/paper/2611.00000/vote", project_id=pid, vote=5)
    html = _as(client, team, "Felix").get(f"/projects/{SLUG}").get_data(as_text=True)
    assert "⌀ 4.5 (2)" in html and "★4" in html
    _post(_as(client, team, "Felix"), "/paper/2611.00000/vote", project_id=pid, vote=5)  # same again: taken back
    with app.app_context():
        pp = ProjectPaper.query.filter_by(project_id=pid, paper_id=team["papers"][0]).one()
        assert [v for _, v in members.votes(pp)] == [4]


def test_a_vote_needs_a_person(client, team):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    r = client.post("/paper/2611.00000/vote", data={"project_id": team["id"], "vote": 3})
    assert r.status_code == 400


def test_shared_notes_keep_who_wrote_which_run(client, team, app):
    pid = team["id"]
    ju, fe = team["people"]["Justus"], team["people"]["Felix"]
    c = _as(client, team, "Felix")
    runs = [[ju, "Good "], [ju, "baseline. "], [fe, "Agreed."], [fe, ""]]
    assert c.post("/paper/2611.00000/notes-runs", data={"project_id": pid, "runs": json.dumps(runs)}).status_code == 204
    with app.app_context():
        pp = ProjectPaper.query.filter_by(project_id=pid, paper_id=team["papers"][0]).one()
        assert members.runs(pp) == [[ju, "Good baseline. ", None], [fe, "Agreed.", None]]
        assert pp.notes == "Good baseline. Agreed."  # plain text for export, Copy / Move and the chat
    assert c.post("/paper/2611.00000/notes-runs", data={"project_id": pid, "runs": json.dumps([[999999, "x"]])}).status_code == 400
    html = c.get(f"/projects/{SLUG}").get_data(as_text=True)
    assert 'class="rich-notes"' in html


def test_each_run_of_a_note_keeps_the_day_it_was_written(client, team, app):
    """Since 2026-11-04 (hover shows "Name, Sep 28, 2026"); the same person on two days stays two runs."""
    pid = team["id"]
    ju = team["people"]["Justus"]
    c = _as(client, team, "Justus")
    runs = [[ju, "Old part. ", "2026-09-28"], [ju, "New ", "2026-11-04"], [ju, "part.", "2026-11-04"]]
    assert c.post("/paper/2611.00000/notes-runs", data={"project_id": pid, "runs": json.dumps(runs)}).status_code == 204
    with app.app_context():
        pp = ProjectPaper.query.filter_by(project_id=pid, paper_id=team["papers"][0]).one()
        assert members.runs(pp) == [[ju, "Old part. ", "2026-09-28"], [ju, "New part.", "2026-11-04"]]
    bad = [[ju, "x", "yesterday"]]
    assert c.post("/paper/2611.00000/notes-runs", data={"project_id": pid, "runs": json.dumps(bad)}).status_code == 400


def test_the_chat_is_per_person(team, app):
    with app.app_context():
        pp = ProjectPaper.query.filter_by(project_id=team["id"], paper_id=team["papers"][0]).one()
        ju, fe = team["people"]["Justus"], team["people"]["Felix"]
        db.session.add_all([ChatMessage(project_paper_id=pp.id, role="user", content="old", member_id=None),
                            ChatMessage(project_paper_id=pp.id, role="user", content="mine", member_id=ju),
                            ChatMessage(project_paper_id=pp.id, role="user", content="his", member_id=fe)])
        db.session.commit()
        from app import chat
        assert sorted(m.content for m in chat._msgs(pp, ju)) == ["mine", "old"]
        assert sorted(m.content for m in chat._msgs(pp, fe)) == ["his", "old"]


def test_who_are_you_until_chosen_then_remembered(client, team):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    html = client.get(f"/projects/{SLUG}").get_data(as_text=True)
    assert re.search(r'<div id="whoami"\s+role="dialog"', html)  # open: nobody chosen yet
    r = client.post(f"/projects/{SLUG}/whoami", data={"member_id": team["people"]["Felix"]})
    assert r.status_code == 302
    html = client.get(f"/projects/{SLUG}").get_data(as_text=True)
    assert '<div id="whoami" hidden' in html and "You: Felix" in html


def test_nobody_adds_themselves(client, team, app):
    """The people are the notification list; the dialog asks anyone else not to change anything."""
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    html = client.get(f"/projects/{SLUG}").get_data(as_text=True)
    assert "please don't change anything" in html and 'name="name"' not in html
    client.post(f"/projects/{SLUG}/whoami", data={"name": "Stephan"})
    with app.app_context():
        assert ProjectMember.query.filter_by(project_id=team["id"]).count() == 2


def test_colors_are_given_not_chosen(client, team, app):
    c = _as(client, team, "Felix")
    fe = team["people"]["Felix"]
    c.post(f"/projects/{SLUG}/members/{fe}", data={"name": "Felix G", "color": members.COLORS[-1]})
    with app.app_context():
        m = db.session.get(ProjectMember, fe)
        assert m.name == "Felix G" and m.color == members.COLORS[1]  # second person, second color


def test_an_email_link_says_who_clicked(client, team):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    fe = team["people"]["Felix"]
    r = client.get(f"/projects/{SLUG}?as={fe}&paper={team['papers'][0]}")
    assert r.status_code == 302 and "as=" not in r.location and f"paper={team['papers'][0]}" in r.location
    assert "You: Felix" in client.get(r.location).get_data(as_text=True)


def test_everyone_gets_their_own_email_with_their_own_links(team, app):
    with app.app_context():
        paper = Paper.query.get(team["papers"][0])
        with patch("app.crawl.send_notification", return_value=True) as send, \
                patch.object(crawl, "_fetch_figure_html", return_value=""):
            crawl.notify_crawl_complete("Team Proj", SLUG, 3, 1, [paper], project_id=team["id"], added_papers=[paper])
    assert send.call_count == 2
    to = [c.kwargs["project_emails"] for c in send.call_args_list]
    assert to == ["justus.westerhoff@example.org", "FelixMaria.Doe@example.org"]
    fe_html = send.call_args_list[1].kwargs["content_html"]
    assert f"/projects/{SLUG}?as={team['people']['Felix']}&paper=2611.00000" in fe_html
    assert f'/projects/{SLUG}?as={team["people"]["Felix"]}"' in fe_html  # "View and tag papers"


def test_personal_links_leave_other_urls_alone():
    d = "https://x.org/projects/p"
    out = crawl._personal_links(f'<a href="{d}?paper=3">t</a> <a href="{d}">v</a> {d}#keywords https://arxiv.org/abs/1', d, 7)
    assert out == (f'<a href="{d}?as=7&paper=3">t</a> <a href="{d}?as=7">v</a> {d}?as=7#keywords '
                   "https://arxiv.org/abs/1")


def test_a_project_without_per_person_mode_is_unchanged(client, team, app):
    with app.app_context():
        db.session.get(Project, team["id"]).per_person = False
        db.session.commit()
    c = _as(client, team, "Justus")
    html = c.get(f"/projects/{SLUG}").get_data(as_text=True)
    assert 'id="whoami"' not in html and "You:" not in html and "vote-star" not in html
    _post(c, "/paper/2611.00000/tag", project_id=team["id"], tag="important")
    with app.app_context():
        pp = ProjectPaper.query.filter_by(project_id=team["id"], paper_id=team["papers"][0]).one()
        assert pp.tags_list == ["important"] and pp.member_states == [] and pp.moved_by_id is None


def test_a_resize_saves_only_the_height_not_the_text(client, team, app):
    """The notes box's resize watcher used to send the text it had at page load too, so hiding a card
    (carousel, tab) re-saved an old text over a note someone else had saved meanwhile (found 2026-11-02)."""
    pid = team["id"]
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    client.post("/paper/2611.00000/notes", data={"project_id": pid, "notes": "Justus's note"})
    client.post("/paper/2611.00000/notes", data={"project_id": pid, "notes_height": 120})
    with app.app_context():
        pp = ProjectPaper.query.filter_by(project_id=pid, paper_id=team["papers"][0]).one()
        assert (pp.notes, pp.notes_height) == ("Justus's note", 120)


def test_my_collection_sorts_by_average_vote(client, team, app):
    pid = team["id"]
    c = _as(client, team, "Justus")
    for i, v in ((0, 2), (1, 5)):
        _post(c, f"/paper/2611.0000{i}/tag", project_id=pid, tag="related")
        _post(c, f"/paper/2611.0000{i}/vote", project_id=pid, vote=v)
    _post(c, "/paper/2611.00002/tag", project_id=pid, tag="related")  # no vote: last
    html = c.get(f"/projects/{SLUG}?col_by=vote").get_data(as_text=True)
    assert 'id="col-sort-vote"' in html
    section = html.split('id="curated-list"', 1)[1]
    assert re.findall(r">\s*(Paper \d)\s*</a>", section) == ["Paper 1", "Paper 0", "Paper 2"]


def test_a_note_edited_with_per_person_mode_off_is_not_lost_when_it_is_on_again(team, app):
    with app.app_context():
        pp = ProjectPaper.query.filter_by(project_id=team["id"], paper_id=team["papers"][0]).one()
        pp.notes_runs = json.dumps([[team["people"]["Felix"], "old"]])
        pp.notes = "old, then edited with the mode off"
        assert members.runs(pp) == [[None, "old, then edited with the mode off", None]]
        db.session.rollback()


# ── Turning per-person mode off and on: round trips change nothing ───────────────────────────────────────

def _views(app, team):
    """What everyone sees: per person (section and tags of every paper) and shared (section and tags)."""
    with app.app_context():
        project = db.session.get(Project, team["id"])
        pps = sorted(ProjectPaper.query.filter_by(project_id=team["id"]).all(), key=lambda pp: pp.id)
        people = members.members(project)
        per_person = {m.name: [(members.section_for(pp, m), sorted(members.tags_for(pp, m))) for pp in pps]
                      for m in people}
        shared = [(members.shared_section(pp), sorted(pp.tags_list)) for pp in pps]
        return per_person, shared


def _switch(client, on):
    r = client.post(f"/projects/{SLUG}/per-person", data={"enabled": "1" if on else "0"})
    assert r.status_code == 204


def _some_history(client, team, app):
    """Shared tags from before, then each person's own: collected, tagged differently, one tag removed."""
    pid = team["id"]
    with app.app_context():
        for i in (0, 1):
            pp = ProjectPaper.query.filter_by(project_id=pid, paper_id=team["papers"][i]).one()
            pp.paper_tags = '["to_read"]'
            pp.manual_tag = "related"
        db.session.commit()
    _post(_as(client, team, "Justus"), "/paper/2611.00000/tag", project_id=pid, tag="important")
    _post(_as(client, team, "Justus"), "/paper/2611.00001/tag", project_id=pid, tag="to_read")  # removes it
    _post(_as(client, team, "Felix"), "/paper/2611.00002/tag", project_id=pid, tag="to_discuss")  # collects it


def test_on_off_on_changes_nothing(client, team, app):
    _some_history(client, team, app)
    before = _views(app, team)[0]
    assert before["Justus"][0] == ("col", ["important", "to_read"]) and before["Felix"][0] == ("col", ["to_read"])
    assert before["Felix"][2] == ("col", ["to_discuss"]) and before["Justus"][2][0] == "new"  # Felix's move, unseen
    _switch(client, False)
    _switch(client, True)
    assert _views(app, team)[0] == before


def test_off_on_off_changes_nothing(client, team, app):
    _some_history(client, team, app)
    _switch(client, False)
    before = _views(app, team)[1]
    _switch(client, True)
    _switch(client, False)
    assert _views(app, team)[1] == before


def test_turning_it_off_keeps_every_tag_anyone_sees(client, team, app):
    _some_history(client, team, app)
    _switch(client, False)
    shared = _views(app, team)[1]
    assert shared[0] == ("col", ["important", "to_read"])
    assert shared[1] == ("col", ["to_read"])  # Justus removed it, but Felix still sees the old one
    assert shared[2] == ("col", ["to_discuss"])


def test_tags_changed_with_the_mode_off_apply_to_everyone_when_it_is_on_again(client, team, app):
    _some_history(client, team, app)
    _switch(client, False)
    _post(_as(client, team, "Felix"), "/paper/2611.00000/tag", project_id=team["id"], tag="to_discuss")
    _switch(client, True)
    per_person = _views(app, team)[0]
    assert per_person["Justus"][0][1] == per_person["Felix"][0][1] == ["important", "to_discuss", "to_read"]
    assert per_person["Felix"][2][1] == ["to_discuss"]  # untouched paper: own tags as before


def test_a_paper_moved_with_the_mode_off_is_nobodys_move(client, team, app):
    _some_history(client, team, app)
    _switch(client, False)
    _post(_as(client, team, "Felix"), "/paper/2611.00002/trash", project_id=team["id"])
    _switch(client, True)
    per_person = _views(app, team)[0]
    assert per_person["Justus"][2][0] == per_person["Felix"][2][0] == "trash"  # not "Felix moved this" in New


@pytest.mark.slow
def test_typing_in_a_note_shows_name_and_day_on_hover(live_server, browser_session, team, app):
    """rich_notes.js in a real browser: new text gets today's date, the hover says "Justus, Nov 4, 2026"."""
    from datetime import date
    browser, state = browser_session
    context = browser.new_context(storage_state=state)
    page = context.new_page()
    page.goto(f"{live_server}/projects/{SLUG}?as={team['people']['Justus']}")
    page.wait_for_function("window.htmx !== undefined")
    note = page.locator(".rich-notes:visible").first
    note.click()
    page.keyboard.type("Nice idea.")
    page.wait_for_function("document.querySelector('.rich-notes-status').textContent === 'saved'")
    label = date.today().strftime("%b %-d, %Y")
    assert note.locator("span").first.get_attribute("data-tip") == f"Justus, {label}"
    with app.app_context():
        pps = ProjectPaper.query.filter_by(project_id=team["id"]).all()
        runs = [members.runs(pp) for pp in pps if pp.notes]
        assert runs == [[[team["people"]["Justus"], "Nice idea.", date.today().isoformat()]]]
    context.close()
