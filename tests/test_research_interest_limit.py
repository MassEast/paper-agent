"""The research interest goes into every screening prompt, so saving one is limited to
RESEARCH_INTEREST_MAX_CHARS (a pasted 48k-char design doc cost one project 5-7M tokens a night, 2026-10).
Generate/Improve ask the model for at most 300 words, well under the limit."""
from app import db, prompts
from app.models import Project, RESEARCH_INTEREST_MAX_CHARS


def _save(client, project, text):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    return client.post(f"/projects/{project.slug}/research-interest", data={"research_interest": text},
                       headers={"X-Requested-With": "fetch"})


def test_a_research_interest_at_the_limit_is_saved(client, project):
    text = "x" * RESEARCH_INTEREST_MAX_CHARS
    assert _save(client, project, text).status_code == 204
    db.session.expire_all()
    assert Project.query.get(project.id).research_interest == text


def test_a_longer_one_is_refused_and_the_old_one_kept(client, project):
    before = project.research_interest
    r = _save(client, project, "<!DOCTYPE html>" + "x" * 48_000)
    assert r.status_code == 400
    assert "too long (48,015 / 4,000 characters)" in r.get_data(as_text=True)
    db.session.expire_all()
    assert Project.query.get(project.id).research_interest == before


def test_newlines_count_once(client, project):
    """Form posts send a newline as \\r\\n; the field shows it as one character, so it is stored and counted as \\n."""
    text = ("y" * 99 + "\r\n") * 39 + "z" * 100  # 4,000 characters as the field counts them
    assert _save(client, project, text).status_code == 204
    db.session.expire_all()
    stored = Project.query.get(project.id).research_interest
    assert "\r" not in stored and len(stored) == RESEARCH_INTEREST_MAX_CHARS


def test_the_generate_prompts_ask_for_at_most_300_words():
    assert "at most 300 words" in prompts.RESEARCH_INTEREST_FROM_COLLECTION
    assert "at most 300 words" in prompts.RESEARCH_INTEREST_IMPROVE
    assert 300 * 7 < RESEARCH_INTEREST_MAX_CHARS  # ~6-7 characters per English word incl. the space
