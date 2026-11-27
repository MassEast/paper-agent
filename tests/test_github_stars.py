"""GitHub stars of a paper's code repository (app/github_stars.py): which links count as a repo, the nightly
refresh (weekly, oldest check first, stops at GitHub's rate limit), and a new code link dropping the old stars."""
from datetime import datetime, timedelta
from unittest.mock import patch

import httpx
import pytest

from app import db, github_stars, resources
from app.models import Paper, Project, ProjectPaper


def test_repo_of():
    assert github_stars.repo_of("https://github.com/MassEast/paper-agent") == "MassEast/paper-agent"
    assert github_stars.repo_of("https://github.com/org/repo.git") == "org/repo"
    assert github_stars.repo_of("https://github.com/org/repo/tree/main/src") == "org/repo"
    assert github_stars.repo_of("https://github.com/someone") is None
    assert github_stars.repo_of("https://github.com/orgs/x/repositories") is None
    assert github_stars.repo_of("https://name.github.io/project") is None
    assert github_stars.repo_of(None) is None


@pytest.fixture
def papers(app):
    with app.app_context():
        project = Project(name="Stars Proj", slug="stars-proj")
        db.session.add(project)
        db.session.flush()
        now = datetime.utcnow()
        rows = {}
        for key, code_url, checked, trashed in (
                ("never", "https://github.com/a/never", None, None),
                ("stale", "https://github.com/a/stale", now - timedelta(days=8), None),
                ("fresh", "https://github.com/a/fresh", now - timedelta(days=1), None),
                ("gone", "https://github.com/a/gone", None, None),
                ("trashed", "https://github.com/a/trashed", None, now),
                ("gitlab", "https://gitlab.com/a/b", None, None)):
            p = Paper(arxiv_id=f"2611.9{len(rows):04d}", title=key, code_url=code_url, github_checked_at=checked,
                      github_stars=1 if checked else None)
            db.session.add(p)
            db.session.flush()
            db.session.add(ProjectPaper(project_id=project.id, paper_id=p.id, trashed_at=trashed))
            rows[key] = p.id
        db.session.commit()
        yield rows
        ProjectPaper.query.filter_by(project_id=project.id).delete()
        Paper.query.filter(Paper.id.in_(rows.values())).delete()
        db.session.delete(project)
        db.session.commit()


def _answer(stars, remaining="50"):
    def handler(request):
        repo = request.url.path.removeprefix("/repos/")
        if repo == "a/gone":
            return httpx.Response(404, headers={"X-RateLimit-Remaining": remaining})
        return httpx.Response(200, json={"stargazers_count": stars[repo]}, headers={"X-RateLimit-Remaining": remaining})
    return handler


def _client(handler):
    real = httpx.Client
    return patch.object(github_stars.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_refresh_fetches_due_repos_only(app, papers):
    with app.app_context(), _client(_answer({"a/never": 3, "a/stale": 1200})):
        assert github_stars.refresh() == 2
        get = lambda k: db.session.get(Paper, papers[k])
        assert (get("never").github_stars, get("stale").github_stars, get("fresh").github_stars) == (3, 1200, 1)
        assert get("gone").github_stars is None and get("gone").github_checked_at is not None  # not asked again soon
        assert get("trashed").github_checked_at is None and get("gitlab").github_checked_at is None


def test_refresh_stops_at_the_rate_limit_and_the_next_run_goes_on(app, papers):
    calls = []

    def limited(request):
        calls.append(request.url.path)
        return httpx.Response(403, json={"message": "API rate limit exceeded"}, headers={"X-RateLimit-Remaining": "0"})
    with app.app_context(), _client(limited):
        assert github_stars.refresh() == 0
        assert len(calls) == 1
        assert db.session.get(Paper, papers["never"]).github_checked_at is None
    calls.clear()
    answer = _answer({"a/never": 3, "a/stale": 1}, remaining="0")
    with app.app_context(), _client(lambda r: calls.append(r) or answer(r)):
        github_stars.refresh()
        assert len(calls) == 1  # the last request allowed: stop after it


def test_a_new_code_link_drops_the_old_stars_and_finding_none_keeps_the_link(app, papers):
    with app.app_context():
        p = db.session.get(Paper, papers["fresh"])
        resources.apply(p, {**resources.EMPTY, "code_url": "https://github.com/a/fresh"})
        assert p.github_stars == 1
        resources.apply(p, {**resources.EMPTY, "code_url": "https://github.com/b/other"})
        assert p.github_stars is None and p.github_checked_at is None
        resources.apply(p, resources.EMPTY)  # a re-read that finds nothing keeps a link added by hand
        assert p.code_url == "https://github.com/b/other"
        db.session.rollback()
