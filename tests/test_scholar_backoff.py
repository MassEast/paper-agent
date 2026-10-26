"""Semantic Scholar 429 handling: quick retries (_scholar_call), citation fetch pause after repeated 429s,
citation counts in one batch request."""
from unittest.mock import MagicMock, patch

import app.crawl as crawl


def _resp(status, payload=None):
    r = MagicMock()
    r.status_code = status
    r.text = ""
    r.json.return_value = payload if payload is not None else {}
    return r


def _client_returning(responses):
    client = MagicMock()
    client.__enter__.return_value = client
    client.request.side_effect = responses
    return client


def _patched(client):
    return patch.object(crawl, "_http_client", return_value=client)


def test_scholar_call_retries_quickly_on_429_and_5xx():
    client = _client_returning([_resp(429), _resp(503), _resp(200)])
    with _patched(client), patch.object(crawl, "_ScholarLockCtx", MagicMock()), \
            patch.object(crawl.time, "sleep") as sleep:
        r = crawl._scholar_call("GET", "https://api.semanticscholar.org/x")
    assert r.status_code == 200
    assert client.request.call_count == 3
    assert all(c.args[0] < 10 for c in sleep.call_args_list)  # a few seconds, not minutes


def test_scholar_call_returns_the_last_429():
    client = _client_returning([_resp(429)] * crawl.SCHOLAR_ATTEMPTS)
    with _patched(client), patch.object(crawl, "_ScholarLockCtx", MagicMock()), patch.object(crawl.time, "sleep"):
        assert crawl._scholar_call("GET", "https://api.semanticscholar.org/x").status_code == 429
    assert client.request.call_count == crawl.SCHOLAR_ATTEMPTS


def test_scholar_call_does_not_retry_a_404():
    client = _client_returning([_resp(404)])
    with _patched(client), patch.object(crawl, "_ScholarLockCtx", MagicMock()):
        assert crawl._scholar_call("GET", "https://api.semanticscholar.org/x").status_code == 404
    assert client.request.call_count == 1


def test_recommendations_give_up_after_all_attempts():
    client = _client_returning([_resp(429)] * crawl.SCHOLAR_ATTEMPTS)
    with _patched(client), patch.object(crawl, "_ScholarLockCtx", MagicMock()), patch.object(crawl.time, "sleep"):
        assert crawl.get_semantic_scholar_recommendations(["2501.00001"]) == []


def test_citation_fetch_pauses_after_repeated_429s():
    crawl._citation_429_streak = 0
    crawl._citation_paused_until = 0.0
    client = _client_returning([_resp(429)] * 1000)
    with _patched(client), patch.object(crawl, "_ScholarLockCtx", MagicMock()), patch.object(crawl.time, "sleep"):
        for i in range(crawl._CITATION_429_GIVE_UP):
            assert crawl._fetch_citations(f"arXiv:2601.{i:05d}") is None
        calls = client.request.call_count
        assert crawl._fetch_citations("arXiv:2601.99999") is None  # paused: no request made
    assert client.request.call_count == calls
    assert crawl._citation_paused_until > 0
    crawl._citation_paused_until = 0.0


def test_citation_counts_come_from_one_batch_request():
    refs = ["arXiv:2601.00001", "DOI:10.1101/2025.01.01.000001", "arXiv:2601.00002"]
    client = _client_returning([_resp(429), _resp(200, [{"citationCount": 5}, None, {"citationCount": None}])])
    with _patched(client), patch.object(crawl, "_ScholarLockCtx", MagicMock()), patch.object(crawl.time, "sleep"):
        counts = crawl.get_citation_counts(refs)
    assert counts == {"arXiv:2601.00001": 5, "arXiv:2601.00002": 0}  # the unknown DOI is left out
    method, url = client.request.call_args.args
    assert (method, url.rsplit("/", 1)[1]) == ("POST", "batch")
    assert client.request.call_args.kwargs["json"] == {"ids": refs}


def test_recommendations_are_asked_for_in_groups_of_ten_seeds():
    """One call has few recent papers among its 500 (ranked by similarity over months); groups of 10 seeds
    each get their own 500, which reaches more recent ones. At most 100 seeds, so at most 10 calls."""
    rec = lambda aid: {"externalIds": {"ArXiv": aid}, "title": aid, "publicationDate": "2026-10-12"}
    responses = [_resp(200, {"recommendedPapers": [rec("2610.00001"), rec("2610.00002")]}),
                 _resp(404),  # one group failing doesn't lose the others
                 _resp(200, {"recommendedPapers": [rec("2610.00002v2"), rec("2610.00003"), {"externalIds": {}}]})]
    client = _client_returning(responses)
    seeds = [f"2501.{i:05d}" for i in range(25)]
    with _patched(client), patch.object(crawl, "_ScholarLockCtx", MagicMock()):
        papers = crawl.get_semantic_scholar_recommendations(seeds)
    calls = client.request.call_args_list
    assert [len(c.kwargs["json"]["positivePaperIds"]) for c in calls] == [10, 10, 5]
    assert {c.kwargs["params"]["limit"] for c in calls} == {500}
    assert [p["arxiv_id"] for p in papers] == ["2610.00001", "2610.00002", "2610.00003"]

    client = _client_returning([_resp(200, {"recommendedPapers": []})] * 20)
    with _patched(client), patch.object(crawl, "_ScholarLockCtx", MagicMock()):
        crawl.get_semantic_scholar_recommendations([f"2501.{i:05d}" for i in range(174)])
    assert client.request.call_count == 10
    assert client.request.call_args_list[0].kwargs["json"]["positivePaperIds"][0] == "ArXiv:2501.00000"  # best first
