"""Scholar 429 handling: recommendations retry, citation fetch pause after repeated 429s."""
from unittest.mock import MagicMock, patch

import app.crawl as crawl


def _resp(status, payload=None):
    r = MagicMock()
    r.status_code = status
    r.text = ""
    r.json.return_value = payload or {}
    return r


def _client_returning(responses):
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.side_effect = responses
    client.get.side_effect = responses
    return client


def test_recommendations_retries_after_429():
    ok = _resp(200, {"recommendedPapers": [{"externalIds": {"ArXiv": "2601.00001"}, "title": "t", "publicationDate": "2026-01-01"}]})
    client = _client_returning([_resp(429), ok])
    with patch.object(crawl.httpx, "Client", return_value=client), patch.object(crawl.time, "sleep") as sleep:
        papers = crawl.get_semantic_scholar_recommendations(["2501.00001"])
    assert [p["arxiv_id"] for p in papers] == ["2601.00001"]
    assert 15 in [c.args[0] for c in sleep.call_args_list]


def test_recommendations_gives_up_after_four_429s():
    client = _client_returning([_resp(429)] * 4)
    with patch.object(crawl.httpx, "Client", return_value=client), patch.object(crawl.time, "sleep"):
        assert crawl.get_semantic_scholar_recommendations(["2501.00001"]) == []
    assert client.post.call_count == 4


def test_citation_fetch_pauses_after_repeated_429s():
    crawl._citation_429_streak = 0
    crawl._citation_paused_until = 0.0
    client = _client_returning([_resp(429)] * 100)
    with patch.object(crawl.httpx, "Client", return_value=client), patch.object(crawl.time, "sleep"):
        for i in range(crawl._CITATION_429_GIVE_UP):
            assert crawl._fetch_citations(f"arXiv:2601.{i:05d}") is None
        calls = client.get.call_count
        assert crawl._fetch_citations("arXiv:2601.99999") is None  # paused: no request made
    assert client.get.call_count == calls
    assert crawl._citation_paused_until > 0
    crawl._citation_paused_until = 0.0
