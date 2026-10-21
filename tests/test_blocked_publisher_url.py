"""
Pasting a publisher URL that blocks server-side fetches (ScienceDirect answers 403 behind Cloudflare):
the block page must not become the paper, and title / abstract / authors come from OpenAlex and Crossref.
"""

import httpx

from app import crawl

SD_URL = "https://www.sciencedirect.com/science/article/pii/S0263224125022432"
DOI = "10.1016/j.measurement.2025.118884"


def _patch_http(monkeypatch, handler):
    real = crawl._http_client
    monkeypatch.setattr(crawl, "_http_client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def _handler(openalex_status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if host == "www.sciencedirect.com":
            return httpx.Response(403, text="<html><title>Just a moment...</title>Cloudflare</html>")
        if host == "api.crossref.org" and path == "/works":
            return httpx.Response(200, json={"message": {"items": [{"DOI": DOI}]}})
        if host == "api.crossref.org":
            return httpx.Response(200, json={"message": {
                "title": ["Crossref title"], "author": [{"given": "Ada", "family": "Lovelace"}]}})
        if host == "api.openalex.org":
            if openalex_status != 200:
                return httpx.Response(openalex_status)
            return httpx.Response(200, json={
                "title": "Deep learning for Raman spectroscopy",
                "abstract_inverted_index": {"Raman": [0, 3], "spectra": [1], "are": [2], "useful": [4]},
                "authorships": [{"author": {"display_name": "Christoph Lange"}}],
            })
        return httpx.Response(404)
    return handler


def test_sciencedirect_block_page_is_ignored_and_openalex_fills_in(monkeypatch):
    monkeypatch.setattr(crawl, "_is_safe_external_url", lambda url: True)
    _patch_http(monkeypatch, _handler())
    info = crawl.extract_paper_info_from_url(SD_URL)
    assert info["title"] == "Deep learning for Raman spectroscopy"
    assert info["abstract"] == "Raman spectra are Raman useful"
    assert info["authors"] == ["Christoph Lange"]
    assert info["doi"] == DOI
    assert not info["full_text"]  # no Cloudflare text


def test_falls_back_to_crossref_when_openalex_has_nothing(monkeypatch):
    monkeypatch.setattr(crawl, "_is_safe_external_url", lambda url: True)
    _patch_http(monkeypatch, _handler(openalex_status=404))
    info = crawl.extract_paper_info_from_url(SD_URL)
    assert info["title"] == "Crossref title"
    assert info["authors"] == ["Ada Lovelace"]
    assert info["abstract"] is None
