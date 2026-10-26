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


def test_elsevier_pii_from_sciencedirect_cell_and_lancet_links():
    assert crawl._elsevier_pii(SD_URL) == "S0263224125022432"
    assert crawl._elsevier_pii("https://www.cell.com/iscience/fulltext/S2589-0042(25)01350-1") == "S2589004225013501"
    assert crawl._elsevier_pii(
        "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)00001-2/fulltext") == "S0140673624000012"
    assert crawl._elsevier_pii("https://www.frontiersin.org/journals/x/articles/10.3389/fmicb.2024.1/full") is None


def test_cell_link_is_looked_up_by_its_pii(monkeypatch):
    monkeypatch.setattr(crawl, "_is_safe_external_url", lambda url: True)
    seen = []

    def handler(request):
        if request.url.host == "api.crossref.org" and request.url.path == "/works":
            seen.append(request.url.params["filter"])
        if request.url.host == "www.cell.com":
            return httpx.Response(403, text="<title>Just a moment...</title>")
        return _handler()(request)
    _patch_http(monkeypatch, handler)
    info = crawl.extract_paper_info_from_url("https://www.cell.com/iscience/fulltext/S2589-0042(25)01350-1")
    assert seen == ["alternative-id:S2589004225013501"]
    assert info["doi"] == DOI and info["title"] == "Deep learning for Raman spectroscopy"


def test_doi_link_to_a_blocking_publisher_uses_the_doi(monkeypatch):
    monkeypatch.setattr(crawl, "_is_safe_external_url", lambda url: True)

    def handler(request):
        if request.url.host == "doi.org":
            return httpx.Response(403, text="<title>Just a moment...</title>")
        if request.url.host == "api.openalex.org":
            assert request.url.path == "/works/doi:10.1002/jrs.6402"
        return _handler()(request)
    _patch_http(monkeypatch, handler)
    info = crawl.extract_paper_info_from_url("https://doi.org/10.1002/jrs.6402")
    assert info["doi"] == "10.1002/jrs.6402" and info["abstract"]


def test_retry_repairs_a_paper_added_from_a_block_page(app, monkeypatch):
    """Papers added before this fix have the block page's title and no abstract: the Retry button and the
    nightly backfill (both _enrich_one_web_paper) fill them in and rewrite the summary."""
    import json
    from app import db
    from app.models import Paper
    info = {"title": "Deep learning for Raman spectroscopy", "abstract": "Raman spectra are useful",
            "authors": ["Christoph Lange"], "doi": DOI, "pdf_url": None}
    monkeypatch.setattr(crawl, "extract_paper_info_from_url", lambda url: info)
    summaries = []
    monkeypatch.setattr(crawl, "write_summary", lambda paper, content: summaries.append(content))
    with app.app_context():
        p = Paper(arxiv_id="web:blocked00001", title="Just a moment...", authors="[]", abstract="",
                  pdf_url=SD_URL, source="web", web_enrich_attempts=3)
        db.session.add(p)
        db.session.commit()
        try:
            crawl._enrich_one_web_paper(p)
            db.session.commit()
            assert (p.title, p.abstract, json.loads(p.authors), p.doi) == (
                info["title"], info["abstract"], ["Christoph Lange"], DOI)
            assert summaries == ["Raman spectra are useful"]

            crawl._enrich_one_web_paper(p)  # has an abstract now: left alone
            assert summaries == ["Raman spectra are useful"]
        finally:
            db.session.delete(p)
            db.session.commit()
