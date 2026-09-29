"""
Tests for publisher-fetch politeness: open-access PDF allowlist, SSRF guard on
add-by-URL, Pass-3 backfill retry cap, and the outbound User-Agent header.

No real network calls — httpx.Client/_http_client is mocked throughout.
"""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from tests.conftest import make_llm_json_response


class TestOpenAccessAllowlist:
    def test_non_oa_domain_skips_pdf_download(self, ctx):
        """A non-allowlisted publisher domain must never trigger a PDF fetch."""
        from app.crawl import _enrich_web_paper_pdf_metadata
        from app.models import Paper

        paper = Paper(arxiv_id="web:test1", title="Some Paper")
        with patch("app.crawl._http_client") as mock_client:
            _enrich_web_paper_pdf_metadata(paper, "https://dl.acm.org/doi/pdf/10.1145/fake")
            mock_client.assert_not_called()

        assert paper.page_count is None
        assert paper.institutions is None

    def test_oa_domain_downloads_pdf(self, ctx):
        """An allowlisted OA domain should have its PDF actually fetched and parsed."""
        from app.crawl import _enrich_web_paper_pdf_metadata
        from app.models import Paper
        import io
        from pypdf import PdfWriter

        buf = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.add_blank_page(width=612, height=792)
        writer.write(buf)

        paper = Paper(arxiv_id="web:test2", title="An OA Paper")
        mock_resp = MagicMock(status_code=200, content=buf.getvalue())

        with patch("app.crawl._llm_json", return_value=make_llm_json_response([])):
            with patch("app.crawl._http_client") as mock_client:
                mock_client.return_value.__enter__.return_value.get.return_value = mock_resp
                _enrich_web_paper_pdf_metadata(paper, "https://aclanthology.org/2026.acl-long.1.pdf")
                mock_client.assert_called_once()

        assert paper.page_count == 2

    def test_subdomain_of_allowlisted_host_matches(self):
        from app.crawl import _is_open_access_pdf_domain
        assert _is_open_access_pdf_domain("https://ar5iv.labs.arxiv.org/html/1234.5678")
        assert not _is_open_access_pdf_domain("https://not-arxiv.org.evil.com/x.pdf")


class TestSSRFGuard:
    def test_rejects_loopback(self):
        from app.crawl import extract_paper_info_from_url
        info = extract_paper_info_from_url("http://127.0.0.1/secret")
        assert info["title"] is None
        assert info["abstract"] is None

    def test_rejects_cloud_metadata_address(self):
        from app.crawl import extract_paper_info_from_url
        info = extract_paper_info_from_url("http://169.254.169.254/latest/meta-data/")
        assert info["title"] is None

    def test_rejects_non_http_scheme(self):
        from app.crawl import _is_safe_external_url
        assert not _is_safe_external_url("file:///etc/passwd")
        assert not _is_safe_external_url("ftp://example.com/x")

    def test_accepts_public_host_without_network_call(self):
        """A public-looking hostname passes the guard itself (network call is separately mocked
        in extract_paper_info_from_url's own tests / the slow live suite)."""
        from app.crawl import _is_safe_external_url
        assert _is_safe_external_url("https://example.com/paper")

    def test_fails_closed_on_dns_resolution_error(self):
        """A hostname that can't be resolved at all (typo, dead domain, DNS outage) must be
        treated as unsafe, not silently allowed through."""
        from app.crawl import _is_safe_external_url
        with patch("socket.getaddrinfo", side_effect=OSError("nodename nor servname provided")):
            assert not _is_safe_external_url("https://this-domain-does-not-resolve.invalid/x")


class TestEnrichOneWebPaper:
    def test_increments_attempts_and_updates_metadata_on_oa_match(self, ctx):
        """A single call must always bump the attempt counter/timestamp, and pull in whatever
        extract_paper_info_from_url resolves (new pdf_url, page_count), regardless of whether
        that pdf_url later turns out to be open-access or not."""
        from app.crawl import _enrich_one_web_paper
        from app.models import Paper

        paper = Paper(arxiv_id="web:enrich-unit-test", title="Some Paper",
                      pdf_url="https://dl.acm.org/doi/fake", web_enrich_attempts=0)
        fake_info = {"pdf_url": "https://aclanthology.org/2026.acl-long.1.pdf", "page_count": None}

        with patch("app.crawl.extract_paper_info_from_url", return_value=fake_info) as mock_extract, \
             patch("app.crawl._enrich_web_paper_pdf_metadata") as mock_pdf_enrich:
            _enrich_one_web_paper(paper)
            mock_extract.assert_called_once_with("https://dl.acm.org/doi/fake")
            mock_pdf_enrich.assert_called_once_with(paper, "https://aclanthology.org/2026.acl-long.1.pdf")

        assert paper.web_enrich_attempts == 1
        assert paper.web_enrich_last_attempt_at is not None
        assert paper.pdf_url == "https://aclanthology.org/2026.acl-long.1.pdf"

    def test_does_not_call_pdf_enrich_when_no_pdf_url_resolved(self, ctx):
        """If the landing page still yields no PDF link at all, the attempt still counts but
        there's nothing to download — _enrich_web_paper_pdf_metadata must not be called."""
        from app.crawl import _enrich_one_web_paper
        from app.models import Paper

        paper = Paper(arxiv_id="web:enrich-unit-test-2", title="Some Paper",
                      pdf_url="https://example.com/paper", web_enrich_attempts=1)
        fake_info = {"pdf_url": None, "page_count": None}

        with patch("app.crawl.extract_paper_info_from_url", return_value=fake_info), \
             patch("app.crawl._enrich_web_paper_pdf_metadata") as mock_pdf_enrich:
            _enrich_one_web_paper(paper)
            mock_pdf_enrich.assert_not_called()

        assert paper.web_enrich_attempts == 2


class TestPass3RetryCap:
    def test_paper_at_attempt_cap_is_excluded(self, app):
        """A web paper that already hit MAX_WEB_ENRICH_ATTEMPTS must not be re-queried,
        even though page_count is still NULL."""
        from app import db
        from app.models import Paper
        from app.crawl import backfill_missing_paper_metadata

        with app.app_context():
            stuck = Paper(
                arxiv_id="web:stuck-paper", title="Stuck Paper", source="web",
                page_count=None, institutions=None, web_enrich_attempts=3,
                web_enrich_last_attempt_at=datetime.utcnow() - timedelta(days=1),
            )
            db.session.add(stuck)
            db.session.commit()
            stuck_id = stuck.id

            try:
                # Pass 1/2 may pick up unrelated incomplete papers left over by other tests in
                # this session-scoped DB — mock the arXiv lock and time.sleep too so that's
                # instant, not real multi-second throttling (this test only exercises Pass 3).
                with patch("app.crawl._ArxivLockCtx"), patch("app.crawl.time.sleep"):
                    with patch("app.crawl.arxiv.Client") as mock_arxiv_client:
                        mock_arxiv_client.return_value.results.return_value = []
                        with patch("app.crawl.get_citation_count", return_value=None):
                            with patch("app.crawl.extract_paper_info_from_url") as mock_extract:
                                backfill_missing_paper_metadata()
                                mock_extract.assert_not_called()

                refreshed = Paper.query.get(stuck_id)
                assert refreshed.web_enrich_attempts == 3, "attempt count must not increase past the cap"
            finally:
                db.session.delete(Paper.query.get(stuck_id))
                db.session.commit()

    def test_paper_under_cap_is_retried_and_attempts_increments(self, app):
        from app import db
        from app.models import Paper
        from app.crawl import backfill_missing_paper_metadata

        with app.app_context():
            retryable = Paper(
                arxiv_id="web:retryable-paper", title="Retryable Paper", source="web",
                pdf_url="https://dl.acm.org/doi/fake", page_count=None, institutions=None,
                web_enrich_attempts=1,
            )
            db.session.add(retryable)
            db.session.commit()
            retryable_id = retryable.id

            try:
                with patch("app.crawl._ArxivLockCtx"), patch("app.crawl.time.sleep"):
                    with patch("app.crawl.arxiv.Client") as mock_arxiv_client:
                        mock_arxiv_client.return_value.results.return_value = []
                        with patch("app.crawl.get_citation_count", return_value=None):
                            with patch(
                                "app.crawl.extract_paper_info_from_url",
                                return_value={"pdf_url": None, "page_count": None},
                            ) as mock_extract:
                                backfill_missing_paper_metadata()
                                mock_extract.assert_called_once()

                refreshed = Paper.query.get(retryable_id)
                assert refreshed.web_enrich_attempts == 2
                assert refreshed.web_enrich_last_attempt_at is not None
            finally:
                db.session.delete(Paper.query.get(retryable_id))
                db.session.commit()


class TestRetryWebEnrichRoute:
    """POST /paper/<id>/retry-web-enrich — the UI 'Retry' button for stuck web papers."""

    class _SyncThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    def _login(self, client):
        with client.session_transaction() as sess:
            sess["logged_in"] = True

    def test_rejects_non_web_paper(self, app, client):
        from app import db
        from app.models import Paper

        with app.app_context():
            paper = Paper(arxiv_id="2501.00001", title="An arXiv Paper", source="arxiv")
            db.session.add(paper)
            db.session.commit()
            paper_id = paper.id

            try:
                self._login(client)
                resp = client.post("/paper/2501.00001/retry-web-enrich")
                assert resp.status_code == 400
                assert resp.get_json()["ok"] is False
            finally:
                db.session.delete(Paper.query.get(paper_id))
                db.session.commit()

    def test_resets_attempts_and_dispatches_retry(self, app, client):
        from app import db
        from app.models import Paper

        with app.app_context():
            paper = Paper(
                arxiv_id="web:retry-route-test", title="Stuck Web Paper", source="web",
                pdf_url="https://dl.acm.org/doi/fake", web_enrich_attempts=3,
                web_enrich_last_attempt_at=datetime.utcnow(),
            )
            db.session.add(paper)
            db.session.commit()
            paper_id = paper.id

            try:
                self._login(client)
                with patch("threading.Thread", self._SyncThread), \
                     patch("app.routes.papers._enrich_one_web_paper") as mock_enrich:
                    resp = client.post("/paper/web:retry-route-test/retry-web-enrich")
                    assert resp.status_code == 200
                    assert resp.get_json()["ok"] is True
                    mock_enrich.assert_called_once()

                refreshed = Paper.query.get(paper_id)
                assert refreshed.web_enrich_attempts == 0, "route must reset the counter before retrying"
                assert refreshed.web_enrich_last_attempt_at is None
            finally:
                db.session.delete(Paper.query.get(paper_id))
                db.session.commit()


class TestRetryButtonRendering:
    """paper_card.html's 'PDF metadata failed / Retry' branch — rendered for real through the
    tag_paper route's HTMX card-fragment response, not just asserted from the template source."""

    def _login(self, client):
        with client.session_transaction() as sess:
            sess["logged_in"] = True

    def _render_card(self, app, client, project, paper):
        from app import db
        from app.models import ProjectPaper

        with app.app_context():
            pp = ProjectPaper(project_id=project.id, paper_id=paper.id)
            db.session.add(pp)
            db.session.commit()
            pp_id, paper_id = pp.id, paper.id

        try:
            self._login(client)
            card_id = paper.arxiv_id.replace(".", "-").replace("/", "-").replace(":", "-")
            resp = client.post(
                f"/paper/{paper.arxiv_id}/tag",
                data={"project_id": project.id, "tag": ""},
                headers={"HX-Request": "true", "HX-Target": f"paper-card-{card_id}-{project.id}"},
            )
            return resp.get_data(as_text=True)
        finally:
            with app.app_context():
                from app.models import ProjectPaper as PP
                pp = PP.query.get(pp_id)
                if pp:
                    db.session.delete(pp)
                    db.session.commit()

    def test_retry_button_shown_at_attempt_cap(self, app, client, project):
        from app import db
        from app.models import Paper

        with app.app_context():
            paper = Paper(arxiv_id="web:template-at-cap", title="Stuck Paper", source="web",
                          web_enrich_attempts=3, page_count=None, authors="[]")
            db.session.add(paper)
            db.session.commit()
            paper_id = paper.id
            paper_obj = paper

        try:
            html = self._render_card(app, client, project, paper_obj)
            assert "PDF metadata failed" in html
            assert "3/3 attempts" in html
            assert "retry-web-enrich" in html
        finally:
            with app.app_context():
                p = Paper.query.get(paper_id)
                if p:
                    db.session.delete(p)
                    db.session.commit()

    def test_retry_button_hidden_under_attempt_cap(self, app, client, project):
        from app import db
        from app.models import Paper

        with app.app_context():
            paper = Paper(arxiv_id="web:template-under-cap", title="Retrying Paper", source="web",
                          web_enrich_attempts=1, page_count=None, authors="[]")
            db.session.add(paper)
            db.session.commit()
            paper_id = paper.id
            paper_obj = paper

        try:
            html = self._render_card(app, client, project, paper_obj)
            assert "PDF metadata failed" not in html
        finally:
            with app.app_context():
                p = Paper.query.get(paper_id)
                if p:
                    db.session.delete(p)
                    db.session.commit()


class TestUserAgent:
    def test_http_client_sets_custom_user_agent(self):
        from app.utils import _http_client, USER_AGENT
        client = _http_client()
        try:
            assert client.headers["user-agent"] == USER_AGENT
        finally:
            client.close()

    def test_scholar_headers_include_user_agent(self, ctx):
        from app.crawl import _scholar_headers
        from app.utils import USER_AGENT
        assert _scholar_headers()["User-Agent"] == USER_AGENT
