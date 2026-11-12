"""
Shared fixtures for the test suite.

Sets required env vars before any app imports so llm.py doesn't raise on import.
Uses a file-based SQLite (not :memory:) so threaded run_crawl tests share the
same DB connection.
"""

import os
import tempfile
import threading

# Must be set before any app.* import — llm.py raises EnvironmentError otherwise
os.environ.setdefault("LLM_API_KEY", "test-key-not-real")
os.environ.setdefault("LLM_API_BASE", "http://test-llm-endpoint.invalid/v1")
os.environ.setdefault("LLM_MODELS", "test-model-large,test-model-small")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("SITE_PASSWORD", "testpass")
os.environ["FIGURES_DIR"] = tempfile.mkdtemp(prefix="figures-")  # Figure 1 PNGs cut from PDFs (app/figures.py)

import pytest

from app import create_app, db as _db
from app.models import Project


@pytest.fixture(scope="session")
def db_file():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["DATABASE_URL"] = f"sqlite:///{path}"
    yield path
    os.unlink(path)


@pytest.fixture(scope="session")
def app(db_file):
    _app = create_app()
    _app.config["TESTING"] = True
    yield _app


@pytest.fixture
def ctx(app):
    with app.app_context():
        yield


@pytest.fixture
def client(app):
    with app.app_context():
        with app.test_client() as c:
            yield c


@pytest.fixture
def project(app):
    """A minimal project with a research interest."""
    with app.app_context():
        p = Project(
            name="Test LLM Agents",
            slug="test-llm-agents",
            research_interest="efficient transformer attention mechanisms for long-context LLMs",
        )
        _db.session.add(p)
        _db.session.commit()
        yield p
        _db.session.delete(p)
        _db.session.commit()


# ── Browser tests (slow): live server + one logged-in browser per module ────────

@pytest.fixture(scope="module")
def live_server(app):
    from werkzeug.serving import make_server

    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(scope="module")
def browser_session(live_server):
    """One browser + one login per module: /login is rate-limited to 5 per minute, which a run of more
    than five browser modules would hit, so the limiter's counts are cleared first."""
    sync_api = pytest.importorskip("playwright.sync_api")
    from app import limiter
    limiter.reset()
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        pg = context.new_page()
        pg.goto(f"{live_server}/login")
        pg.fill("input[name=password]", "testpass")
        pg.click("button[type=submit]")
        state = context.storage_state()
        context.close()
        yield browser, state
        browser.close()


# ── LLM response factories ────────────────────────────────────────────────────

def make_llm_json_response(payload: dict, model="test-model", elapsed=42, tokens=100):
    """Returns a mock _llm_json return value."""
    usage = {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": tokens}
    return payload, model, elapsed, usage


FAKE_KEYWORDS = ["efficient attention mechanisms", "long-context transformers", "sparse attention"]

FAKE_RELEVANCE_YES = {"relevant": True, "reason": "Directly addresses attention efficiency."}
FAKE_RELEVANCE_NO = {"relevant": False, "reason": "Unrelated to attention or transformers."}

FAKE_SUMMARY = {
    "summary": "This paper proposes XYZ attention.",
    "introduces_dataset": False,
    "introduces_architecture": True,
    "introduces_method": True,
    "is_survey": False,
    "is_benchmark": False,
    "relevance_score": 8,
}

FAKE_CONTRIBUTIONS = {
    "main_contributions": ["Reduces attention complexity to O(n)", "New sparse pattern"],
    "contribution_summary": "Proposes efficient sparse attention.",
}

FAKE_PAPER = {
    "arxiv_id": "2501.00001",
    "title": "SuperSparse Attention for Long Sequences",
    "authors": ["Alice Smith", "Bob Jones"],
    "abstract": "We propose SuperSparse attention, reducing complexity to O(n log n).",
    "pdf_url": "https://arxiv.org/pdf/2501.00001.pdf",
    "published_date": __import__("datetime").date(2025, 1, 1),
    "year": 2025,
    "categories": ["cs.LG"],
}


@pytest.fixture(autouse=True)
def _no_arxiv_pdf_download(monkeypatch):
    """No test downloads a PDF for its Figure 1 (get_paper_figure, nightly backfill Pass 5); tests of it patch in a PDF."""
    monkeypatch.setattr("app.figures._download_pdf", lambda arxiv_id: None)


@pytest.fixture(autouse=True)
def _no_card_data_passes(monkeypatch):
    """The nightly backfill's passes for the card (arXiv version/comment, links/emails/venue, references) would ask
    arXiv, Semantic Scholar and the model: off in every test; their own tests call the real functions
    (REAL_CARD_PASSES)."""
    monkeypatch.setattr("app.arxiv_meta.refresh", lambda *a, **k: 0)
    monkeypatch.setattr("app.crawl.backfill_resources", lambda *a, **k: 0)
    monkeypatch.setattr("app.connections.fetch_references", lambda *a, **k: 0)
    # a paper added by hand gets its why-relevant line from the model; tests/test_enrich_status.py uses the real one
    monkeypatch.setattr("app.crawl._add_relevance_reason", lambda *a, **k: None)


import app.arxiv_meta as _arxiv_meta_mod  # noqa: E402
import app.connections as _connections_mod  # noqa: E402
import app.crawl as _crawl_mod  # noqa: E402

REAL_CARD_PASSES = {"arxiv_meta.refresh": _arxiv_meta_mod.refresh, "crawl.backfill_resources": _crawl_mod.backfill_resources,
                    "connections.fetch_references": _connections_mod.fetch_references,
                    "crawl._add_relevance_reason": _crawl_mod._add_relevance_reason}
