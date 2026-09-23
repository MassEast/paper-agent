"""
Shared fixtures for the test suite.

Sets required env vars before any app imports so llm.py doesn't raise on import.
Uses a file-based SQLite (not :memory:) so threaded run_crawl tests share the
same DB connection.
"""

import os
import tempfile

# Must be set before any app.* import — llm.py raises EnvironmentError otherwise
os.environ.setdefault("LLM_API_KEY", "test-key-not-real")
os.environ.setdefault("LLM_API_BASE", "http://test-llm-endpoint.invalid/v1")
os.environ.setdefault("LLM_MODELS", "test-model-large,test-model-small")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("SITE_PASSWORD", "testpass")

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
    "key_findings": ["Finding 1", "Finding 2"],
    "methodology": "Sparse attention with linear complexity.",
    "limitations": "Only tested on language tasks.",
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
