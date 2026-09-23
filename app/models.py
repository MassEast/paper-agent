import hashlib
import json
import os
import re
from datetime import datetime

from app import db


class Project(db.Model):
    __tablename__ = "projects"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text, nullable=False)
    slug = db.Column(db.Text, unique=True, nullable=False)
    research_interest = db.Column(db.Text)
    notification_emails = db.Column(db.Text)
    saved_keywords = db.Column(db.Text)  # JSON array; persisted per project for nightly crawl
    crawl_hour = db.Column(db.Integer)   # hour (1-4 UTC) when the nightly CronJob crawls this project
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    trashed_at = db.Column(db.DateTime, nullable=True)

    papers = db.relationship("ProjectPaper", back_populates="project", cascade="all, delete-orphan")
    crawl_logs = db.relationship("CrawlLog", back_populates="project", cascade="all, delete-orphan",
                                order_by="CrawlLog.started_at.desc()")

    @property
    def saved_keywords_list(self):
        if self.saved_keywords:
            try:
                return json.loads(self.saved_keywords)
            except (json.JSONDecodeError, TypeError):
                return []
        return []


class Paper(db.Model):
    __tablename__ = "papers"

    id = db.Column(db.Integer, primary_key=True)
    arxiv_id = db.Column(db.Text, unique=True, nullable=False)
    title = db.Column(db.Text, nullable=False)
    authors = db.Column(db.Text)
    institutions = db.Column(db.Text)
    year = db.Column(db.Integer)
    published_date = db.Column(db.Date)
    abstract = db.Column(db.Text)
    pdf_url = db.Column(db.Text)
    citation_count = db.Column(db.Integer, default=0)
    citation_fetched_at = db.Column(db.DateTime)
    citation_manual = db.Column(db.Boolean, default=False)
    page_count = db.Column(db.Integer)
    semantic_scholar_id = db.Column(db.Text)
    source = db.Column(db.Text)
    figure_url = db.Column(db.Text)
    figure_caption = db.Column(db.Text)
    enrich_complete = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    summary = db.relationship("PaperSummary", back_populates="paper", uselist=False)
    project_links = db.relationship("ProjectPaper", back_populates="paper")

    @property
    def is_arxiv(self) -> bool:
        """False for web papers added directly (arxiv_id starts with 'web:')."""
        return bool(self.arxiv_id) and not self.arxiv_id.startswith("web:")

    @property
    def authors_list(self):
        if self.authors:
            try:
                return json.loads(self.authors)
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    @property
    def institutions_list(self):
        if self.institutions:
            try:
                return [i for i in json.loads(self.institutions) if i and i.strip()]
            except (json.JSONDecodeError, TypeError):
                return []
        return []


class PaperSummary(db.Model):
    __tablename__ = "paper_summaries"

    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id"), unique=True, nullable=False)
    summary_text = db.Column(db.Text)
    main_contributions = db.Column(db.Text)
    model_used = db.Column(db.Text)
    introduces_dataset = db.Column(db.Boolean, default=False)
    introduces_architecture = db.Column(db.Boolean, default=False)
    introduces_method = db.Column(db.Boolean, default=False)
    is_survey = db.Column(db.Boolean, default=False)
    is_benchmark = db.Column(db.Boolean, default=False)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    elapsed_ms = db.Column(db.Integer)
    tokens_used = db.Column(db.Integer, default=0)  # total tokens for relevance+summary+contributions

    paper = db.relationship("Paper", back_populates="summary")


class ProjectPaper(db.Model):
    __tablename__ = "project_papers"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=False)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id"), nullable=False)
    manual_tag = db.Column(db.Text)
    paper_tags = db.Column(db.Text, default='[]')
    notes = db.Column(db.Text)
    notes_height = db.Column(db.Integer)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    collected_at = db.Column(db.DateTime, nullable=True)
    trashed_at = db.Column(db.DateTime, nullable=True)

    __table_args__ = (
        db.UniqueConstraint("project_id", "paper_id"),
    )

    project = db.relationship("Project", back_populates="papers")
    paper = db.relationship("Paper", back_populates="project_links")

    @property
    def tags_list(self) -> list[str]:
        if self.paper_tags:
            try:
                return json.loads(self.paper_tags)
            except (json.JSONDecodeError, TypeError):
                return []
        return []


class CrawlLog(db.Model):
    """One row per crawl run (manual or nightly cron). status is "running"/"success"/
    "cancelled"/"error"; the progress counters (papers_found/checked/added) are updated
    live during a run and polled by the crawl-status UI."""
    __tablename__ = "crawl_logs"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=False)
    triggered_by = db.Column(db.Text)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    finished_at = db.Column(db.DateTime)
    status = db.Column(db.Text)
    date_from = db.Column(db.Date)
    date_to = db.Column(db.Date)
    papers_found = db.Column(db.Integer, default=0)
    papers_added = db.Column(db.Integer, default=0)
    papers_checked = db.Column(db.Integer, default=0)
    paper_limit = db.Column(db.Integer, default=0)
    keywords_used = db.Column(db.Text)
    error_message = db.Column(db.Text)
    total_tokens = db.Column(db.Integer, default=0)
    sources_used = db.Column(db.Text)  # e.g. "arxiv,ss", "arxiv", "ss"

    project = db.relationship("Project", back_populates="crawl_logs")

    @property
    def keywords_list(self):
        if self.keywords_used:
            try:
                return json.loads(self.keywords_used)
            except (json.JSONDecodeError, TypeError):
                return []
        return []
    
    @property
    def duration_minutes(self):
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds() / 60
        return None
    
    @property
    def estimated_cost(self):
        """Cost comparison figure — not what we actually pay (self-hosted inference has no
        real per-token cost), but what this many tokens would have cost on a metered API.
        Model/rates configurable via LLM_COST_COMPARISON_MODEL / _INPUT_PER_1M / _OUTPUT_PER_1M
        (default: Claude Sonnet 4.6 pricing as of May 2026, $3/$15 per million tokens)."""
        if not self.total_tokens:
            return 0
        input_cost_per_1m = float(os.environ.get("LLM_COST_COMPARISON_INPUT_PER_1M", "3.00"))
        output_cost_per_1m = float(os.environ.get("LLM_COST_COMPARISON_OUTPUT_PER_1M", "15.00"))
        # Typical ratio: ~70% input tokens, 30% output tokens
        ratio = 0.7
        input_tokens = int(self.total_tokens * ratio)
        output_tokens = int(self.total_tokens * (1 - ratio))
        return (input_tokens / 1_000_000 * input_cost_per_1m) + (output_tokens / 1_000_000 * output_cost_per_1m)


class ScreenedPaper(db.Model):
    """Records every paper screened for a project + the context hash at screening time.
    Lets the crawl skip re-screening papers whose relevance was already determined with
    the same research interest and My Collection contents."""
    __tablename__ = "screened_papers"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=False)
    arxiv_id = db.Column(db.Text, nullable=False)   # base ID without version suffix
    screening_hash = db.Column(db.Text, nullable=False)
    is_relevant = db.Column(db.Boolean, nullable=False)
    screened_at = db.Column(db.DateTime, default=datetime.utcnow)

    project = db.relationship("Project", backref=db.backref("screened_papers", cascade="all, delete-orphan"))

    __table_args__ = (
        db.Index("idx_screened_lookup", "project_id", "arxiv_id", "screening_hash"),
    )


def compute_screening_hash(research_interest: str, collection_papers: list) -> str:
    """Stable 16-char hex hash of the project's screening context.

    Hash is over research_interest + sorted titles of all non-trashed My Collection papers.
    Changes when the user updates their research interest or adds/removes collection papers.
    """
    ri = (research_interest or "").strip()
    collection_titles = sorted(
        pp.paper.title for pp in collection_papers
        if pp.paper and not pp.trashed_at and pp.manual_tag
    )
    content = f"{ri}|||{'|||'.join(collection_titles)}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def slugify(text):
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")