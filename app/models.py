import hashlib
import json
import os
import re
from datetime import datetime

from app import db

# The research interest goes into every screening prompt, so its length is paid once per paper screened.
# A pasted 48k-char design doc made one project cost 5-7M tokens a night (2026-10). Generate/Improve aim
# for at most 300 words (~2,000 chars), so their output fits with room to spare.
RESEARCH_INTEREST_MAX_CHARS = 4000


class Project(db.Model):
    __tablename__ = "projects"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text, nullable=False)
    slug = db.Column(db.Text, unique=True, nullable=False)
    research_interest = db.Column(db.Text)
    notification_emails = db.Column(db.Text)
    saved_keywords = db.Column(db.Text)  # JSON array; persisted per project for nightly crawl
    crawl_hour = db.Column(db.Integer)   # hour (1-4 UTC) when the nightly CronJob crawls this project
    use_biorxiv = db.Column(db.Boolean, default=False)  # nightly crawl also searches bioRxiv; Find Papers starts from it
    # Several people in one project (app/members.py): each has their own New Papers, tags, votes and chat;
    # My Collection and Trash stay shared. Off: one shared view, as before.
    per_person = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    trashed_at = db.Column(db.DateTime, nullable=True)

    papers = db.relationship("ProjectPaper", back_populates="project", cascade="all, delete-orphan")
    crawl_logs = db.relationship("CrawlLog", back_populates="project", cascade="all, delete-orphan",
                                order_by="CrawlLog.started_at.desc()")
    people = db.relationship("ProjectMember", cascade="all, delete-orphan", order_by="ProjectMember.id")

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
    citation_attempted_at = db.Column(db.DateTime)  # last backfill fetch attempt, success or not (cooldown)
    citation_manual = db.Column(db.Boolean, default=False)
    page_count = db.Column(db.Integer)
    semantic_scholar_id = db.Column(db.Text)  # a non-arXiv paper's Semantic Scholar id, found by title (scholar_seed)
    page_url = db.Column(db.Text)  # the link a paper was added from by hand (its page; pdf_url may be the PDF)
    doi = db.Column(db.Text)  # when the source gives one (bioRxiv, Semantic Scholar, the arXiv API, a pasted page)
    scholar_lookup_at = db.Column(db.DateTime)  # when resolve_scholar_ids looked a non-arXiv paper up by title
    source = db.Column(db.Text)
    figure_url = db.Column(db.Text)
    figure_caption = db.Column(db.Text)
    figure_pdf_tried_at = db.Column(db.DateTime)  # the nightly backfill tried to cut Figure 1 from the PDF
    enrich_complete = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Web-paper (non-arXiv) PDF metadata backfill (Pass 3 of backfill_missing_paper_metadata):
    # capped-retry bookkeeping so a permanently-failing publisher fetch (403/bot-block, dead
    # link, ...) doesn't get re-attempted forever by the nightly cronjob.
    web_enrich_attempts = db.Column(db.Integer, default=0)
    web_enrich_last_attempt_at = db.Column(db.DateTime)
    # Failed tries of the nightly full-text re-read (refresh_from_full_text) to get this paper's full text
    full_text_attempts = db.Column(db.Integer, default=0)
    # From the paper itself (app/resources.py): its own code repository and project page, the corresponding
    # authors [{"name", "email"}] (name as in authors), the venue it says it appeared at
    code_url = db.Column(db.Text)
    project_url = db.Column(db.Text)
    corresponding_authors = db.Column(db.Text)
    venue = db.Column(db.Text)
    resources_checked_at = db.Column(db.DateTime)  # when that was read (None: not yet, or again after a new version)
    # From the arXiv API (app/arxiv_meta.py): latest version, when it came out, the authors' comment, journal ref
    arxiv_version = db.Column(db.Integer)
    arxiv_updated = db.Column(db.Date)
    arxiv_comment = db.Column(db.Text)
    journal_ref = db.Column(db.Text)
    arxiv_meta_at = db.Column(db.DateTime)
    # The paper's references from Semantic Scholar (PaperReference), for the card's "also cites"
    references_fetched_at = db.Column(db.DateTime)
    references_attempted_at = db.Column(db.DateTime)

    summary = db.relationship("PaperSummary", back_populates="paper", uselist=False)
    project_links = db.relationship("ProjectPaper", back_populates="paper")

    @property
    def landing_url(self) -> str:
        """The paper's page, for the card title, "Source" and email links: the arXiv abstract page; the bioRxiv
        page; doi.org (the publisher's article page); the link it was added from; else its PDF. Not the PDF
        first: Frontiers and PLOS, for example, send it as a download (Content-Disposition: attachment)."""
        from app.utils import doi_from_url
        if self.is_arxiv:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        doi = self.doi or doi_from_url(self.page_url) or doi_from_url(self.pdf_url)
        if doi and doi.startswith(("10.1101/", "10.64898/")):
            return f"https://www.biorxiv.org/content/{doi}"
        if doi:
            return f"https://doi.org/{doi}"
        return self.page_url or self.pdf_url or ""

    @property
    def pdf_link(self) -> str | None:
        """The PDF to offer next to the page link, if we have one that is not just the page link."""
        if self.is_arxiv:
            return f"https://arxiv.org/pdf/{self.arxiv_id}"
        url = self.pdf_url or ""
        if not url or url == self.landing_url or url.startswith("https://doi.org/") or url == self.page_url and not url.lower().split("?")[0].endswith(".pdf"):
            return None
        return url

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
    def corresponding_list(self) -> list[dict]:
        return json.loads(self.corresponding_authors) if self.corresponding_authors else []

    @property
    def corresponding_emails(self) -> dict[str, str]:
        """Author name -> email, for the mailto links on the card."""
        return {c["name"]: c["email"] for c in self.corresponding_list}

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
    tokens_used = db.Column(db.Integer, default=0)  # summary + contributions of the latest run
    # What the summary and main contributions were made from: "full_text" | "web_page" (a web paper's
    # landing-page text) | "abstract" (no full text could be fetched). None: made before this was recorded.
    content_source = db.Column(db.Text)

    paper = db.relationship("Paper", back_populates="summary")


class PaperReference(db.Model):
    """One paper a paper cites, as a key that can be matched against papers in the DB: "arxiv:<id>",
    "doi:<doi, lower case>" or "s2:<Semantic Scholar paperId>" (one row per key). From Semantic Scholar,
    app/connections.py."""
    __tablename__ = "paper_references"

    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id"), nullable=False)
    ref_key = db.Column(db.Text, nullable=False)


class PaperText(db.Model):
    """A paper's full text, fetched once when someone first chats about it (app/chat.py). Its own table
    so the project page, which loads every Paper, never pulls hundreds of KB of text per card."""
    __tablename__ = "paper_texts"

    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id"), unique=True, nullable=False)
    text = db.Column(db.Text, nullable=False)
    source = db.Column(db.Text, nullable=False)  # "html" | "pdf"
    fetched_at = db.Column(db.DateTime, default=datetime.utcnow)


class ChatMessage(db.Model):
    """One message of the "ask about this paper" chat on a card, per paper per project (like notes).
    An assistant row is created empty with status "streaming" when its question is sent to the model,
    and filled in when the answer is done."""
    __tablename__ = "chat_messages"
    # Ids are never reused after a chat is cleared: they key the in-memory answer state and the
    # /chat/message/<id> polls of pages still open
    __table_args__ = {"sqlite_autoincrement": True}

    id = db.Column(db.Integer, primary_key=True)
    project_paper_id = db.Column(db.Integer, db.ForeignKey("project_papers.id"), nullable=False, index=True)
    role = db.Column(db.Text, nullable=False)  # "user" | "assistant"
    content = db.Column(db.Text, nullable=False, default="")
    # user: "queued" (waiting for an earlier answer) | "done" (sent to the model)
    # assistant: "streaming" | "done" | "stopped" (by the user; the partial answer is kept) | "error"
    status = db.Column(db.Text, nullable=False, default="done")
    reply_to_id = db.Column(db.Integer)  # assistant rows: the question they answer
    context_source = db.Column(db.Text)  # what the answer was based on: "html" | "pdf" | "abstract"
    model = db.Column(db.Text)
    # Assistant rows: the whole conversation's size as the model counted it (prompt + this answer),
    # shown as context usage on the chat
    context_tokens = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    member_id = db.Column(db.Integer)  # per_person projects: whose chat this is (None: shared, from before)

    project_paper = db.relationship("ProjectPaper", back_populates="chat_messages")


class ProjectMember(db.Model):
    """A person working in a project, one per notification address (members.sync_from_emails). Not a login: the site has one shared password, and the browser
    remembers who you are per project (cookie)."""
    __tablename__ = "project_members"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"), nullable=False, index=True)
    name = db.Column(db.Text, nullable=False)
    email = db.Column(db.Text)
    color = db.Column(db.Text, nullable=False)  # CSS color, from members.COLORS
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class PaperMemberState(db.Model):
    """One person's own view of one paper in a per_person project."""
    __tablename__ = "paper_member_states"
    __table_args__ = (db.UniqueConstraint("project_paper_id", "member_id"),)

    id = db.Column(db.Integer, primary_key=True)
    project_paper_id = db.Column(db.Integer, db.ForeignKey("project_papers.id"), nullable=False, index=True)
    member_id = db.Column(db.Integer, db.ForeignKey("project_members.id"), nullable=False)
    tags = db.Column(db.Text)  # JSON list; None: not set yet, the paper's shared tags from before apply
    vote = db.Column(db.Integer)  # 1-5
    # When this person last acted on the paper; a move by someone else after this puts it back in their New Papers
    seen_at = db.Column(db.DateTime)

    project_paper = db.relationship("ProjectPaper", back_populates="member_states")
    member = db.relationship("ProjectMember")


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
    # Per project (the summary is shared across projects): one or two sentences from the
    # screening LLM on why this paper matters for this project's research interest.
    relevance_reason = db.Column(db.Text, nullable=True)
    # What the deep relevance check read: "full_text" or "abstract" (no full text could be fetched; the
    # nightly refresh_from_full_text tries again). "unknown": screened before this was recorded and marked
    # for that re-read too. None: never screened (added by hand, copied) or screened before this was recorded
    screen_source = db.Column(db.Text)
    trashed_reason = db.Column(db.Text)  # why an automatic step moved it to Trash; None when trashed by hand
    # How the paper came into this project: "crawl" (nightly or Find Papers, via New Papers), "manual" (added
    # by URL), "copied" / "moved" (Copy / Move from origin_project_id). None: from before this was recorded
    # and not safely inferable (see the backfill in create_app).
    origin = db.Column(db.Text)
    origin_project_id = db.Column(db.Integer)
    # For origin "crawl": JSON list of how the crawl found it, one entry per way: {"via": "keyword", "name": k}
    # (no name when no keyword appears literally in title/abstract), {"via": "recommended"} (Semantic Scholar
    # recommendations), {"via": "biorxiv", "name": k} (bioRxiv keyword search), {"via": "cites", "arxiv_id",
    # "title"} (cites that My Collection paper). None: from
    # before this was recorded
    found_via = db.Column(db.Text)
    # When a crawl last asked Semantic Scholar which papers cite this My Collection paper (and got an answer):
    # the citations source checks the CITATION_MAX_SEEDS checked longest ago first (citation_seeds)
    citations_checked_at = db.Column(db.DateTime, nullable=True)

    __table_args__ = (
        db.UniqueConstraint("project_id", "paper_id"),
    )

    # per_person projects: who last moved the paper between sections (collected, trashed, restored) and when;
    # it then shows up in everyone else's New Papers until they acknowledge it (members.section_for)
    moved_by_id = db.Column(db.Integer, db.ForeignKey("project_members.id"), nullable=True)
    moved_at = db.Column(db.DateTime, nullable=True)
    # per_person projects: JSON list of [member_id or null, text] runs (who wrote what); joined, they are the
    # note text, which `notes` keeps too
    notes_runs = db.Column(db.Text)
    # The shared tags as members.set_per_person left them when per-person mode was turned off; turning it on
    # again compares: tags changed meanwhile apply to everyone, unchanged ones give everyone their own tags back
    tags_at_off = db.Column(db.Text)

    project = db.relationship("Project", back_populates="papers")
    paper = db.relationship("Paper", back_populates="project_links")
    moved_by = db.relationship("ProjectMember", foreign_keys=[moved_by_id])
    member_states = db.relationship("PaperMemberState", back_populates="project_paper", cascade="all, delete-orphan")
    chat_messages = db.relationship("ChatMessage", back_populates="project_paper", cascade="all, delete-orphan",
                                    order_by="ChatMessage.id")

    @property
    def origin_project(self):
        return db.session.get(Project, self.origin_project_id) if self.origin_project_id else None

    @property
    def found_via_list(self) -> list[dict]:
        return json.loads(self.found_via) if self.found_via else []

    def found_via_text(self, full: bool = False) -> str:
        """How the crawl found the paper, for the card: keywords by name, the recommendation, cited
        My Collection papers by title. Short form for the card line (first cited paper + "(+N)"), full
        form (every cited paper) for its tooltip. "" when not recorded."""
        via = self.found_via_list
        names = [v["name"] for v in via if v["via"] == "keyword" and v.get("name")]
        bio_names = [v["name"] for v in via if v["via"] == "biorxiv" and v.get("name")]
        cited = [v["title"] for v in via if v["via"] == "cites"]
        parts = []
        if names:
            parts.append(("keyword " if len(names) == 1 else "keywords ") + ", ".join(f'"{n}"' for n in names))
        elif any(v["via"] == "keyword" for v in via):
            parts.append("arXiv keyword search")
        if bio_names:
            parts.append(("bioRxiv keyword " if len(bio_names) == 1 else "bioRxiv keywords ")
                         + ", ".join(f'"{n}"' for n in bio_names))
        elif any(v["via"] == "biorxiv" for v in via):
            parts.append("bioRxiv keyword search")
        if any(v["via"] == "recommended" for v in via):
            parts.append("Semantic Scholar recommendation")
        if cited:
            if full:
                parts.append("cites " + "; ".join(f'"{t}"' for t in cited))
            else:
                first = cited[0] if len(cited[0]) <= 50 else cited[0][:48].rstrip() + "…"
                parts.append(f'cites "{first}"' + (f" (+{len(cited) - 1})" if len(cited) > 1 else ""))
        return " and ".join(parts)

    def found_via_tip(self) -> str:
        """The card tooltip's longer version of found_via_text: one sentence per way the paper was found."""
        via = self.found_via_list
        quote = lambda xs: ", ".join(f'"{x}"' for x in xs[:-1]) + (" and " if len(xs) > 1 else "") + f'"{xs[-1]}"'
        names = [v["name"] for v in via if v["via"] == "keyword" and v.get("name")]
        bio_names = [v["name"] for v in via if v["via"] == "biorxiv" and v.get("name")]
        cited = [v["title"] for v in via if v["via"] == "cites"]
        out = []
        if names:
            out.append(f"Its title or abstract contains your keyword{'s' if len(names) > 1 else ''} {quote(names)}"
                       " (arXiv keyword search).")
        elif any(v["via"] == "keyword" for v in via):
            out.append("The arXiv keyword search found it.")
        if bio_names:
            out.append(f"Its title or abstract contains your keyword{'s' if len(bio_names) > 1 else ''} "
                       f"{quote(bio_names)} (bioRxiv keyword search).")
        if any(v["via"] == "recommended" for v in via):
            out.append("Semantic Scholar recommends it as similar to papers in My Collection.")
        if len(cited) == 1:
            out.append(f'It cites "{cited[0]}" from My Collection.')
        elif cited:
            out.append(f"It cites {len(cited)} papers from My Collection: " + "; ".join(f'"{t}"' for t in cited) + ".")
        return " ".join(out)

    @property
    def tags_list(self) -> list[str]:
        if self.paper_tags:
            try:
                return json.loads(self.paper_tags)
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    @property
    def any_tags(self) -> set[str]:
        """Every tag anyone set (per_person projects keep tags per person); the crawl's reference papers use it."""
        tags = set(self.tags_list)
        for st in self.member_states:
            if st.tags is not None:
                tags.update(json.loads(st.tags))
        return tags


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
        return estimated_cost_eur(self.total_tokens)


def estimated_cost(tokens) -> float:
    """Cost comparison figure — not what we actually pay (self-hosted inference has no
    real per-token cost), but what this many tokens would have cost on a metered API.
    Model/rates configurable via LLM_COST_COMPARISON_MODEL / _INPUT_PER_1M / _OUTPUT_PER_1M
    (default: Claude Sonnet 4.6 pricing as of May 2026, $3/$15 per million tokens)."""
    if not tokens:
        return 0
    input_cost_per_1m = float(os.environ.get("LLM_COST_COMPARISON_INPUT_PER_1M", "3.00"))
    output_cost_per_1m = float(os.environ.get("LLM_COST_COMPARISON_OUTPUT_PER_1M", "15.00"))
    # Typical ratio: ~70% input tokens, 30% output tokens
    ratio = 0.7
    input_tokens = int(tokens * ratio)
    output_tokens = int(tokens * (1 - ratio))
    return (input_tokens / 1_000_000 * input_cost_per_1m) + (output_tokens / 1_000_000 * output_cost_per_1m)


def estimated_cost_eur(tokens) -> float:
    """estimated_cost in euros (the rates are in dollars); every cost the UI and the emails show.
    LLM_COST_COMPARISON_EUR_PER_USD, default 0.86 (the rate in mid-2026, close enough for an estimate)."""
    return estimated_cost(tokens) * float(os.environ.get("LLM_COST_COMPARISON_EUR_PER_USD", "0.86"))


class LlmUsage(db.Model):
    """Tokens the model used for one action (a crawl, a chat answer, a Regenerate, ...), written by
    app.usage.counted when the action ends. project_id is NULL for work on shared papers that belongs
    to no project (the nightly metadata backfill). Crawls before this table existed were backfilled
    from crawl_logs.total_tokens (calls NULL)."""
    __tablename__ = "llm_usage"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"))
    kind = db.Column(db.Text, nullable=False)
    tokens = db.Column(db.Integer, nullable=False, default=0)
    calls = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


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