"""Token usage per project, split by what it was for. Every action that calls the model runs inside
`with counted(project_id, kind):`; app.llm adds each call's tokens to the current meter, and one
LlmUsage row is written when the action ends. Project totals (footer, project list, the usage panel
under the crawl calendar) are sums over these rows."""
import functools
import logging

from sqlalchemy import func

from app import db, llm
from app.models import LlmUsage

_log = logging.getLogger(__name__)

# One action (a crawl pass, a chat answer, ...) over this many tokens gets a warning in the log. A crawl
# takes 20k-300k on a usual night, the busiest since 2026-10-16 1.3M; one with a pasted 48k-char research
# interest took 5-7M.
TOKEN_ALARM = 2_000_000

# kind -> label shown in the usage panel, in display order
KINDS = {
    "crawl": "Crawls",
    "chat": "Paper chat",
    "add": "Adding papers",
    "regenerate": "Regenerate",
    "rescreen": "Re-screen",
    "refresh": "Nightly full-text re-read",
    "relevance_reason": "Why-relevant lines (Copy / Move)",
    "keywords": "Keyword generation",
    "research_interest": "Research interest generation",
    "metadata": "Metadata backfill",
}


class counted:
    """`with counted(project_id, "chat"):` writes one LlmUsage row with the tokens used inside, also
    when the block raises (the tokens were spent all the same)."""

    def __init__(self, project_id, kind: str):
        assert kind in KINDS, kind
        self.project_id, self.kind = project_id, kind

    def __enter__(self):
        self._metered = llm.metered()
        self.meter = self._metered.__enter__()
        return self.meter

    def __exit__(self, exc_type, *exc):
        self._metered.__exit__(exc_type, *exc)
        if not self.meter.calls:
            return
        if exc_type is not None:
            db.session.rollback()  # the session may hold the failed work; only the usage row goes in
        db.session.add(LlmUsage(project_id=self.project_id, kind=self.kind, tokens=self.meter.tokens,
                                calls=self.meter.calls))
        db.session.commit()
        _log.info("[usage] project %s %s: %d tokens in %d calls", self.project_id, self.kind,
                  self.meter.tokens, self.meter.calls)
        if self.meter.tokens > TOKEN_ALARM:
            _log.warning("[usage] ALARM project %s %s used %d tokens in one go (over %d): check its research "
                         "interest and keywords", self.project_id, self.kind, self.meter.tokens, TOKEN_ALARM)


def current_tokens() -> int:
    """Tokens counted so far by the action running in this thread (0 outside any)."""
    m = llm._meter.get()
    return m.tokens if m else 0


def project_usage(project_id) -> dict:
    """{kind: tokens} for one project, kinds in KINDS order, only those with any usage."""
    rows = dict(db.session.query(LlmUsage.kind, func.sum(LlmUsage.tokens))
                .filter(LlmUsage.project_id == project_id).group_by(LlmUsage.kind).all())
    return {k: int(rows[k]) for k in KINDS if rows.get(k)}


def usage_by_project() -> dict:
    """{project_id: total tokens} for every project with any usage (the project list)."""
    return {pid: int(t) for pid, t in db.session.query(LlmUsage.project_id, func.sum(LlmUsage.tokens))
            .filter(LlmUsage.project_id.isnot(None)).group_by(LlmUsage.project_id).all()}


def counts(kind: str, per_project: bool = True):
    """Decorator form of counted for functions whose first argument is the project id (or, with
    per_project=False, that work on shared papers and count for no project)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            project_id = (args[0] if args else kwargs["project_id"]) if per_project else None
            with counted(project_id, kind):
                return fn(*args, **kwargs)
        return wrapper
    return deco
