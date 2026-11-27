"""New arXiv versions of a paper (2026-11-27).

arxiv_meta keeps paper.arxiv_version fresh (weekly per paper). On top of that:
- The summary records the version it was read from (PaperSummary.summary_version). When arXiv has a newer one,
  the card says so and offers "Update summary" (Regenerate).
- Update summary compares the old version's text with the new one (arXiv keeps every version at
  /abs/<id>v<n>), and the LLM names what changed from the changed sentences and the authors' arXiv comment.
  The old summary is kept beside the new one.
- In Collection a new version is marked like a message: it stays marked for each person until they mark it seen
  (PaperMemberState.version_seen; ProjectPaper.version_seen in projects without per-person mode).
"""
import difflib
import json
import logging
import re
from datetime import datetime

from app import members, prompts
from app.utils import _ArxivLockCtx, _http_client, base_arxiv_id

_log = logging.getLogger(__name__)

DIFF_CHARS = 40_000  # changed sentences given to the LLM; a reformatted paper (HTML vs PDF) changes "everything"
_HISTORY_RE = re.compile(r"\[v(\d+)\]\s+\w{3}, (\d{1,2} \w{3} \d{4} \d{2}:\d{2}:\d{2}) UTC")


# ── What the card shows ────────────────────────────────────────────────────────────────────────────────

def summary_outdated(paper) -> bool:
    """arXiv has a newer version than the one the summary was read from (or that one is unknown and older)."""
    s = paper.summary
    if not paper.is_arxiv or s is None or not paper.arxiv_version or paper.arxiv_version < 2:
        return False
    return s.summary_version is None or s.summary_version < paper.arxiv_version


def _seen(pp, member) -> int | None:
    if pp.project.per_person and member is not None:
        st = members.state(pp, member)
        return st.version_seen if st is not None else None
    return pp.version_seen


def unseen(pp, member) -> bool:
    """A Collection paper got a version after it was collected that this person hasn't marked seen."""
    p = pp.paper
    if pp.trashed_at or pp.manual_tag is None or not p.is_arxiv or not p.arxiv_version or p.arxiv_version < 2:
        return False
    since = pp.collected_at or pp.added_at
    if p.arxiv_updated is None or (since is not None and p.arxiv_updated <= since.date()):
        return False  # already out when the paper was collected
    seen = _seen(pp, member)
    return seen is None or seen < p.arxiv_version


def mark_seen(pp, member) -> None:
    """Caller commits."""
    if pp.project.per_person and member is not None:
        members.state(pp, member, create=True).version_seen = pp.paper.arxiv_version
    else:
        pp.version_seen = pp.paper.arxiv_version


# ── Which version the old summary was read from ──────────────────────────────────────────────────────────

def history(arxiv_id: str) -> list[tuple[int, datetime]]:
    """[(version, submitted at UTC)] from the paper's abstract page."""
    with _ArxivLockCtx():
        with _http_client(timeout=30.0, follow_redirects=True) as http:
            r = http.get(f"https://arxiv.org/abs/{base_arxiv_id(arxiv_id)}")
    r.raise_for_status()
    text = re.sub(r"<[^>]+>", " ", r.text)
    return [(int(v), datetime.strptime(d, "%d %b %Y %H:%M:%S")) for v, d in _HISTORY_RE.findall(text)]


def old_version(paper) -> int:
    """The version the current summary was read from: recorded, else the newest one out when it was written."""
    s = paper.summary
    if s.summary_version:
        return s.summary_version
    before = [v for v, at in history(paper.arxiv_id) if s.generated_at is None or at <= s.generated_at]
    return max(before) if before else 1


# ── What changed ───────────────────────────────────────────────────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text) if len(s.strip()) > 20]


def _key(sentence: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", sentence.lower())


def text_changes(old: str, new: str, limit: int = DIFF_CHARS) -> str:
    """The sentences removed from the old text and added in the new one, in reading order, as REMOVED:/ADDED:
    lines. Compared sentence by sentence with case, spaces and punctuation ignored, so line breaks and a
    different text extraction count less. "" when nothing changed."""
    a, b = _sentences(old), _sentences(new)
    sm = difflib.SequenceMatcher(None, [_key(x) for x in a], [_key(x) for x in b], autojunk=False)
    out, size = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for label, part in (("REMOVED", a[i1:i2]), ("ADDED", b[j1:j2])):
            if part:
                line = f"{label}: {' '.join(part)}"
                out.append(line)
                size += len(line)
        if size >= limit:
            out.append("(more changes cut off)")
            break
    return "\n".join(out)[:limit + 100]


def describe(paper, old_v: int, old_text: str | None, new_text: str | None) -> list[str]:
    """2-4 bullets on what changed from old_v to the latest version. The authors' arXiv comment and journal
    reference go in too: they often say what changed or where it was accepted."""
    from app.llm import _llm_json
    if old_text and new_text:
        diff = text_changes(old_text, new_text) or "(no changes in the text found)"
    else:
        diff = "(the text of one of the versions could not be fetched)"
    prompt = prompts.VERSION_CHANGES.format(
        title=paper.title, old=old_v, new=paper.arxiv_version, comment=paper.arxiv_comment or "(none)",
        journal_ref=paper.journal_ref or "(none)", diff=diff)
    result, _, _, _ = _llm_json([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=4096)
    bullets = result.get("bullets") if isinstance(result, dict) else None
    if not isinstance(bullets, list):
        _log.warning("[versions] unparseable answer for %s: %r", paper.arxiv_id, result)
        return []
    return [str(b).strip() for b in bullets if str(b).strip()]


def snapshot(paper, old_v: int) -> str:
    """The current summary as the old version's (PaperSummary.previous_summary), taken before it is rewritten."""
    s = paper.summary
    return json.dumps({"version": old_v, "summary": s.summary_text, "contributions": s.main_contributions})


def store_changes(paper, old_v: int, bullets: list[str]) -> None:
    """Caller commits."""
    paper.summary.version_changes = json.dumps({"from": old_v, "to": paper.arxiv_version, "bullets": bullets})
