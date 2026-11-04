"""Several people in one project (Project.per_person).

The site has one shared password, so this is not a login: each browser remembers per project who it is
(cookie rw_member_<project id>), set by the "Who are you?" dialog or by the `as=<member id>` in the links of
that person's own crawl email. The people are the project's notification addresses; nobody adds themselves,
and colors are given automatically.

What is shared and what is per person:
- My Collection and Trash are shared. Who last moved a paper between sections, and when, is kept on the
  ProjectPaper (moved_by_id, moved_at).
- New Papers is per person: a paper is in your New Papers while it is in the shared New Papers, or when
  someone else moved it (collected, trashed, restored) since you last acted on it (PaperMemberState.seen_at).
  So when Justus collects a paper, Felix still finds it in his New Papers, marked "added by Justus", until he
  keeps it, tags it, trashes it or restores it. Each paper is in exactly one of your sections.
- Tags, votes and the paper chat are per person; everyone sees the others' tags and votes.
- Notes stay one shared text, but its runs remember who wrote them and on which day (ProjectPaper.notes_runs), so each
  person's words show in their color. Old notes (ProjectPaper.notes) stay as the shared note from before.
"""

import json
import re
from datetime import datetime

from flask import g, has_request_context, request

from app import db
from app.models import PaperMemberState, ProjectMember  # noqa: F401 (ProjectMember used by routes)

# Person colors: distinct from each other and from the section colors, readable on light and dark paper
COLORS = ["#2563eb", "#db2777", "#059669", "#d97706", "#0891b2", "#9333ea", "#65a30d", "#e11d48"]
COOKIE_MAX_AGE = 2 * 365 * 24 * 3600
TAGS = ("important", "to_read", "to_discuss")


def cookie_name(project) -> str:
    return f"rw_member_{project.id}"


def members(project) -> list:
    return ProjectMember.query.filter_by(project_id=project.id).order_by(ProjectMember.id).all()


def current(project):
    """The person this browser is in a per_person project, or None (project not per person, or not chosen yet)."""
    if project is None or not project.per_person or not has_request_context():
        return None
    raw = request.cookies.get(cookie_name(project), "")
    cache = g.setdefault("_members_current", {})  # keyed by the cookie too: g can outlive a request
    if (project.id, raw) not in cache:
        m = db.session.get(ProjectMember, int(raw)) if raw.isdigit() else None
        cache[(project.id, raw)] = m if m is not None and m.project_id == project.id else None
    return cache[(project.id, raw)]


def name_from_email(email: str) -> str:
    """'stephan.mueller@…' -> 'Stephan', 'FelixMaria.Doe@…' -> 'Felix'. Editable afterwards."""
    first = re.split(r"[._+\-]", email.split("@")[0])[0]
    first = re.split(r"(?<=[a-z])(?=[A-Z])", first)[0]
    return first[:1].upper() + first[1:].lower() if first else email


def next_color(project) -> str:
    used = [m.color for m in members(project)]
    free = [c for c in COLORS if c not in used]
    return free[0] if free else COLORS[len(used) % len(COLORS)]


def add(project, name: str, email: str | None = None) -> ProjectMember:
    m = ProjectMember(project_id=project.id, name=name.strip(), email=email, color=next_color(project))
    db.session.add(m)
    db.session.flush()
    return m


def sync_from_emails(project) -> None:
    """Every notification address gets a person (named from the address). Removing an address keeps the
    person, with their tags and votes. Caller commits."""
    have = {(m.email or "").lower() for m in members(project)}
    for email in [e.strip() for e in (project.notification_emails or "").split(",") if e.strip()]:
        if email.lower() not in have:
            add(project, name_from_email(email), email)
            have.add(email.lower())


def by_email(project, email: str):
    for m in members(project):
        if (m.email or "").lower() == email.strip().lower():
            return m
    return None


# ── One person's view of one paper ─────────────────────────────────────────────────────────────────────

def state(pp, member, create: bool = False):
    if member is None:
        return None
    for st in pp.member_states:
        if st.member_id == member.id:
            return st
    if not create:
        return None
    st = PaperMemberState(project_paper_id=pp.id, member_id=member.id)
    db.session.add(st)
    pp.member_states.append(st)
    return st


def shared_section(pp) -> str:
    return "trash" if pp.trashed_at else "new" if pp.manual_tag is None else "col"


def unseen_move(pp, member, st=None) -> bool:
    """Someone else moved the paper between sections after this person last acted on it."""
    if member is None or not pp.moved_by_id or pp.moved_by_id == member.id or not pp.moved_at:
        return False
    st = st if st is not None else state(pp, member)
    return st is None or st.seen_at is None or st.seen_at < pp.moved_at


def section_for(pp, member, st=None) -> str:
    """Which of this person's sections the paper is in (see module doc)."""
    shared = shared_section(pp)
    if member is None or shared == "new":
        return shared
    return "new" if unseen_move(pp, member, st) else shared


def tags_for(pp, member, st=None) -> list:
    """This person's tags; before they set any, the paper's shared tags from before per-person mode."""
    if member is None:
        return pp.tags_list
    st = st if st is not None else state(pp, member)
    return json.loads(st.tags) if st is not None and st.tags is not None else pp.tags_list


def set_tags(pp, member, tags: list) -> None:
    state(pp, member, create=True).tags = json.dumps([t for t in TAGS if t in tags])


def seen(pp, member) -> None:
    state(pp, member, create=True).seen_at = datetime.utcnow()


def moved(pp, member) -> None:
    """Record a move between sections by this person (None: not per person, nothing to record)."""
    if member is None:
        return
    pp.moved_by_id = member.id
    pp.moved_at = datetime.utcnow()
    seen(pp, member)


def others(pp, member) -> list:
    """[(member, tags, vote)] for everyone else who tagged or voted on this paper."""
    out = []
    for st in pp.member_states:
        if member is not None and st.member_id == member.id:
            continue
        tags = json.loads(st.tags) if st.tags is not None else []
        if tags or st.vote:
            out.append((st.member, tags, st.vote))
    return sorted(out, key=lambda t: t[0].id)


def votes(pp) -> list:
    """[(member, vote)] for everyone who voted."""
    return sorted(((st.member, st.vote) for st in pp.member_states if st.vote), key=lambda t: t[0].id)


def average_vote(pp):
    v = [st.vote for st in pp.member_states if st.vote]
    return sum(v) / len(v) if v else None


# ── Switching per-person mode on and off ───────────────────────────────────────────────────────────────

def set_per_person(project, on: bool) -> None:
    """Turn per-person mode on or off. Round trips change nothing (tests/test_per_person.py):
    - Off: every paper's shared tags become every tag anyone currently sees on it (their own, or the shared
      ones if they never changed them), so nothing tagged meanwhile seems lost. Whoever still saw the old shared
      tags gets those as their own first, so turning it on again shows everyone what they saw before.
    - On: a paper whose shared tags were changed while the mode was off follows those tags for everyone
      (everyone saw that change); otherwise everyone gets their own tags back.
    Caller commits."""
    if bool(project.per_person) == on:
        return
    people = members(project)
    for pp in project.papers:
        if on:
            if pp.tags_at_off is not None and set(json.loads(pp.tags_at_off)) != set(pp.tags_list):
                for st in pp.member_states:
                    st.tags = None
            pp.tags_at_off = None
        else:
            seen = {m.id: tags_for(pp, m) for m in people}
            merged = [t for t in TAGS if any(t in tags for tags in seen.values())] if people else pp.tags_list
            if set(merged) != set(pp.tags_list):
                for m in people:
                    st = state(pp, m, create=True)
                    if st.tags is None:
                        st.tags = json.dumps(seen[m.id])
                pp.paper_tags = json.dumps(merged)
            pp.tags_at_off = json.dumps(pp.tags_list)
    project.per_person = on
    if on:
        sync_from_emails(project)


# ── Notes ───────────────────────────────────────────────────────────────────────────────────────────────

def runs(pp) -> list:
    """[[member_id or None, text, day "YYYY-MM-DD" or None], ...]. Old notes become one run nobody wrote; runs
    from before 2026-11-04 have no day."""
    if pp.notes_runs:
        runs = [[r[0], r[1], r[2] if len(r) > 2 else None] for r in json.loads(pp.notes_runs)]
        if "".join(r[1] for r in runs) == (pp.notes or ""):
            return runs
        # edited while per-person mode was off: the plain text is newer, its authors are unknown
    return [[None, pp.notes, None]] if pp.notes else []


def merge_runs(runs_in: list) -> list:
    """Drop empty runs and join neighbours by the same person on the same day."""
    out = []
    for r in runs_in:
        mid, text, day = r[0], r[1], r[2] if len(r) > 2 else None
        if not text:
            continue
        if out and out[-1][0] == mid and out[-1][2] == day:
            out[-1][1] += text
        else:
            out.append([mid, text, day])
    return out


def notes_for_llm(pp) -> str:
    """The note as plain text (the paper chat reads it)."""
    return pp.notes or ""
