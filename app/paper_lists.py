"""The three paper sections of a project page (New Papers, My Collection, Trash), sorted and filtered on
the server and sent a page at a time.

A long project (several hundred cards, ~20 KB each) was one multi-MB page that phones struggled to render,
and every collect/trash click re-sent all of it. Now each section renders PAGE_SIZE cards; a sentinel at
the end of a list fetches the next page when it scrolls into view (/projects/<slug>/papers/more), and a
card action re-renders only the section the paper moved into.

The page keeps the sort/filter state in JS (window._paperSort / _paperFilter, see paper_list.html) and
sends it with every htmx request; list_state() reads it back. Defaults match the JS defaults.
"""

from datetime import datetime

from app.models import Paper, ProjectPaper

PAGE_SIZE = 10
SECTIONS = ("new", "col", "trash")
TAGS = ("important", "to_read", "to_discuss")
_TAG_GROUP_ORDER = {"to_read": 0, "to_discuss": 1, "important": 2}  # "Group by tag" order, untagged last


def list_state(values) -> dict:
    """Sort/filter state from request args/form (sent by the page with every htmx request)."""
    def _by(key):
        return "pub" if values.get(key) == "pub" else "added"

    def _dir(key):
        return "asc" if values.get(key) == "asc" else "desc"

    return {
        "new_by": _by("new_by"), "new_dir": _dir("new_dir"),
        "col_by": _by("col_by"), "col_dir": _dir("col_dir"),
        "col_group": values.get("col_group") == "1",
        "col_tags": [t for t in (values.get("col_tags") or "").split(",") if t in TAGS],
        "search": (values.get("search") or "").strip(),
    }


def added_key(pp):
    """'Date added': for a My Collection paper, when it was collected (older links have no collected_at)."""
    return (pp.collected_at if pp.manual_tag and pp.collected_at else pp.added_at) or datetime.min


def _date_sort(papers, by, direction):
    if by == "pub":
        key = lambda pp: pp.paper.published_date.isoformat() if pp.paper.published_date else ""
    else:
        key = lambda pp: added_key(pp).isoformat()
    return sorted(papers, key=key, reverse=direction == "desc")


def _tag_group(pp):
    return min((_TAG_GROUP_ORDER[t] for t in pp.tags_list if t in _TAG_GROUP_ORDER), default=3)


def sections(project, state) -> dict:
    """Every section's papers, sorted and filtered, plus the counts the headers show."""
    query = ProjectPaper.query.filter_by(project_id=project.id)
    if state["search"]:
        q = f"%{state['search']}%"
        query = query.join(Paper).filter(
            Paper.title.ilike(q) | Paper.abstract.ilike(q) | Paper.authors.ilike(q)
            | Paper.institutions.ilike(q) | ProjectPaper.notes.ilike(q)
        )
    new, col, trash = [], [], []
    for pp in query.all():
        if pp.trashed_at:
            trash.append(pp)
        elif pp.manual_tag is None:
            new.append(pp)
        else:
            col.append(pp)

    tag_counts = {t: sum(t in pp.tags_list for pp in col) for t in TAGS}
    col_total = len(col)
    if state["col_tags"]:
        col = [pp for pp in col if any(t in pp.tags_list for t in state["col_tags"])]
    col = _date_sort(col, state["col_by"], state["col_dir"])
    if state["col_group"]:
        col = sorted(col, key=_tag_group)  # stable: date order kept inside each group
    trash.sort(key=lambda pp: pp.trashed_at, reverse=True)
    return {
        "new": _date_sort(new, state["new_by"], state["new_dir"]),
        "col": col,
        "trash": trash,
        "col_total": col_total,
        "tag_counts": tag_counts,
    }
