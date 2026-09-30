"""One-off: strip the version suffix from stored arXiv ids, merging papers that exist as several versions.

Usage: python scripts/normalize_arxiv_ids.py <sqlite-db-path> [--apply]   (dry run without --apply)

Papers whose ids differ only by version (`2402.02834`, `2402.02834v2`) become one Paper row. Survivor =
the highest version (unversioned counts as newest). Per project the two ProjectPaper links are merged:
notes are concatenated, tags unioned, and the state follows: in collection (either) > trashed (either) > new.
A summary is kept from the survivor, else moved from a loser. Back up the DB before --apply.
"""
import json
import re
import sqlite3
import sys
from collections import defaultdict

TAG_ORDER = ["important", "to_read", "to_discuss"]


def _base(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def _version(arxiv_id: str) -> float:
    m = re.search(r"v(\d+)$", arxiv_id)
    return int(m.group(1)) if m else float("inf")  # unversioned resolves to the latest


def _merge_links(a: dict, b: dict) -> dict:
    """Merge link row b into a (same project). Returns the fields to write on a."""
    tags = set(json.loads(a["paper_tags"] or "[]")) | set(json.loads(b["paper_tags"] or "[]"))
    tags_out = [t for t in TAG_ORDER if t in tags] + sorted(tags - set(TAG_ORDER))
    notes = [n for n in (a["notes"], b["notes"]) if n and n.strip()]
    notes_out = notes[0] if len(notes) == 1 or (len(notes) == 2 and notes[0].strip() == notes[1].strip()) else "\n\n---\n\n".join(notes) or None
    in_collection = a["manual_tag"] == "related" or b["manual_tag"] == "related"
    trashed = [t for t in (a["trashed_at"], b["trashed_at"]) if t]
    collected = [t for t in (a["collected_at"], b["collected_at"]) if t]
    return {
        "manual_tag": "related" if in_collection else None,
        "paper_tags": json.dumps(tags_out),
        "notes": notes_out,
        "notes_height": max(filter(None, (a["notes_height"], b["notes_height"])), default=None),
        "added_at": min(filter(None, (a["added_at"], b["added_at"])), default=None),
        "collected_at": min(collected) if collected else None,
        "trashed_at": None if in_collection else (min(trashed) if trashed else None),
    }


def normalize(conn: sqlite3.Connection, apply: bool) -> dict:
    conn.row_factory = sqlite3.Row
    papers = conn.execute("SELECT * FROM papers WHERE arxiv_id NOT LIKE 'web:%'").fetchall()
    groups = defaultdict(list)
    for p in papers:
        groups[_base(p["arxiv_id"])].append(p)

    stats = {"renamed": 0, "merged_groups": 0, "papers_deleted": 0, "links_merged": 0, "links_moved": 0}
    for base, members in groups.items():
        members.sort(key=lambda p: _version(p["arxiv_id"]), reverse=True)
        survivor, losers = members[0], members[1:]
        survivor_id = survivor["id"]
        if losers:
            stats["merged_groups"] += 1
        for loser in losers:
            lid = loser["id"]
            # fill gaps on the survivor from the loser (never overwrite survivor data)
            for col in ("authors", "institutions", "abstract", "page_count", "semantic_scholar_id", "figure_url", "figure_caption", "published_date", "year"):
                if survivor[col] in (None, "", "[]") and loser[col] not in (None, "", "[]"):
                    conn.execute(f"UPDATE papers SET {col} = ? WHERE id = ?", (loser[col], survivor_id))
            conn.execute("UPDATE papers SET citation_count = MAX(COALESCE(citation_count,0), ?) WHERE id = ?", (loser["citation_count"] or 0, survivor_id))
            # summary: keep survivor's, else adopt the loser's
            if conn.execute("SELECT 1 FROM paper_summaries WHERE paper_id = ?", (survivor_id,)).fetchone():
                conn.execute("DELETE FROM paper_summaries WHERE paper_id = ?", (lid,))
            else:
                conn.execute("UPDATE paper_summaries SET paper_id = ? WHERE paper_id = ?", (survivor_id, lid))
            # project links
            for link in conn.execute("SELECT * FROM project_papers WHERE paper_id = ?", (lid,)).fetchall():
                other = conn.execute("SELECT * FROM project_papers WHERE paper_id = ? AND project_id = ?", (survivor_id, link["project_id"])).fetchone()
                if other is None:
                    conn.execute("UPDATE project_papers SET paper_id = ? WHERE id = ?", (survivor_id, link["id"]))
                    stats["links_moved"] += 1
                else:
                    m = _merge_links(dict(other), dict(link))
                    conn.execute(
                        "UPDATE project_papers SET manual_tag=?, paper_tags=?, notes=?, notes_height=?, added_at=?, collected_at=?, trashed_at=? WHERE id=?",
                        (m["manual_tag"], m["paper_tags"], m["notes"], m["notes_height"], m["added_at"], m["collected_at"], m["trashed_at"], other["id"]),
                    )
                    conn.execute("DELETE FROM project_papers WHERE id = ?", (link["id"],))
                    stats["links_merged"] += 1
            conn.execute("DELETE FROM papers WHERE id = ?", (lid,))
            stats["papers_deleted"] += 1
        if survivor["arxiv_id"] != base:
            conn.execute("UPDATE papers SET arxiv_id = ? WHERE id = ?", (base, survivor_id))
            stats["renamed"] += 1
    # pdf_url of every arXiv paper, including ones whose id was already unversioned
    for row in conn.execute("SELECT id, pdf_url FROM papers WHERE arxiv_id NOT LIKE 'web:%' AND pdf_url IS NOT NULL").fetchall():
        stripped = re.sub(r"v\d+$", "", row["pdf_url"])
        if stripped != row["pdf_url"]:
            conn.execute("UPDATE papers SET pdf_url = ? WHERE id = ?", (stripped, row["id"]))
            stats["pdf_urls_fixed"] = stats.get("pdf_urls_fixed", 0) + 1
    if apply:
        conn.commit()
    else:
        conn.rollback()
    return stats


if __name__ == "__main__":
    path, apply = sys.argv[1], "--apply" in sys.argv[2:]
    conn = sqlite3.connect(path)
    print(("APPLIED" if apply else "DRY RUN (rolled back)"), normalize(conn, apply))
    print("integrity:", conn.execute("PRAGMA integrity_check").fetchone()[0], "| fk violations:", len(conn.execute("PRAGMA foreign_key_check").fetchall()))
