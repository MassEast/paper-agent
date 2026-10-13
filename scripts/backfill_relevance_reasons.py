#!/usr/bin/env python3
"""
One-off: fill the per-project "Why it's relevant" line (ProjectPaper.relevance_reason) for papers
that came in before it existed. New crawls write it themselves (deep relevance check).

Only projects with the nightly crawl on, and only papers not in Trash. A project with the
nightly crawl off is skipped on purpose: its research interest may be stale or oversized
(Hindsight-related: ~48k characters), and the line would be written against it.

Uses RELEVANCE_REASON (title + abstract, no full-text fetch, so no arXiv traffic): one LLM call
per paper, LLM_PARALLEL calls at a time. Safe to re-run: papers that already have a line are skipped,
so if it crashes (LLM endpoint down), just start it again.

    python scripts/backfill_relevance_reasons.py --dry-run     # counts only
    python scripts/backfill_relevance_reasons.py               # fill
On the cluster: kubectl exec deploy/related-work -- python scripts/backfill_relevance_reasons.py
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LLM_PARALLEL = int(os.environ.get("LLM_PARALLEL", "6"))


def todo_links():
    """(link_id, project name, title, abstract, research interest) for every paper still missing its line."""
    from app.models import Project, ProjectPaper

    rows = []
    projects = Project.query.filter(Project.crawl_hour.isnot(None), Project.trashed_at.is_(None)).order_by(Project.id)
    for project in projects:
        if not (project.research_interest or "").strip():
            continue
        for pp in ProjectPaper.query.filter_by(project_id=project.id, trashed_at=None, relevance_reason=None):
            if pp.paper:
                rows.append((pp.id, project.name, pp.paper.title, pp.paper.abstract or "", project.research_interest))
    return rows


def backfill(dry_run: bool = False) -> dict:
    from app import db
    from app.crawl import generate_relevance_reason
    from app.models import ProjectPaper

    rows = todo_links()
    per_project: dict[str, int] = {}
    for row in rows:
        per_project[row[1]] = per_project.get(row[1], 0) + 1
    for name, n in per_project.items():
        print(f"  {name}: {n} papers without a why-line", flush=True)
    print(f"{len(rows)} papers to fill, {LLM_PARALLEL} LLM calls in parallel", flush=True)
    result = {"total": len(rows), "filled": 0, "empty": 0}
    if dry_run or not rows:
        return result

    started = time.time()

    def _one(row):
        link_id, _, title, abstract, interest = row
        return link_id, title, generate_relevance_reason(title, abstract, interest)

    with ThreadPoolExecutor(max_workers=LLM_PARALLEL) as executor:
        futures = [executor.submit(_one, row) for row in rows]
        for done, future in enumerate(as_completed(futures), 1):
            link_id, title, why = future.result()  # LLM down → crash; a re-run skips filled papers
            if why:
                db.session.get(ProjectPaper, link_id).relevance_reason = why
                db.session.commit()
                result["filled"] += 1
            else:
                result["empty"] += 1
            elapsed = time.time() - started
            eta = elapsed / done * (len(rows) - done)
            print(f"[{done}/{len(rows)}] {elapsed:.0f}s, ~{eta:.0f}s left | {title[:70]} -> {why[:90]}", flush=True)

    print(f"Done in {time.time() - started:.0f}s: {result['filled']} filled, {result['empty']} empty answers "
          f"(re-run to retry those).", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="only count the papers that would be filled")
    args = parser.parse_args()

    from app import create_app
    with create_app().app_context():
        backfill(dry_run=args.dry_run)
