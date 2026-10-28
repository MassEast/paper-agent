#!/usr/bin/env python3
"""How often did arXiv's search refuse the nightly crawl? Reads the `[arxiv-outcome]` lines that
scripts/nightly_crawl.py writes (one per project and night) from crawl.log and its rotated copies.

On the cluster: kubectl exec deploy/related-work -- python scripts/arxiv_misses.py

A missed night loses nothing yet (each night searches the last CRAWL_DAYS_BACK days again). The
trigger for moving off the search API: a project missing 3 nights in a row, or more than 20% of all
project-nights missed.
"""
import glob
import os
import re
import sys
from collections import defaultdict

LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) [\d:,]+ INFO: \[arxiv-outcome\] answered=(yes|no) tries=(\d+) "
                  r"project_id=(\d+) name=(.*?) \[in ")
MAX_STREAK = 3
MAX_MISS_RATE = 0.20


def parse_outcomes(lines) -> dict[int, dict]:
    """{project_id: {"name": str, "nights": {date: (answered, tries)}}}. A project that ran twice in
    one night (a manual nightly test) keeps its last outcome."""
    out: dict[int, dict] = defaultdict(lambda: {"name": "", "nights": {}})
    for line in lines:
        m = LINE.match(line)
        if not m:
            continue
        day, answered, tries, pid, name = m.groups()
        out[int(pid)]["name"] = name
        out[int(pid)]["nights"][day] = (answered == "yes", int(tries))
    return dict(out)


def longest_miss_streak(nights: dict[str, tuple[bool, int]]) -> int:
    best = cur = 0
    for day in sorted(nights):
        cur = 0 if nights[day][0] else cur + 1
        best = max(best, cur)
    return best


def summarize(outcomes: dict[int, dict]) -> tuple[list[str], bool]:
    """Report lines and whether a trigger fired."""
    rows, total, missed, fired = [], 0, 0, False
    for pid, o in sorted(outcomes.items(), key=lambda kv: kv[1]["name"].lower()):
        nights = o["nights"]
        n_miss = sum(1 for a, _ in nights.values() if not a)
        n_retry = sum(1 for a, t in nights.values() if a and t > 1)
        streak = longest_miss_streak(nights)
        total += len(nights)
        missed += n_miss
        flag = "  <-- 3+ in a row" if streak >= MAX_STREAK else ""
        fired |= streak >= MAX_STREAK
        rows.append(f"  {o['name'][:32]:32s} {len(nights):3d} nights  {n_miss:3d} missed  "
                    f"{n_retry:3d} needed a retry  longest miss streak {streak}{flag}")
    rate = missed / total if total else 0.0
    fired |= rate > MAX_MISS_RATE
    days = sorted({d for o in outcomes.values() for d in o["nights"]})
    head = [f"arXiv outcomes {days[0]} .. {days[-1]} ({len(days)} nights)" if days else "no [arxiv-outcome] lines found"]
    tail = [f"Total: {missed}/{total} project-nights missed ({rate:.0%}); "
            f"trigger (3 in a row or >{MAX_MISS_RATE:.0%}): {'FIRED' if fired else 'not reached'}"]
    return head + rows + tail, fired


def main():
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LOGS_DIR", "/data/logs")
    def rotation(path):  # crawl.log.10 is the oldest, crawl.log the newest
        suffix = path.rsplit("crawl.log", 1)[1].lstrip(".")
        return -int(suffix) if suffix else 0
    paths = sorted(glob.glob(os.path.join(logs_dir, "crawl.log*")), key=rotation)
    lines = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines.extend(f)
    report, _ = summarize(parse_outcomes(lines))
    print(f"read {len(paths)} log file(s) from {logs_dir}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
