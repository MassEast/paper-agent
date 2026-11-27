"""Email the latest CHANGELOG.md entry: one email per address, so nobody sees the others.

    python scripts/send_changelog.py --test          # only to Ju, subject prefixed "[Test]"
    python scripts/send_changelog.py --to a@x,b@y    # the real send (addresses from the projects' email lists)
    python scripts/send_changelog.py --dry           # print subject, recipients and HTML, send nothing
    ... --since 2026-11-12                           # also list older entries from that date on, under "Also new"

Needs RESEND_API_KEY / RESEND_FROM / NOTIFIER_TYPE and SERVER_URL (the site link) in the environment or .env.
"""
import argparse
import html
import os
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

TEST_TO = "justus.westerhoff@bht-berlin.de"


def entries() -> list[tuple[date, list[str], str | None]]:
    """(date, bullet lines, code-changes link) of every "## YYYY-MM-DD" entry, newest first."""
    text = (ROOT / "CHANGELOG.md").read_text()
    out = []
    for m in re.finditer(r"^## (\d{4}-\d{2}-\d{2})\n(.*?)(?=^## |\Z)", text, re.M | re.S):
        body = m.group(2)
        bullets = [line[2:].strip() for line in body.splitlines() if line.startswith("- ")]
        link = re.search(r"\[Code changes\]\((\S+)\)", body)
        out.append((date.fromisoformat(m.group(1)), bullets, link.group(1) if link else None))
    return out


def code_link(picked) -> str | None:
    """One compare link from the oldest picked entry's base to the newest one's head."""
    links = [link for _, _, link in picked if link]
    if not links:
        return None
    head = re.search(r"\.\.\.(\w+)$", links[0])
    base = re.search(r"/compare/(\w+)\.\.\.", links[-1])
    return links[0][:base.start(1)] + base.group(1) + "..." + head.group(1)


def _list(bullets: list[str]) -> str:
    items = "".join(f'<li style="margin: 0 0 8px 0;">{html.escape(b)}</li>' for b in bullets)
    return f'<ul style="margin: 0 0 18px 0; padding-left: 20px; line-height: 1.5;">{items}</ul>'


def render(picked, site: str) -> tuple[str, str]:
    day = picked[0][0]
    when = f"{day.strftime('%B')} {day.day}, {day.year}"
    older = [b for _, bullets, _ in picked[1:] for b in bullets]
    link = code_link(picked)
    body = (f'<p style="margin: 0 0 14px 0;">Hi, here is what changed in Paper Agent ({when}):</p>'
            + _list(picked[0][1])
            + (f'<p style="margin: 0 0 10px 0;">Also new in the last weeks:</p>' + _list(older) if older else "")
            + f'<p style="margin: 0 0 18px 0;"><a href="{html.escape(site)}" style="color: #7c3aed;">Open the app</a></p>'
            f'<p style="margin: 0; font-size: 12px; color: #6b7280;">You get this because your address is on a '
            f'project\'s email list. Questions, ideas or rather no such emails: write to {TEST_TO}.'
            + (f' <a href="{html.escape(link)}" style="color: #6b7280;">Technical details</a>' if link else "")
            + "</p>")
    return f"What's new in Paper Agent ({when})", body


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test", action="store_true")
    g.add_argument("--to")
    g.add_argument("--dry", action="store_true")
    ap.add_argument("--since", type=date.fromisoformat, help="also list entries from this date on")
    args = ap.parse_args()

    all_entries = entries()
    picked = [all_entries[0]] + [e for e in all_entries[1:] if args.since and e[0] >= args.since]
    subject, body = render(picked, os.environ["SERVER_URL"])
    to = [TEST_TO] if args.test else [a.strip() for a in (args.to or "").split(",") if a.strip()]
    if args.test:
        subject = "[Test] " + subject
    print(f"Subject: {subject}\nEntries: {', '.join(str(d) for d, _, _ in picked)}, "
          f"{sum(len(b) for _, b, _ in picked)} points, recipients: {len(to)}", flush=True)
    if args.dry:
        print(body)
        return
    from app.crawl import send_notification
    sent = 0
    for i, addr in enumerate(to, 1):
        ok = send_notification(subject, "", project_emails=addr, content_html=body)
        sent += bool(ok)
        print(f"[{i}/{len(to)}] {addr}: {'sent' if ok else 'FAILED'}", flush=True)
    print(f"Done: {sent}/{len(to)} sent", flush=True)
    if sent < len(to):
        sys.exit(1)


if __name__ == "__main__":
    main()
