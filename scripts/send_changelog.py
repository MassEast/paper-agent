"""Email the latest CHANGELOG.md entry: one email per address, so nobody sees the others.

    python scripts/send_changelog.py --test          # only to Ju, subject prefixed "[Test]"
    python scripts/send_changelog.py --to a@x,b@y    # the real send (addresses from the projects' email lists)
    python scripts/send_changelog.py --dry           # print subject, recipients and HTML, send nothing

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


def latest_entry() -> tuple[date, list[str], str | None]:
    """(date, bullet lines, code-changes link) of the newest "## YYYY-MM-DD" entry."""
    text = (ROOT / "CHANGELOG.md").read_text()
    m = re.search(r"^## (\d{4}-\d{2}-\d{2})\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    body = m.group(2)
    bullets = [line[2:].strip() for line in body.splitlines() if line.startswith("- ")]
    link = re.search(r"\[Code changes\]\((\S+)\)", body)
    return date.fromisoformat(m.group(1)), bullets, link.group(1) if link else None


def render(day: date, bullets: list[str], link: str | None, site: str) -> tuple[str, str]:
    when = f"{day.strftime('%B')} {day.day}, {day.year}"
    items = "".join(f'<li style="margin: 0 0 8px 0;">{html.escape(b)}</li>' for b in bullets)
    body = (f'<p style="margin: 0 0 14px 0;">Hi, here is what changed in the Related Work Agent ({when}):</p>'
            f'<ul style="margin: 0 0 18px 0; padding-left: 20px; line-height: 1.5;">{items}</ul>'
            f'<p style="margin: 0 0 18px 0;"><a href="{html.escape(site)}" style="color: #7c3aed;">Open the app</a></p>'
            f'<p style="margin: 0; font-size: 12px; color: #6b7280;">You get this because your address is on a '
            f'project\'s email list. Questions, ideas or rather no such emails: write to {TEST_TO}.'
            + (f' <a href="{html.escape(link)}" style="color: #6b7280;">Technical details</a>' if link else "")
            + "</p>")
    return f"What's new in the Related Work Agent ({when})", body


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test", action="store_true")
    g.add_argument("--to")
    g.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    day, bullets, link = latest_entry()
    subject, body = render(day, bullets, link, os.environ["SERVER_URL"])
    to = [TEST_TO] if args.test else [a.strip() for a in (args.to or "").split(",") if a.strip()]
    if args.test:
        subject = "[Test] " + subject
    print(f"Subject: {subject}\nEntry {day}: {len(bullets)} points, recipients: {len(to)}", flush=True)
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
