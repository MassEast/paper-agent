"""scripts/send_changelog.py: reads CHANGELOG.md and builds the email."""
import importlib.util
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("send_changelog", ROOT / "scripts" / "send_changelog.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def test_every_entry_has_bullets_and_a_code_link_newest_first():
    es = sc.entries()
    assert es and all(bullets and link for _, bullets, link in es)
    assert [d for d, _, _ in es] == sorted((d for d, _, _ in es), reverse=True)


def test_older_entries_go_under_also_new_and_one_link_spans_all():
    es = [(date(2026, 11, 27), ["new thing"], "https://github.com/x/y/compare/bbb...ccc"),
          (date(2026, 11, 12), ["older thing"], "https://github.com/x/y/compare/aaa...bbb")]
    subject, body = sc.render(es, "https://site")
    assert subject == "What's new in Paper Agent (November 27, 2026)"
    assert body.index("new thing") < body.index("Also new in the last weeks") < body.index("older thing")
    assert "compare/aaa...ccc" in body


def test_latest_only_has_no_also_new():
    _, body = sc.render(sc.entries()[:1], "https://site")
    assert "Also new" not in body
