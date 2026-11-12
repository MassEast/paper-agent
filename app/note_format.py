"""Notes as HTML for emails (Notify), formatted like the card shows them (app/static/notes_format.js): links are
clickable, a link to a paper of this app shows as that paper's title, **text** is bold."""
import html
import os
import re
from urllib.parse import parse_qs, urlparse

_TOKEN = re.compile(r"\*\*([^*\n]+?)\*\*|(https?://[^\s<>\"']+)")
_LINK_STYLE = "color:#7c3aed;"


def _trim_url(url: str) -> str:
    """Without a sentence's full stop, or a closing bracket the link didn't open (like notes_format.js)."""
    url = re.sub(r"[.,;:!?]+$", "", url)
    while url and url[-1] in ")]":
        opening = "(" if url[-1] == ")" else "["
        if url.count(opening) >= url.count(url[-1]):
            break
        url = re.sub(r"[.,;:!?]+$", "", url[:-1])
    return url


def _paper_ref(url: str) -> str | None:
    """What a link to a paper of this app names (SERVER_URL's host, /projects/<slug>?paper=<ref>, Paper.by_ref)."""
    u, server = urlparse(url), urlparse(os.environ.get("SERVER_URL", "http://localhost:5001"))
    if u.netloc != server.netloc or not re.fullmatch(r"/projects/[\w-]+/?", u.path):
        return None
    return parse_qs(u.query).get("paper", [""])[0] or None


def _link(url: str) -> str:
    from app import db
    from app.models import Paper
    ref = _paper_ref(url)
    paper = Paper.by_ref(ref) if ref else None
    text = f"“{paper.title}”" if paper else url
    return f'<a href="{html.escape(url)}" style="{_LINK_STYLE}">{html.escape(text)}</a>'


def to_html(text: str) -> str:
    """Escaped HTML of a note or message: newlines as <br>, links as <a>, **text** as <b>."""
    out, last = [], 0
    for m in _TOKEN.finditer(text):
        if m.start() < last:
            continue
        raw = m.group(0) if m.group(1) is not None else _trim_url(m.group(2))
        out.append(html.escape(text[last:m.start()]))
        out.append(f"<b>{to_html(m.group(1))}</b>" if m.group(1) is not None else _link(raw))
        last = m.start() + len(raw)
    out.append(html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>")
