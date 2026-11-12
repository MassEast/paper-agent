"""What the paper says about itself, for the card (redesign 2026-11-12): its own code repository and project
page, the corresponding authors' emails and the venue it appeared at.

Regex collects the candidates (URLs and emails in the abstract, the arXiv comment and the paper's text before its
references); only if there is something to decide, one small LLM call picks the paper's own links, matches the
emails to authors and names the venue. Every answer is checked against the candidates: a URL or email the model
returns must be one we found in the text, an author must be on the paper, and the venue must be named in the
text, so a made-up link can't get onto a card.
"""
import json
import logging
import re
from datetime import datetime
from urllib.parse import urlparse

from app import prompts
from app.llm import LLMUnavailableError, _llm_json

_log = logging.getLogger(__name__)

EMAIL_CHARS = 15000  # emails are in the author block or a footnote on the first page
START_CHARS = 2500  # the start of the paper the model sees: author block, markers, a venue header
FIRST_PAGE_CHARS = 5000  # the PDF's first page: author block, marks and their legend, footnotes
CONTEXT_CHARS = 160  # text on each side of a candidate the model sees
MAX_URLS = 25
MAX_EMAILS = 20
MAX_CORRESPONDING = 3

# Links that are never the paper's code or page
_SKIP_HOSTS = ("arxiv.org", "doi.org", "creativecommons.org", "orcid.org", "latexml", "nist.gov", "w3.org",
               "acm.org", "ieee.org", "springer.com", "aclanthology.org", "openreview.net", "neurips.cc",
               "mlr.press", "thecvf.com", "aaai.org", "ijcai.org", "jmlr.org", "biorxiv.org", "medrxiv.org",
               "scholar.google", "wikipedia.org", "bit.ly", "goo.gl")
_URL_RE = re.compile(r"https?://[^\s<>\"'(){}\[\]|\\^`]+", re.I)
# Without a scheme, as papers often print them: github.com/org/repo, name.github.io
_BARE_URL_RE = re.compile(
    r"(?<![\w@/.:-])((?:www\.)?(?:github\.com|gitlab\.com|huggingface\.co|bitbucket\.org)/[^\s<>\"'(){}\[\]|\\^`]+"
    r"|[\w-]+\.github\.io(?:/[^\s<>\"'(){}\[\]|\\^`]*)?)", re.I)
_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
# Grouped emails as in many PDFs: {alice, bob}@uni.edu or (alice|bob)@uni.edu
_EMAIL_GROUP_RE = re.compile(r"[{(]\s*([A-Za-z0-9._%+\-,;|\s]{1,300}?)\s*[})]\s*@\s*"
                             r"([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})")
_REFERENCES_RE = re.compile(r"\b(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\b")
_VENUE_HINT_RE = re.compile(r"\b(accepted|published|to appear|appear(?:s|ed)? (?:at|in)|proceedings|conference|"
                            r"workshop|journal|camera[- ]ready|oral|spotlight)\b", re.I)
# "ICLR 2025", "NeurIPS'24", "ACL 2025": an acronym and a year, as arXiv comments often name the venue alone
_VENUE_NAME_RE = re.compile(r"\b[A-Z][A-Za-z]*[A-Z]\s?'?(?:19|20)?\d{2}\b")
# Words that don't name a venue on their own (the check needs one that does, e.g. "ICLR")
_GENERIC_VENUE_WORDS = {"conference", "workshop", "proceedings", "international", "annual", "journal", "the",
                        "and", "track", "main", "findings", "paper", "papers", "spotlight", "oral", "poster",
                        "transactions", "symposium", "meeting", "association", "advances", "systems", "learning",
                        "for", "with", "datasets", "benchmarks", "short", "long", "special", "issue", "letters"}


def body_text(text: str) -> str:
    """The paper's text before its references (the last "References" heading in the second half or later of the
    text, so a sentence that mentions references early on doesn't cut it)."""
    last = None
    for m in _REFERENCES_RE.finditer(text):
        last = m
    return text[:last.start()] if last and last.start() > len(text) * 0.3 else text


def _clean_url(url: str) -> str:
    url = url.rstrip(".,;:!?'\"*")
    url = re.sub(r"(\.git)?/?$", "", url) if "github.com/" in url.lower() else url.rstrip("/")
    return url if url.lower().startswith(("http://", "https://")) else "https://" + url


def _url_key(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url.lower()).rstrip("/")


def _context(text: str, needle: str) -> str:
    i = text.find(needle)
    if i < 0:
        return ""
    return re.sub(r"\s+", " ", text[max(0, i - CONTEXT_CHARS):i + len(needle) + CONTEXT_CHARS]).strip()


def candidate_urls(sources: list[str]) -> list[tuple[str, str]]:
    """(url, context) for every link in the sources (abstract, comment, body), first occurrence first, without
    links that are never a paper's own code or page."""
    seen, out = set(), []
    for src in sources:
        found = [(m.start(), m.group(0)) for m in _URL_RE.finditer(src)]
        found += [(m.start(), m.group(1)) for m in _BARE_URL_RE.finditer(src)]
        for _, raw in sorted(found):
            url = _clean_url(raw)
            host = (urlparse(url).hostname or "").lower()
            key = _url_key(url)
            if not host or "." not in host or key in seen or any(h in host for h in _SKIP_HOSTS):
                continue
            seen.add(key)
            out.append((url, _context(src, raw)))
            if len(out) >= MAX_URLS:
                return out
    return out


def candidate_emails(text: str) -> list[tuple[str, str]]:
    """(email, context) for every email in the text, grouped ones ({a, b}@uni.edu) split up, in order."""
    found = []
    for m in _EMAIL_GROUP_RE.finditer(text):
        for local in re.split(r"[,;|\s]+", m.group(1)):
            if re.fullmatch(r"[A-Za-z0-9._%+-]+", local or ""):
                found.append((m.start(), f"{local}@{m.group(2)}".lower(), m.group(0)))
    for m in _EMAIL_RE.finditer(text):
        found.append((m.start(), m.group(0).rstrip(".").lower(), m.group(0)))
    seen, out = set(), []
    for _, email, raw in sorted(found):
        if email in seen:
            continue
        seen.add(email)
        out.append((email, _context(text, raw)))
        if len(out) >= MAX_EMAILS:
            break
    return out


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", name or "").strip().lower()


def _email_fits(email: str, name: str) -> bool:
    """The email's local part looks like this person's: it holds a part of the name (swang08 for Shaohuang Wang)
    or the first and last initials in either order (jcl for John Langford, bsmn0223 for Sangmin Bae). Keeps the
    model from giving one author's email to another (seen on 2611.13732); a lab's contact@ address fits nobody."""
    local = re.sub(r"[^a-z]", "", email.split("@")[0].lower())
    parts = [t for t in re.findall(r"[a-z]+", _norm_name(name)) if len(t) >= 2]
    if not parts or not local:
        return False
    if any(t in local for t in parts):
        return True

    def subseq(a: str, b: str) -> bool:
        it = iter(b)
        return all(ch in it for ch in a)
    first, last = parts[0][0], parts[-1][0]
    return len(parts) > 1 and (subseq(first + last, local) or subseq(last + first, local))


def _venue_backed(venue: str, source: str) -> bool:
    """The venue names something the text names too (e.g. "ICLR" in "ICLR 2026"), not only generic words."""
    words = [w for w in re.findall(r"[A-Za-z]{3,}", venue) if w.lower() not in _GENERIC_VENUE_WORDS]
    src = source.lower()
    return bool(words) and any(w.lower() in src for w in words)


def validate(raw: dict, urls: list[str], emails: list[str], authors: list[str], venue_source: str) -> dict:
    """Keep only what the text backs: links and emails we found, authors on the paper, a venue the text names."""
    raw = raw if isinstance(raw, dict) else {}
    by_key = {_url_key(u): u for u in urls}

    def pick_url(value):
        return by_key.get(_url_key(value)) if isinstance(value, str) and value.strip() else None

    code_url = pick_url(raw.get("code_url"))
    project_url = pick_url(raw.get("project_url"))
    if project_url == code_url:
        project_url = None

    names = {_norm_name(a): a for a in authors}
    email_set = set(emails)
    corresponding, used = [], set()
    for item in raw.get("corresponding") or []:
        if not isinstance(item, dict):
            continue
        name = names.get(_norm_name(item.get("author") or ""))
        email = (item.get("email") or "").strip().lower()
        if name and email in email_set and name not in used and _email_fits(email, name):
            corresponding.append({"name": name, "email": email})
            used.add(name)
    venue = raw.get("venue")
    venue = venue.strip() if isinstance(venue, str) and venue.strip() else None
    if venue and venue.lower() in ("null", "none", "n/a", "unknown"):
        venue = None
    if venue and (len(venue) > 80 or not _venue_backed(venue, venue_source)):
        venue = None
    return {"code_url": code_url, "project_url": project_url,
            "corresponding": corresponding[:MAX_CORRESPONDING], "venue": venue}


EMPTY = {"code_url": None, "project_url": None, "corresponding": [], "venue": None}


def extract(title: str, authors: list[str], text: str | None, abstract: str = "", comment: str | None = None,
            journal_ref: str | None = None, first_page: str | None = None) -> tuple[dict, int]:
    """({code_url, project_url, corresponding: [{name, email}], venue}, tokens) from what the paper says. No LLM
    call (0 tokens) when the text has no link, no email and nothing that sounds like a venue. first_page: the
    text of the PDF's first page (figures.first_page_text), which keeps the marks after the author names."""
    body = body_text(text or "")
    first_page = (first_page or "")[:FIRST_PAGE_CHARS]
    meta = " ".join(x for x in (comment, journal_ref) if x)
    urls = candidate_urls([abstract or "", meta, body])
    emails = list({e: c for e, c in candidate_emails(body[:EMAIL_CHARS]) + candidate_emails(first_page)}.items())
    start = body[:START_CHARS]
    venue_source = f"{meta} {start}"
    if not urls and not emails and not (journal_ref or _VENUE_HINT_RE.search(venue_source)
                                        or _VENUE_NAME_RE.search(meta)):
        return dict(EMPTY), 0
    prompt = prompts.PAPER_RESOURCES.format(
        title=title,
        authors="\n".join(f"- {a}" for a in authors) or "(unknown)",
        comment=comment or "(none)",
        journal_ref=journal_ref or "(none)",
        start=start or "(no full text)",
        first_page=first_page or "(not available)",
        urls="\n".join(f'[{i}] {u}\n    context: "{c}"' for i, (u, c) in enumerate(urls, 1)) or "(none)",
        emails="\n".join(f'[{i}] {e}\n    context: "{c}"' for i, (e, c) in enumerate(emails, 1)) or "(none)",
    )
    try:
        raw, _, _, usage = _llm_json([{"role": "user", "content": prompt}], temperature=0, max_tokens=2048)
    except LLMUnavailableError:
        raise
    except Exception as e:  # an answer that isn't JSON: nothing found, the card just shows no buttons
        _log.warning("[resources] %s: no usable answer (%s)", title[:60], e)
        return dict(EMPTY), 0
    found = validate(raw, [u for u, _ in urls], [e for e, _ in emails], authors, venue_source)
    return found, int(usage.get("total_tokens", 0))


def apply(paper, found: dict) -> None:
    """Store what extract found on the paper (caller commits)."""
    paper.code_url = found["code_url"]
    paper.project_url = found["project_url"]
    paper.corresponding_authors = json.dumps(found["corresponding"]) if found["corresponding"] else None
    paper.venue = found["venue"]
    paper.resources_checked_at = datetime.utcnow()


def first_page(paper) -> str | None:
    """The PDF's first page for an arXiv paper (one more download unless Figure 1 just fetched the PDF)."""
    from app import figures
    return figures.first_page_text(paper.arxiv_id) if paper.is_arxiv else None


def extract_for(paper, text: str | None) -> tuple[dict, int]:
    return extract(paper.title, paper.authors_list, text, paper.abstract or "", paper.arxiv_comment,
                   paper.journal_ref, first_page(paper))
