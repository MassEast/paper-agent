"""Figure 1 of an arXiv paper, cut out of its PDF as a PNG.

arXiv's HTML version only has an image file for figures that were images in the LaTeX source; a figure drawn
in TikZ is inline SVG there, and a figure made of several images is several files. The PDF has every figure
as drawn, so we find the "Figure 1" caption, take the drawings, images and labels right above it, and render
that area. Saved in FIGURES_DIR (next to the SQLite DB unless set) and served at /figures/<file>.
"""

import os
import re
from typing import Optional

import httpx
import pymupdf

from app.utils import _ArxivLockCtx, _http_client

URL_PREFIX = "/figures/"
DPI = 200
GAP_PT = 14            # elements further apart than this vertically are not the same figure
BODY_TEXT_WORDS = 30  # a text block with this many words is a paragraph, not a label (hidden LaTeX data in a figure has long strings without spaces)
MIN_HEIGHT_PT = 30
HEADER_FRAC = 0.08     # the top strip of a page holds the running header and its rule, never the figure
MARGIN_PT = 4
CAPTION_MAX_CHARS = 1200
_CAPTION_RE = re.compile(r"\s*(?:Figure|Fig\.?)\s*1\s*[:.|]")


def figures_dir() -> str:
    if os.environ.get("FIGURES_DIR"):
        return os.environ["FIGURES_DIR"]
    db_url = os.environ.get("DATABASE_URL", "sqlite:////tmp/test.db")
    return os.path.join(os.path.dirname(db_url.removeprefix("sqlite:///")), "figures")


def file_name(arxiv_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]", "_", arxiv_id) + ".png"


def local_path(figure_url: str) -> Optional[str]:
    """The file behind a figure_url of ours, None for an external URL."""
    if not figure_url.startswith(URL_PREFIX):
        return None
    return os.path.join(figures_dir(), os.path.basename(figure_url))


def _clean_caption(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)  # words hyphenated at a line end
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > CAPTION_MAX_CHARS:
        text = text[:CAPTION_MAX_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def crop_figure_1(pdf: bytes) -> tuple[Optional[bytes], Optional[str]]:
    """(PNG, caption) of Figure 1, or (None, None) if there is no "Figure 1" caption with a figure above it."""
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    for page in doc:
        blocks = page.get_text("blocks")
        cap = next((b for b in blocks if _CAPTION_RE.match(b[4])), None)
        if cap is None:
            continue
        cap_rect = pymupdf.Rect(cap[:4])

        header = page.rect.y0 + page.rect.height * HEADER_FRAC

        def beside_caption(r):  # same column as the caption, above it, below the running header
            return r.x1 > cap_rect.x0 and r.x0 < cap_rect.x1 and header < r.y1 <= cap_rect.y0 + 2

        # Everything that can be part of the figure, and the paragraphs that bound it
        graphics, labels, bounds = [], [], []
        for d in page.get_drawings():
            r = d["rect"]
            if beside_caption(r) and r.height < page.rect.height * 0.8:  # not a page background
                graphics.append(r)
        for img in page.get_images(full=True):
            for r in page.get_image_rects(img[0]):
                if beside_caption(r):
                    graphics.append(r)
        # A caption of two or more lines is as wide as its column; a one-line caption may be centered and narrower
        column_known = len(cap[4].strip().splitlines()) >= 2

        stick_out = page.rect.width * 0.05

        def is_body_text(r, text):
            if len(text.split()) >= BODY_TEXT_WORDS:
                return True
            if not column_known or any(g.contains(r) for g in graphics):  # text in a box is the figure's
                return False
            # Sticks out of the column: title, authors or the other column's text (Figure 1 beside the abstract)
            if r.x0 < cap_rect.x0 - stick_out or r.x1 > cap_rect.x1 + stick_out:
                return True
            # A single line as wide as the column: one line of a paragraph (PDFs made in Word give every line its
            # own block). Labels side by side on one row make a block of several lines, so they don't count.
            return len(text.strip().splitlines()) == 1 and len(text.split()) >= 5 and r.width >= 0.9 * cap_rect.width

        for b in blocks:
            r = pymupdf.Rect(b[:4])
            if b is cap or not beside_caption(r):
                continue
            (bounds if is_body_text(r, b[4]) else labels).append(r)
        # The short last line of a paragraph: starts at the column's edge right below a paragraph line
        if column_known:
            for r in sorted(labels, key=lambda r: r.y0):
                if (abs(r.x0 - cap_rect.x0) < 2 and not any(g.contains(r) for g in graphics)
                        and any(0 <= r.y0 - p.y1 < 10 and abs(p.x0 - r.x0) < 2 for p in bounds)):
                    labels.remove(r)
                    bounds.append(r)
        # Grow upward from the caption while the next part touches what we have
        floor = max((r.y1 for r in bounds), default=page.rect.y0)
        parts = sorted((r for r in graphics + labels if r.y0 >= floor - 1), key=lambda r: -r.y1)
        if not parts or cap_rect.y0 - parts[0].y1 > GAP_PT * 2:
            return None, None
        area = pymupdf.Rect(parts[0])
        for r in parts[1:]:
            if r.y1 < area.y0 - GAP_PT:
                break
            area |= r
        area &= page.rect
        # Labels alone (no drawing or image in the area) are not a figure
        if area.height < MIN_HEIGHT_PT or not any(r.y0 >= area.y0 - 1 for r in graphics):
            return None, None
        # A little margin, but never into the paragraph above or the caption below
        clip = pymupdf.Rect(area.x0 - MARGIN_PT, max(area.y0 - MARGIN_PT, floor + 0.5),
                            area.x1 + MARGIN_PT, min(area.y1 + MARGIN_PT, cap_rect.y0 - 0.5)) & page.rect
        png = page.get_pixmap(clip=clip, dpi=DPI).tobytes("png")
        return png, _clean_caption(cap[4])
    return None, None


def _download_pdf(arxiv_id: str) -> Optional[bytes]:
    try:
        with _ArxivLockCtx():
            with _http_client(timeout=httpx.Timeout(60.0), follow_redirects=True) as http:
                r = http.get(f"https://arxiv.org/pdf/{arxiv_id}")
    except httpx.HTTPError as e:
        print(f"PDF download failed for {arxiv_id}: {e}")
        return None
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        print(f"no PDF for {arxiv_id} (HTTP {r.status_code})")
        return None
    return r.content


def pdf_figure(arxiv_id: str) -> tuple[Optional[str], Optional[str]]:
    """(figure_url, caption) of Figure 1 cut from the arXiv PDF and saved, or (None, None)."""
    pdf = _download_pdf(arxiv_id)
    if pdf is None:
        return None, None
    png, caption = crop_figure_1(pdf)
    if png is None:
        return None, None
    os.makedirs(figures_dir(), exist_ok=True)
    name = file_name(arxiv_id)
    with open(os.path.join(figures_dir(), name), "wb") as f:
        f.write(png)
    return URL_PREFIX + name, caption
