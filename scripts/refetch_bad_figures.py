#!/usr/bin/env python3
"""
One-off: re-fetch figure URLs for papers that have a known-bad figure stored
(UI chrome, funder logos, static/base images, etc.).

Run from repo root:
    DATABASE_URL=sqlite:////data/related_work.db .venv/bin/python scripts/refetch_bad_figures.py
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
_log = logging.getLogger(__name__)

# Patterns that indicate a stored figure_url is UI chrome, not a paper figure.
# Must stay in sync with _FIGURE_SKIP in crawl.py.
_BAD_PATTERNS = (
    "static/base",
    "static/browse",
    "static/v",
    "funder",
    "smileybones",
    "arxiv-logotype",
    "social/",
    "mathjax",
    "favicon",
    "banner",
    "header",
)


def is_bad_figure(url: str) -> bool:
    if not url:
        return False
    url_lower = url.lower()
    return any(p in url_lower for p in _BAD_PATTERNS)


def main():
    from app import create_app, db
    from app.models import Paper
    from app.crawl import get_paper_figure

    app = create_app()
    with app.app_context():
        bad_papers = [p for p in Paper.query.filter(Paper.figure_url.isnot(None)).all()
                      if is_bad_figure(p.figure_url)]

        _log.info("Found %d papers with bad figure URLs", len(bad_papers))
        if not bad_papers:
            _log.info("Nothing to do.")
            return

        fixed = 0
        cleared = 0
        for paper in bad_papers:
            if not paper.arxiv_id or paper.arxiv_id.startswith("web:"):
                _log.info("[skip] %s — no arXiv ID", paper.title[:60])
                continue

            _log.info("[refetch] %s (%s)", paper.arxiv_id, paper.figure_url[:80])
            try:
                new_url, new_caption = get_paper_figure(paper.arxiv_id)
            except Exception as e:
                _log.warning("[error] %s: %s", paper.arxiv_id, e)
                continue

            if new_url:
                _log.info("  → %s", new_url[:100])
                paper.figure_url = new_url
                paper.figure_caption = new_caption
                fixed += 1
            else:
                _log.info("  → no figure found, clearing")
                paper.figure_url = None
                paper.figure_caption = None
                cleared += 1

        db.session.commit()
        _log.info("Done: %d fixed, %d cleared (no figure available)", fixed, cleared)


if __name__ == "__main__":
    main()
