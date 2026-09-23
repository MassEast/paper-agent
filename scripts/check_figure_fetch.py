#!/usr/bin/env python3
"""Manual check of figure fetching from arXiv HTML pages.
Usage: python scripts/check_figure_fetch.py [arxiv_id ...]
"""
import sys
import re
import httpx

_FIGURE_SKIP = (
    "icon", "logo", "social", "badge", "static/browse", "static/v", "favicon",
    "arxiv-logotype", "twitter", "linkedin", "facebook", "mathjax", "header",
    "banner", "button", "arrow", "sprite",
)

ARXIV_IDS = sys.argv[1:] if len(sys.argv) > 1 else [
    "2502.00000",  # placeholder — pass real IDs as args
]


def fetch_and_report(arxiv_id: str):
    print(f"\n{'='*60}")
    print(f"arxiv_id: {arxiv_id}")

    for label, page_url, base_url in [
        ("HTML page", f"https://arxiv.org/html/{arxiv_id}", f"https://arxiv.org/html/{arxiv_id}"),
        ("Abstract page", f"https://arxiv.org/abs/{arxiv_id}", "https://arxiv.org"),
    ]:
        print(f"\n--- {label}: {page_url}")
        try:
            resp = httpx.get(page_url, timeout=20.0, follow_redirects=True)
            print(f"    Status: {resp.status_code} (final url: {resp.url})")
            if resp.status_code != 200:
                continue

            content = resp.text
            img_srcs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', content, re.IGNORECASE)
            print(f"    Total <img> tags: {len(img_srcs)}")

            passing, skipped_why = [], []
            for src in img_srcs:
                src_lower = src.lower()
                skip_reason = next((s for s in _FIGURE_SKIP if s in src_lower), None)
                if skip_reason:
                    skipped_why.append((src[:80], f"skip:{skip_reason}"))
                    continue
                if not (
                    src_lower.endswith(".png")
                    or src_lower.endswith(".jpg")
                    or src_lower.endswith(".jpeg")
                    or src_lower.endswith(".gif")
                    or src_lower.endswith(".webp")
                    or "fig" in src_lower
                ):
                    skipped_why.append((src[:80], "no-image-ext"))
                    continue
                # Resolve relative URL
                if src.startswith("//"):
                    resolved = "https:" + src
                elif src.startswith("/"):
                    resolved = "https://arxiv.org" + src
                elif not src.startswith("http"):
                    resolved = base_url.rstrip("/") + "/" + src
                else:
                    resolved = src
                passing.append((src[:80], resolved))

            print(f"    Passing filter: {len(passing)}, Skipped: {len(skipped_why)}")
            if passing:
                print(f"    WOULD RETURN: {passing[0][1]}")
                for raw, resolved in passing[:5]:
                    print(f"      src={raw!r}  →  {resolved}")
            else:
                print("    No images passed filter.")
                print("    First 10 skipped:")
                for s, r in skipped_why[:10]:
                    print(f"      {r}: {s!r}")

        except Exception as e:
            print(f"    ERROR: {e}")


if __name__ == "__main__":
    if not ARXIV_IDS or ARXIV_IDS == ["2502.00000"]:
        # Try to pull some arxiv_ids from the DB
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.environ.setdefault("SECRET_KEY", "test")
        os.environ.setdefault("SITE_PASSWORD", "test")
        os.environ.setdefault("LLM_API_KEY", "test")
        os.environ.setdefault("LLM_API_BASE", "http://test-llm-endpoint.invalid/v1")
        os.environ.setdefault("LLM_MODELS", "test-model")
        from app import create_app
        app = create_app()
        with app.app_context():
            from app.models import Paper
            papers = Paper.query.filter(Paper.figure_url.isnot(None)).limit(3).all()
            if not papers:
                papers = Paper.query.limit(5).all()
            ids = [p.arxiv_id for p in papers]
            figure_urls = {p.arxiv_id: p.figure_url for p in papers}
        print("Testing papers from DB:")
        for aid in ids:
            print(f"  {aid}  (stored figure_url: {figure_urls.get(aid, 'none')})")
    else:
        ids = ARXIV_IDS
        figure_urls = {}

    for arxiv_id in ids:
        fetch_and_report(arxiv_id)
