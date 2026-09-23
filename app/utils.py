import re
import httpx


def extract_arxiv_id(text: str) -> str | None:
    patterns = [
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?",
        r"(?:^|[\s/])([0-9]{4}\.[0-9]{4,5})(?:v\d+)?(?:$|[\s\.])",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def fetch_arxiv_title(arxiv_id: str) -> str | None:
    try:
        with httpx.Client(timeout=10.0) as http:
            r = http.get(
                "https://export.arxiv.org/api/query",
                params={"id_list": arxiv_id, "max_results": 1},
            )
            r.raise_for_status()
            candidates = re.findall(r"<title>([^<]+)</title>", r.text)
            return candidates[1].strip() if len(candidates) > 1 else None
    except Exception:
        pass
    return None


def fetch_paper_full_text(arxiv_id: str, max_chars: int = 80000) -> str | None:
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as http:
            r = http.get(f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}")
            if r.status_code != 200:
                return None
            html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", r.text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", text).strip()[:max_chars]
    except Exception as e:
        print(f"ar5iv fetch failed for {arxiv_id}: {e}")
    return None


def fetch_paper_figure(arxiv_id: str) -> str | None:
    try:
        resp = httpx.get(f"https://arxiv.org/abs/{arxiv_id}", timeout=15.0)
        for src in re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', resp.text, re.IGNORECASE):
            if any(x in src.lower() for x in ['figure', 'fig', '.png', '.jpg']) or src.startswith('http'):
                return ('https:' + src if src.startswith('//') else
                        'https://arxiv.org' + src if src.startswith('/') else src)
    except Exception:
        pass
    return None