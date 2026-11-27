"""
A blog post added by URL (LessWrong, 2026-11-27) puts 1.2 MB of scripts before the post. The page must be read
whole, links must keep their targets (so the code repo is found), and with no text there is no summary rather
than one made up from the title.
"""

import httpx

from app import crawl

URL = "https://www.lesswrong.com/posts/abc/some-post"


def _page(body: str) -> str:
    return ("<html><head><title>Some Post</title>"
            '<meta name="citation_title" content="Some Post: A Lens">'
            f"<script>{'x' * 1_300_000}</script></head><body>{body}</body></html>")


def _extract(monkeypatch, html_text):
    monkeypatch.setattr(crawl, "_is_safe_external_url", lambda url: True)
    real = crawl._http_client
    handler = lambda request: httpx.Response(200, text=html_text)  # noqa: E731
    monkeypatch.setattr(crawl, "_http_client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return crawl.extract_paper_info_from_url(URL)


def test_post_after_long_scripts_is_read_with_link_targets(monkeypatch):
    post = ("<article><p>TL;DR: we filter noisy gradients &amp; read workspaces. We release "
            '<a href="https://github.com/org/repo">code</a> and a '
            '<a href="https://example.org/demo">demo</a>.</p>' + "<p>More text.</p>" * 300 + "</article>")
    info = _extract(monkeypatch, _page(post))
    assert info["title"] == "Some Post: A Lens"
    text = crawl.page_text_for_summary(info)
    assert text.startswith("TL;DR: we filter noisy gradients & read workspaces.")
    assert "code (https://github.com/org/repo)" in text
    assert "xxxx" not in text  # scripts are not text


def test_page_without_text_gives_no_summary_source(monkeypatch):
    info = _extract(monkeypatch, _page("<p>This website requires javascript.</p>"))
    assert info["full_text"]  # a few words
    assert crawl.page_text_for_summary(info) == ""  # too little: no summary at all
    assert crawl.page_text_for_summary({**info, "abstract": "An abstract."}) == "An abstract."
