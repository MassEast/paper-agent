"""Light theme by default, dark on request: the nav switch stores the choice per browser (localStorage 'theme'),
and a script in <head> applies it before the first paint (since 2026-11-03)."""

import re


def test_every_page_starts_light_unless_this_browser_chose_dark(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    html = client.get("/projects").get_data(as_text=True)
    head = html.split("</head>", 1)[0]
    script = head.index("localStorage.getItem('theme') !== 'dark'")
    assert script < head.index("tokens.css")  # set before the colors load, so no flash of the other theme
    assert 'id="theme-toggle"' in html and "toggleTheme()" in html


def test_both_themes_define_the_same_color_tokens():
    css = open("app/static/tokens.css").read()
    dark = css.split(":root {", 1)[1].split("}", 1)[0]
    light = css.split(':root[data-theme="light"] {', 1)[1].split("}", 1)[0]
    colors = lambda block: {m for m in re.findall(r"(--[\w-]+):\s*(?:oklch|var)", block)}
    missing = {t for t in colors(dark) - colors(light)
               if not t.startswith("--tag-discuss")}  # these point at tokens that switch
    missing.discard("--accent-ink")  # white text on the accent fill, the same in both themes
    assert missing == set(), missing
