"""Figure 1 cut from the arXiv PDF (app/figures.py, since 2026-11-04). Before, the card showed the first image file
of the arXiv HTML: for 2610.22075 that was Figure 2 (one photo of a grid of four), as its Figure 1 is drawn in TikZ
and so inline SVG in the HTML. The PDFs here are built with PyMuPDF in the layouts real papers use."""

import io
import os
from datetime import datetime
from unittest.mock import patch

import pymupdf
import pytest
from PIL import Image

from app import crawl, db, figures
from app.models import Paper, Project, ProjectPaper

PARAGRAPH = ("We study how models learn from data and report results on several benchmarks with careful "
             "ablations of every component and a discussion of the limits of the approach. ") * 2
CAPTION = "Figure 1: Overview of the method. The left panel shows the input, the right panel the out-\nput."


def _pdf(two_columns=False, hidden_latex_data=False, figure=True, caption=CAPTION, at_top=False, word_lines=False,
         title_page=False):
    """A letter page: running header and rule, a paragraph, Figure 1 (panel label above a box with labels in it),
    its caption, another paragraph. In two_columns the figure and caption are in the right column, with a
    paragraph in the left column at the same height. at_top: no paragraph above, the figure starts right below
    the running header (as in CLIP, 2103.00020). word_lines: the paragraph above ends in three lines that are a
    block each, the last one short, close above the figure (PDFs made in Word, 2610.35685). title_page: title and
    authors span both columns right above Figure 1 in the right column (2610.34626, 2610.39927)."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 40), "Preprint. Under review.", fontsize=9)
    page.draw_line((72, 48), (540, 48))
    x0, x1 = (316, 540) if two_columns else (72, 540)
    dy = -156 if at_top else 0  # the panel label then sits ~10 pt below the header rule
    if word_lines:
        for y, line in ((157, "and we used it to prepare the plotting code for the figures and to write analysis "
                                "code for the evaluation"),
                        (178, "statistics of all experiments in the sections on results and on the ablations "
                              "(see the Acknowledgments"),
                        (199, "section).")):
            page.insert_text((x0, y), line, fontsize=9)
    elif title_page:
        page.insert_text((100, 185), "Entity Matching using Decision Models", fontsize=16)
        page.insert_text((240, 205), "Ralph Peeters, University of Mannheim", fontsize=9)
    elif not at_top:
        page.insert_textbox(pymupdf.Rect(x0, 70, x1, 200), PARAGRAPH, fontsize=9)
    if two_columns:
        page.insert_textbox(pymupdf.Rect(72, 70, 296, 600), PARAGRAPH * 3, fontsize=9)
    if figure:
        page.insert_text((x0 + 10, 222 + dy), "(a) Input", fontsize=9)  # panel label above the drawing
        page.draw_rect(pymupdf.Rect(x0 + 10, 230 + dy, x1 - 10, 380 + dy), color=(0, 0, 0), fill=(0.8, 0.9, 1))
        page.insert_text((x0 + 30, 300 + dy), "encoder", fontsize=9)
        if hidden_latex_data:  # TikZ/latexit pictures carry base64 text inside the figure
            page.insert_textbox(pymupdf.Rect(x0 + 30, 320, x1 - 30, 370), "AB7XicbVDLSgNBEOz1GeMr6tHLYBA8hV0R9Rj" * 12,
                                fontsize=4)
    else:
        page.insert_text((x0 + 10, 300), "just a label", fontsize=9)
    page.insert_textbox(pymupdf.Rect(x0, 395 + dy, x1, 440 + dy), caption, fontsize=9)
    page.insert_textbox(pymupdf.Rect(x0, 450, x1, 600), PARAGRAPH, fontsize=9)
    return doc.tobytes()


def _height_pt(png):
    return Image.open(io.BytesIO(png)).height * 72 / figures.DPI


def _width_pt(png):
    return Image.open(io.BytesIO(png)).width * 72 / figures.DPI


def test_figure_1_is_the_drawing_with_its_labels_between_the_paragraph_and_the_caption():
    png, caption = figures.crop_figure_1(_pdf())
    assert png is not None
    # From the panel label (top ~213) to the box's bottom (380), plus a margin: not the header, not the paragraphs
    assert 165 < _height_pt(png) < 185
    assert caption == "Figure 1: Overview of the method. The left panel shows the input, the right panel the output."


def test_two_columns_take_only_the_captions_column():
    png, _ = figures.crop_figure_1(_pdf(two_columns=True))
    assert png is not None and _width_pt(png) < 240 and 165 < _height_pt(png) < 185


def test_a_figure_right_below_the_running_header_leaves_the_header_out():
    png, _ = figures.crop_figure_1(_pdf(at_top=True))
    assert png is not None and 165 < _height_pt(png) < 185


def test_lines_of_a_paragraph_that_are_a_block_each_are_left_out():
    png, _ = figures.crop_figure_1(_pdf(word_lines=True))
    assert png is not None and 165 < _height_pt(png) < 185


def test_title_and_authors_beside_the_column_are_left_out():
    png, _ = figures.crop_figure_1(_pdf(two_columns=True, title_page=True))
    assert png is not None and _width_pt(png) < 240 and 165 < _height_pt(png) < 185


def test_hidden_latex_data_in_a_figure_is_not_taken_for_a_paragraph():
    png, _ = figures.crop_figure_1(_pdf(hidden_latex_data=True))
    assert png is not None and _height_pt(png) > 165


def test_no_figure_1_caption_or_only_text_above_it_gives_none():
    assert figures.crop_figure_1(_pdf(caption="Figure 2: Something else.")) == (None, None)
    assert figures.crop_figure_1(_pdf(figure=False)) == (None, None)


def test_get_paper_figure_saves_the_cutout_and_falls_back_to_the_html(monkeypatch):
    monkeypatch.setattr(figures, "_download_pdf", lambda arxiv_id: _pdf())
    with patch.object(crawl, "_html_figure") as html_figure:
        url, caption = crawl.get_paper_figure("2610.22075")
    html_figure.assert_not_called()
    assert url == "/figures/2610.22075.png" and caption.startswith("Figure 1: Overview")
    assert open(os.path.join(figures.figures_dir(), "2610.22075.png"), "rb").read().startswith(b"\x89PNG")

    monkeypatch.setattr(figures, "_download_pdf", lambda arxiv_id: _pdf(caption="Figure 2: x."))
    with patch.object(crawl, "_html_figure", return_value=("https://arxiv.org/html/x/f.png", "Figure 2: x.")):
        assert crawl.get_paper_figure("2610.00001") == ("https://arxiv.org/html/x/f.png", "Figure 2: x.")


def test_the_card_and_the_email_get_the_saved_file(client, monkeypatch):
    monkeypatch.setattr(figures, "_download_pdf", lambda arxiv_id: _pdf())
    url, caption = figures.pdf_figure("2610.22076")
    assert client.get(url).status_code in (302, 401)  # behind the site login
    with client.session_transaction() as sess:
        sess["logged_in"] = True
    resp = client.get(url)
    assert resp.status_code == 200 and resp.mimetype == "image/png"
    assert client.get("/figures/..%2Fsecret.db").status_code == 404
    html = crawl._fetch_figure_html(url, caption)
    assert "data:image/png;base64," in html and "Overview of the method" in html


def test_nightly_backfill_redoes_older_figures_once_and_skips_trashed_papers(app, monkeypatch):
    monkeypatch.setattr(figures, "_download_pdf",
                        lambda arxiv_id: _pdf() if arxiv_id == "2609.00001" else _pdf(caption="Figure 3: x."))
    with app.app_context():
        project = Project(name="Fig Proj", slug="fig-proj")
        db.session.add(project)
        db.session.flush()
        papers = {}
        for aid, trashed in (("2609.00001", False), ("2609.00002", False), ("2609.00003", True)):
            p = Paper(arxiv_id=aid, title=aid, source="arxiv", figure_url="https://arxiv.org/html/x/old.png")
            db.session.add(p)
            db.session.flush()
            papers[aid] = p
            db.session.add(ProjectPaper(project_id=project.id, paper_id=p.id,
                                        trashed_at=datetime(2026, 11, 1) if trashed else None))
        db.session.commit()
        try:
            assert crawl.redo_pdf_figures() == 1
            assert papers["2609.00001"].figure_url == "/figures/2609.00001.png"
            assert papers["2609.00002"].figure_url == "https://arxiv.org/html/x/old.png"  # no Figure 1: kept
            assert papers["2609.00002"].figure_pdf_tried_at is not None
            assert papers["2609.00003"].figure_pdf_tried_at is None  # only in Trash: not fetched
            assert crawl.redo_pdf_figures() == 0  # each paper tried once
        finally:
            ProjectPaper.query.filter_by(project_id=project.id).delete()
            Paper.query.filter(Paper.id.in_([p.id for p in papers.values()])).delete()
            db.session.delete(project)
            db.session.commit()


# --- Live (slow, hit arXiv): pytest tests/test_figures.py -v -m slow ---
_REAL_DOWNLOAD = figures._download_pdf  # conftest stubs it out for every test


@pytest.mark.slow
def test_live_figure_1_of_a_tikz_figure_comes_from_the_pdf(monkeypatch):
    monkeypatch.setattr(figures, "_download_pdf", _REAL_DOWNLOAD)
    url, caption = crawl.get_paper_figure("2610.22075")
    assert url == "/figures/2610.22075.png" and caption.startswith("Figure 1: Prune-then-loop retrofit")


@pytest.mark.slow
class TestHtmlFigureFallback:
    def test_paper_with_html_version_returns_figure(self):
        """A paper with an arXiv HTML version should return a real figure URL."""
        # Attention Is All You Need — has HTML and a well-known Figure 1
        url, caption = crawl._html_figure("1706.03762")
        assert url is not None, "Expected a figure URL"
        assert url.startswith("http"), f"Expected absolute URL, got {url!r}"
        assert "static/base" not in url, f"Got a UI-chrome image instead of a paper figure: {url!r}"
        assert "funder" not in url.lower(), f"Got a funder logo instead of a paper figure: {url!r}"

    def test_no_html_version_does_not_return_funder_logo(self):
        """2605.09403 has no arXiv HTML version; fallback abs page only has UI/funder images.
        We should return (None, None) rather than a funder/chrome image."""
        url, caption = crawl._html_figure("2605.09403")
        if url is not None:
            assert "funder" not in url.lower(), f"Returned a funder logo as figure: {url!r}"
            assert "static/base" not in url, f"Returned a UI-chrome image as figure: {url!r}"
            assert "static/browse" not in url, f"Returned a browser-UI image as figure: {url!r}"
            assert url.startswith("http"), f"Expected absolute URL, got {url!r}"
