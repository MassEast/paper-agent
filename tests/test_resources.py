"""What a paper says about itself (app/resources.py, card redesign 2026-11-12): its own code repository and project
page, the corresponding authors' emails, the venue. Regex finds the candidates; the model only picks among them, and
anything it returns that the text doesn't back is dropped. The texts here are shaped like real arXiv HTML/PDF text."""
import json
from unittest.mock import patch

import pytest

from app import resources, utils

AUTHORS = ["Nika Zahedi", "Fang-Yi Zhu", "Anna Wu"]
TEXT = (
    "MicroLens: Read, Trace, and Rewire the Residual Stream with Two Lenses Nika Zahedi* Fang-Yi Zhu Anna Wu "
    "Stanford University *Correspondence to: nzahedi@stanford.edu {fzhu, awu}@cs.stanford.edu "
    "Published as a conference paper at ICLR 2026 Abstract We present MicroLens. Code: "
    "https://github.com/nzahedi/microlens. Project page: https://microlens.github.io/ 1 Introduction "
    "We build on TransformerLens (github.com/neelnanda-io/TransformerLens) and the Tuned Lens. "
    + "Body text about lenses. " * 400 +
    " References [1] Belrose et al. Eliciting latent predictions. https://github.com/AlignmentResearch/tuned-lens "
    "[2] contact other@elsewhere.org"
)


def _answer(payload):
    return payload, "m", 1, {"total_tokens": 321}


# ── Candidates ────────────────────────────────────────────────────────────────

def test_the_text_after_the_references_heading_is_left_out():
    body = resources.body_text(TEXT)
    assert "Body text" in body and "tuned-lens" not in body and "other@elsewhere.org" not in body


def test_references_mentioned_early_do_not_cut_the_text():
    text = "Intro. See References for details. " + "More text. " * 200
    assert resources.body_text(text) == text


def test_url_candidates_are_cleaned_deduped_and_get_their_context():
    urls = resources.candidate_urls([
        "Code at https://github.com/a/b.git, and https://github.com/a/b/. Site: x-lab.github.io/proj). "
        "DOI https://doi.org/10.1/x and arXiv https://arxiv.org/abs/2401.00001 and CC https://creativecommons.org/l"])
    assert [u for u, _ in urls] == ["https://github.com/a/b", "https://x-lab.github.io/proj"]
    assert "Code at" in urls[0][1]


def test_bare_repository_links_get_https():
    urls = resources.candidate_urls(["available at github.com/org/repo and at huggingface.co/org/model."])
    assert [u for u, _ in urls] == ["https://github.com/org/repo", "https://huggingface.co/org/model"]


def test_emails_grouped_in_braces_are_split():
    emails = [e for e, _ in resources.candidate_emails(
        "{alice, bob.c}@uni.edu and (dave|eve)@lab.org, Frank@Corp.COM.")]
    assert emails == ["alice@uni.edu", "bob.c@uni.edu", "dave@lab.org", "eve@lab.org", "frank@corp.com"]


def test_no_link_no_email_no_venue_means_no_model_call():
    with patch.object(resources, "_llm_json", side_effect=AssertionError("must not call the model")):
        assert resources.extract("T", AUTHORS, "Plain text without anything. " * 50) == (resources.EMPTY, 0)
        assert resources.extract("T", AUTHORS, None, abstract="An abstract.") == (resources.EMPTY, 0)


def test_a_comment_that_names_a_venue_is_reason_enough_to_ask():
    with patch.object(resources, "_llm_json", return_value=_answer({"venue": "NeurIPS 2026"})) as llm:
        found, tokens = resources.extract("T", AUTHORS, None, comment="Accepted at NeurIPS 2026")
    assert llm.called and found["venue"] == "NeurIPS 2026" and tokens == 321


# ── The model's answer is checked against the text ───────────────────────────

def test_the_models_picks_are_kept_when_the_text_backs_them():
    answer = {"code_url": "https://github.com/nzahedi/microlens", "project_url": "https://microlens.github.io",
              "corresponding": [{"author": "Nika Zahedi", "email": "nzahedi@stanford.edu"}], "venue": "ICLR 2026"}
    with patch.object(resources, "_llm_json", return_value=_answer(answer)) as llm:
        found, _ = resources.extract("MicroLens", AUTHORS, TEXT)
    prompt = llm.call_args.args[0][0]["content"]
    assert "https://github.com/nzahedi/microlens" in prompt and "fzhu@cs.stanford.edu" in prompt
    assert "tuned-lens" not in prompt and "other@elsewhere.org" not in prompt  # from the references
    assert found == {"code_url": "https://github.com/nzahedi/microlens", "project_url": "https://microlens.github.io",
                     "corresponding": [{"name": "Nika Zahedi", "email": "nzahedi@stanford.edu"}], "venue": "ICLR 2026"}


def test_made_up_links_emails_authors_and_venues_are_dropped():
    answer = {"code_url": "https://github.com/someone/else", "project_url": "https://github.com/nzahedi/microlens",
              "corresponding": [{"author": "Nika Zahedi", "email": "nika@gmail.com"},
                                {"author": "Somebody Else", "email": "awu@cs.stanford.edu"},
                                {"author": "anna  wu", "email": "AWU@cs.stanford.edu"}],
              "venue": "CVPR 2026"}
    with patch.object(resources, "_llm_json", return_value=_answer(answer)):
        found, _ = resources.extract("MicroLens", AUTHORS, TEXT)
    assert found["code_url"] is None  # not in the text
    assert found["project_url"] == "https://github.com/nzahedi/microlens"  # in the text (the model's call)
    assert found["corresponding"] == [{"name": "Anna Wu", "email": "awu@cs.stanford.edu"}]  # name as on the paper
    assert found["venue"] is None  # the text never says CVPR


def test_the_same_link_is_not_both_code_and_project_page():
    url = "https://github.com/nzahedi/microlens"
    found = resources.validate({"code_url": url, "project_url": url}, [url], [], AUTHORS, "")
    assert found["code_url"] == url and found["project_url"] is None


def test_generic_or_null_venues_are_dropped():
    src = "Accepted at a workshop of the conference"
    for venue in ("null", "Workshop", "Conference Proceedings", "x" * 81):
        assert resources.validate({"venue": venue}, [], [], AUTHORS, src + " " + venue)["venue"] is None
    assert resources.validate({"venue": "TMLR"}, [], [], AUTHORS, "Published in TMLR (01/2026)")["venue"] == "TMLR"


def test_an_answer_that_is_not_json_leaves_the_card_without_buttons():
    with patch.object(resources, "_llm_json", side_effect=ValueError("no JSON")):
        assert resources.extract("MicroLens", AUTHORS, TEXT) == (resources.EMPTY, 0)


def test_the_model_being_down_is_raised():
    from app.llm import LLMUnavailableError
    with patch.object(resources, "_llm_json", side_effect=LLMUnavailableError("down")), \
            pytest.raises(LLMUnavailableError):
        resources.extract("MicroLens", AUTHORS, TEXT)


def test_at_most_three_corresponding_authors():
    names = ["Ann Lee", "Bo Chen", "Cy Diaz", "Di Eng", "Ed Fox"]
    emails = [f"{n.split()[1].lower()}@x.org" for n in names]
    authors = names
    raw = {"corresponding": [{"author": a, "email": e} for a, e in zip(authors, emails)]}
    assert len(resources.validate(raw, [], emails, authors, "")["corresponding"]) == 3


def test_an_email_must_fit_the_authors_name():
    fits = [("swang08@villanova.edu", "Shaohuang Wang"), ("jcl@microsoft.com", "John Langford"),
            ("bsmn0223@kaist.ac.kr", "Sangmin Bae"), ("nora@eleuther.ai", "Nora Belrose"), ("bo@x.org", "Bo Chen"),
            ("jack.parley@epfl.ch", "Jack T. Parley"), ("caosw25@mails.tsinghua.edu.cn", "Shaowei Cao")]
    for email, name in fits:
        assert resources._email_fits(email, name), (email, name)
    for email, name in [("caosw25@mails.tsinghua.edu.cn", "Haoqian Wang"), ("contact@lab.org", "Ann Lee"),
                        ("x7k2@uni.edu", "Bo Chen")]:
        assert not resources._email_fits(email, name), (email, name)


def test_an_email_given_to_the_wrong_author_is_dropped():
    # 2611.13732: the model gave the only email, a co-author's, to the corresponding author
    raw = {"corresponding": [{"author": "Haoqian Wang", "email": "caosw25@mails.tsinghua.edu.cn"}]}
    out = resources.validate(raw, [], ["caosw25@mails.tsinghua.edu.cn"], ["Shaowei Cao", "Haoqian Wang"], "")
    assert out["corresponding"] == []


def test_a_comment_naming_a_venue_and_year_alone_is_reason_enough_to_ask():
    # 2410.20672: "ICLR 2025; 49 pages, 17 figures, 19 tables"
    with patch.object(resources, "_llm_json", return_value=_answer({"venue": "ICLR 2025"})) as llm:
        found, _ = resources.extract("T", AUTHORS, "Plain text. " * 50, comment="ICLR 2025; 49 pages, 17 figures")
    assert llm.called and found["venue"] == "ICLR 2025"
    with patch.object(resources, "_llm_json", side_effect=AssertionError("no venue named")):
        resources.extract("T", AUTHORS, "Plain text. " * 50, comment="LLNL-CONF-2026172; 9 pages, 4 figures")


def test_apply_stores_the_findings(app):
    from app.models import Paper
    p = Paper(arxiv_id="x", title="t")
    resources.apply(p, {"code_url": "https://github.com/a/b", "project_url": None,
                        "corresponding": [{"name": "A", "email": "a@x.org"}], "venue": "ICLR 2026"})
    assert p.code_url == "https://github.com/a/b" and p.venue == "ICLR 2026" and p.resources_checked_at
    assert json.loads(p.corresponding_authors) == [{"name": "A", "email": "a@x.org"}]
    assert p.corresponding_emails == {"A": "a@x.org"}
    resources.apply(p, dict(resources.EMPTY))
    assert p.corresponding_authors is None and p.corresponding_emails == {}


# ── Full text keeps hidden link targets ──────────────────────────────────────

def test_arxiv_html_keeps_the_target_of_a_link_whose_text_hides_it():
    html = ('<article><p>Our <a href="https://github.com/a/b" class="ltx_href">code</a> and '
            '<a href="https://github.com/a/b" class="ltx_ref ltx_url">https://github.com/a/b</a>; '
            'see <a href="#bib.bib3">[3]</a>, <a href="https://arxiv.org/abs/2401.1">arXiv</a>.</p>'
            '<li class="ltx_bibitem"><a href="https://github.com/other/x" class="ltx_ref ltx_bib_external">Link</a></li>'
            '<p>Q&amp;A</p></article>')
    text = utils._html_to_text(html)
    assert "code (https://github.com/a/b)" in text
    assert text.count("https://github.com/a/b") == 2  # the visible URL isn't repeated
    assert "other/x" not in text and "abs/2401.1" not in text  # bibliography / arXiv links
    assert "Q&A" in text


def test_pdf_link_annotations_are_added_when_the_page_text_hides_them():
    pymupdf = pytest.importorskip("pymupdf")
    import io
    from pypdf import PdfReader
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Code and Project page")
    page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(72, 60, 100, 75), "uri": "https://github.com/a/b"})
    page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(110, 60, 160, 75), "uri": "https://doi.org/10.1/x"})
    reader = PdfReader(io.BytesIO(doc.tobytes()))
    pg = reader.pages[0]
    assert utils._pdf_page_links(pg, pg.extract_text()) == " (https://github.com/a/b)"
    assert utils._pdf_page_links(pg, "see github.com/a/b") == ""  # already in the text


# --- Live (slow): real arXiv papers through the real model. pytest tests/test_resources.py -m slow ---
# Real papers with what a reader finds in them (checked by hand, 2026-11-12). Keys: code / project (the end of the
# URL, lower case, or None: nothing may be found), venue (a part of it, or None), corresponding (exactly these),
# corresponding_has (at least these), not_corresponding (names that must not get an email). A key left out isn't checked.
LIVE = [
    # Tuned Lens: code in the abstract, "Correspondence to" in the author block, many other repos cited
    ("2303.08112", {"code": "github.com/alignmentresearch/tuned-lens", "project": None,
                    "corresponding": {"Nora Belrose": "nora@eleuther.ai"}}),
    # code (a tag of the repo) only in the arXiv comment
    ("2611.13322", {"code": "github.com/ni2002ka/residual_stream_vis/tree/v1.0-pacificvis", "project": None}),
    # project page and workshop in the comment, no code, no emails
    ("2611.13649", {"code": None, "project": "he2sc-st.github.io", "venue": "NeurIPS 2026", "corresponding": {}}),
    # two addresses for the first author: the institutional one is the corresponding one
    ("2611.14221", {"code": "github.com/jackparley/interpretable_inside_outside",
                    "corresponding_has": {"Jack T. Parley": "jack.parley@epfl.ch"}}),
    # code in the comment, five emails, funding sites linked in the acknowledgements
    ("2507.10524", {"code": "github.com/raymin0223/mixture_of_recursions", "project": None,
                    "corresponding_has": {"Sangmin Bae": "bsmn0223@kaist.ac.kr"}}),
    # venue only in the comment ("ICLR 2025; 49 pages, ..."), nothing else to find
    ("2410.20672", {"code": None, "project": None, "venue": "ICLR 2025", "corresponding": {}}),
    # the only email belongs to a student, not to the last author (once misattributed to him)
    ("2611.13732", {"code": None, "not_corresponding": ["Haoqian Wang"]}),
    # own repo next to blog posts and a model on Hugging Face
    ("2608.08888", {"code": "github.com/xidulu/full-bandwidth-transformer", "project": None}),
    # own repo next to a benchmark repo it uses; a gmail address
    ("2611.11531", {"code": "github.com/ambroiseodt/waldo",
                    "corresponding_has": {"Ambroise Odonnat": "ambroise.odonnat@gmail.com"}}),
    # nothing in the paper: no candidates, so no model call either
    ("2510.24824", {"code": None, "project": None, "venue": None, "corresponding": {}, "tokens": 0}),
]


def _url_ok(found, want):
    return found is None if want is None else (found or "").lower().rstrip("/").endswith(want)


@pytest.mark.slow
@pytest.mark.parametrize("arxiv_id,want", LIVE, ids=[a for a, _ in LIVE])
def test_live_links_and_emails_of_real_papers(arxiv_id, want):
    import arxiv as arxiv_lib
    from app import arxiv_meta, crawl
    with utils._ArxivLockCtx():
        r = next(arxiv_lib.Client(delay_seconds=3.1).results(arxiv_lib.Search(id_list=[arxiv_id])))
    meta = arxiv_meta.from_result(r)
    text = crawl.get_paper_content(arxiv_id)
    assert text, "no full text: the test would only check the abstract"
    found, tokens = resources.extract(r.title, [str(a) for a in r.authors], text, r.summary,
                                      meta["arxiv_comment"], meta["journal_ref"])
    print(arxiv_id, found, tokens, flush=True)
    emails = {c["name"]: c["email"] for c in found["corresponding"]}
    if "code" in want:
        assert _url_ok(found["code_url"], want["code"])
    if "project" in want:
        assert _url_ok(found["project_url"], want["project"])
    if "venue" in want:
        assert found["venue"] is None if want["venue"] is None else want["venue"] in (found["venue"] or "")
    if "corresponding" in want:
        assert emails == want["corresponding"]
    for name, email in want.get("corresponding_has", {}).items():
        assert emails.get(name) == email
    for name in want.get("not_corresponding", []):
        assert name not in emails
    if "tokens" in want:
        assert tokens == want["tokens"]
