"""What the model reads of a paper (2026-11-12): the whole text up to utils.FULL_TEXT_CHARS (it was cut at 80,000
characters, so about half of all papers were read only in part), without the reference list (drop_references), and
only by the first model of the cascade, the one whose context holds it (first_model_only)."""
from unittest.mock import patch

import httpx

from app import llm, utils

BODY = "We study looped transformers. " * 300 + "5 Conclusion We showed that loops help. Acknowledgments We thank X. "
REFS = "".join(f"A{i}. Author and B. Author. A paper on topic {i}. In ICML, {2015 + i % 10}. " for i in range(60))


def test_the_reference_list_is_left_out_and_the_appendix_after_it_stays():
    # arXiv HTML: back-links in the references ("Appendix E .") are no headings; "Appendix A Details" is one
    refs = REFS[:2000] + "Gu and Dao (2024) Mamba. Appendix E , §7 . " + REFS[2000:]
    appendix = "Appendix A Details of the setup We train for 3 epochs (Vaswani et al., 2017). " + "More detail. " * 300
    out = utils.drop_references(BODY + "References " + refs + appendix)
    assert out.startswith(BODY) and "We thank X." in out  # the conclusion before the references stays
    assert "A paper on topic" not in out and "Mamba" not in out
    assert out.endswith(appendix) and "[References left out]" in out


def test_without_an_appendix_heading_the_references_end_where_the_years_stop():
    # PDF text: "A Training Details" has no "Appendix" word; the long stretch without a year marks it
    appendix = "A Training Details In this section we give the hyperparameters. " + "We use a batch of 64. " * 120
    out = utils.drop_references(BODY + "References " + REFS + appendix)
    assert "A paper on topic" not in out and out.endswith(appendix)
    out = utils.drop_references(BODY + "References " + REFS)  # references at the very end
    assert out == BODY + "References "[:0] + " [References left out] "


def test_nothing_is_cut_when_nothing_looks_like_a_reference_list():
    toc = "Contents 1 Introduction 2 Method References A Extended related work B Details "
    text = toc + BODY * 2 + "References to prior work are in Section 2. " + "Prose without years. " * 200
    assert utils.drop_references(text) == text
    assert utils.drop_references(BODY) == BODY


def test_first_model_only_skips_the_rest_of_the_cascade(monkeypatch):
    monkeypatch.setattr(llm, "MODELS", ["large", "medium"])
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    asked = []

    def post(self, url, headers=None, json=None):
        asked.append(json["model"])
        raise httpx.ConnectError("down")

    with patch.object(httpx.Client, "post", post):
        try:
            llm._llm([{"role": "user", "content": "x"}], first_model_only=True)
        except llm.LLMUnavailableError:
            pass
    assert set(asked) == {"large"}
