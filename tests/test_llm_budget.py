"""_llm doubles max_tokens when a completion is cut off, instead of repeating the same request.

2026-10-07/08 nightly: bht/large reasoned past the cap (featured pick 1024, summary 2048,
institutions 4096) and the same-budget retries at temperature 0 returned the identical cut-off.
"""
from unittest.mock import MagicMock, patch

import pytest

from app import llm


def _response(content, finish_reason):
    r = MagicMock()
    r.json.return_value = {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
                           "usage": {"total_tokens": 1}}
    return r


def _run(responses, sent=None):
    sent = [] if sent is None else sent
    client = MagicMock()
    client.__enter__.return_value.post.side_effect = lambda url, **kw: (sent.append(kw["json"]["max_tokens"]),
                                                                       responses.pop(0))[1]
    with patch("app.llm.httpx.Client", return_value=client), patch("app.llm.time.sleep"):
        out = llm._llm([{"role": "user", "content": "x"}], max_tokens=1024)
    return out, sent


def test_think_only_cutoff_doubles_budget():
    (text, *_), sent = _run([_response("<think>long reasoning", "length"),
                             _response("<think>more reasoning", "length"),
                             _response("<think>done</think>[\"MIT\"]", "stop")])
    assert text == '["MIT"]'
    assert sent == [1024, 2048, 4096]


def test_truncated_answer_also_doubles_budget():
    (text, *_), sent = _run([_response('<think>ok</think>{"selected":', "length"),
                             _response('<think>ok</think>{"selected": 2}', "stop")])
    assert text == '{"selected": 2}'
    assert sent == [1024, 2048]


def test_budget_stops_at_ceiling():
    sent = []
    responses = [_response("<think>loop", "length") for _ in range(10)]
    with patch.object(llm, "MODELS", ["only-model"]), pytest.raises(llm.LLMUnavailableError):
        _run(responses, sent)
    # doublings count against the model's 5 attempts; the last one runs at the ceiling
    assert sent == [1024, 2048, 4096, 8192, 16384]


@pytest.mark.slow
def test_live_cutoff_recovers_by_doubling(caplog):
    """Real endpoint: a budget far too small for bht/large's reasoning is grown until it answers."""
    from app import prompts
    prompt = prompts.INSTITUTION_EXTRACTION.format(
        source_label="first page of the PDF", title="Homogenization in Multi-Agent Systems",
        text="Prakhar Ganesh1, Kyra Wilson2, Luca Zappella3\n1McGill University & Mila, "
             "2University of Washington, 3Apple\nAbstract: Multi-agent systems ...")
    with caplog.at_level("WARNING", logger="app.llm"):
        text, model, _, _ = llm._llm([{"role": "user", "content": prompt}], temperature=0, max_tokens=256)
    assert "Apple" in text
    assert any("hit max_tokens 256" in r.message for r in caplog.records)
