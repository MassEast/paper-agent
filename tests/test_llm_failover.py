"""_llm moves on to the next model at once when a model can't answer (400 too long, 404 no such model), retries
other errors, logs every failed attempt, and keeps to LLM_MAX_CONCURRENCY requests at a time."""
import threading
import time as real_time
from unittest.mock import MagicMock, patch

import httpx

from app import llm


def _ok(content="fine"):
    r = MagicMock()
    r.raise_for_status.return_value = None
    r.json.return_value = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                           "usage": {"total_tokens": 1}}
    return r


def _error(status):
    r = MagicMock()
    req = httpx.Request("POST", "http://llm.example/v1/chat/completions")
    r.raise_for_status.side_effect = httpx.HTTPStatusError(str(status), request=req,
                                                          response=httpx.Response(status, request=req))
    return r


def _run(by_model):
    sent = []

    def post(url, **kw):
        sent.append(kw["json"]["model"])
        return by_model[kw["json"]["model"]].pop(0)
    client = MagicMock()
    client.__enter__.return_value.post.side_effect = post
    with patch.object(llm, "MODELS", ["first", "second"]), patch("app.llm.httpx.Client", return_value=client), \
            patch("app.llm.time.sleep"):
        out = llm._llm([{"role": "user", "content": "x"}])
    return out, sent


def test_too_long_or_missing_model_moves_on_at_once(caplog):
    for status in (400, 404):
        (text, model, *_), sent = _run({"first": [_error(status)], "second": [_ok()]})
        assert (text, model, sent) == ("fine", "second", ["first", "second"])
    assert "first attempt 1 failed" in caplog.text


def test_overloaded_model_is_retried():
    (text, model, *_), sent = _run({"first": [_error(503), _error(503), _ok()], "second": []})
    assert (model, sent) == ("first", ["first"] * 3)


def test_requests_keep_to_the_concurrency_limit():
    in_flight, peak, lock = [0], [0], threading.Lock()

    def post(url, **kw):
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        real_time.sleep(0.05)
        with lock:
            in_flight[0] -= 1
        return _ok()
    client = MagicMock()
    client.__enter__.return_value.post.side_effect = post
    for nightly, expected in ((False, 3), (True, 1)):
        peak[0] = 0
        with patch.object(llm, "_MAX_CONCURRENCY", 3), patch.object(llm, "_MAX_DURING_NIGHTLY", 1), \
                patch("app.utils.nightly_running", return_value=nightly), \
                patch("app.llm.httpx.Client", return_value=client):
            threads = [threading.Thread(target=llm._llm, args=([{"role": "user", "content": "x"}],))
                       for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert peak[0] == expected  # the web app keeps to 1 while the nightly job runs


def test_nightly_marker_is_seen_by_other_processes_only(tmp_path, monkeypatch):
    from app import utils
    monkeypatch.setattr(utils, "NIGHTLY_MARKER", str(tmp_path / "nightly.running"))
    assert not utils.nightly_running()
    with utils.nightly_marker():
        assert not utils.nightly_running()  # the nightly job itself isn't slowed down
        monkeypatch.setattr(utils, "_holding_marker", False)  # as the web app sees the file
        assert utils.nightly_running()
        old = real_time.time() - utils.NIGHTLY_MARKER_MAX_AGE_S - 1
        import os
        os.utime(utils.NIGHTLY_MARKER, (old, old))
        assert not utils.nightly_running()  # left behind by a killed run: ignored
    assert not (tmp_path / "nightly.running").exists()


def test_lone_surrogates_in_pdf_text_still_encode():
    # pypdf text from 2601.05248 (2026-10-27) held a lone '\ud835', and the request body failed to encode
    sent = []

    def post(url, **kw):
        sent.append(kw["json"]["messages"][0]["content"])
        httpx.Request("POST", url, json=kw["json"])  # encodes the body as httpx does; raised before the fix
        return _ok()
    client = MagicMock()
    client.__enter__.return_value.post.side_effect = post
    with patch.object(llm, "MODELS", ["first"]), patch("app.llm.httpx.Client", return_value=client):
        llm._llm([{"role": "user", "content": "Univ\ud835 of 𝐀"}])
    assert sent == ["Univ� of \U0001d400"]
