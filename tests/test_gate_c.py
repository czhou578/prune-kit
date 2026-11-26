"""Gate C, the part with no server: the new scorer sends the requests of the scorer of the Qwen prune, byte for byte.

``scripts/gate_c.sh`` is the part with the live server. The score of a turn moves between two runs, so equal requests
are the proof that the two scorers measure the same thing.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from prune_moe_kit import capture, replay, sessions

RECORD = Path.home() / "Projects/dsv4-specialize"
BOT = Path.home() / "Projects/stock-research-model-eval/model_eval/replay.py"   # the old scorer imports its post()
NEEDS = (RECORD / "tools/qwen_gate.py", RECORD / "results/qwen-recorded.jsonl",
         RECORD / "data/qwen-data/tools.json", BOT)


def stub(log):
    def post(base, path, body, timeout=3600):
        log.append((path, json.dumps(body)))
        info = {"logprob": -1.0, "rank": 1, "decoded_token": "x"}
        if path == "/tokenize":   # the token of the reasoning end, or the three tokens of a context
            return {"tokens": [9] if "prompt" in body else [1, 2, 3]}
        return {"prompt_logprobs": [None] * 3 + [{"5": info}]}
    return post


@pytest.mark.skipif(not all(p.exists() for p in NEEDS), reason="the record of the Qwen prune is not on this machine")
def test_the_new_scorer_sends_the_requests_of_the_scorer_of_the_record(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("qwen_gate", NEEDS[0])
    old = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "path", list(sys.path))   # the old scorer adds the bot repo to the path
    spec.loader.exec_module(old)
    data = sessions.load(RECORD / "data/qwen-data/sessions.jsonl")
    tools, recorded = json.loads(NEEDS[2].read_text())["tools"], sessions.load(NEEDS[1])
    a, b = [], []
    monkeypatch.setattr(old, "post", stub(a))
    monkeypatch.setattr(capture, "post", stub(b))
    old.score("http://x", "m", data, tools, recorded, tmp_path / "old.jsonl")
    assert replay.score("http://x", "m", data, tools, recorded, tmp_path / "new.jsonl") == 0
    assert len(a) == 2 * len(recorded) + 1 and a == b
    assert sessions.load(tmp_path / "old.jsonl")[0].keys() == sessions.load(tmp_path / "new.jsonl")[0].keys()
    assert [tuple(replay.key(r)) for r in recorded] == replay.pick(data, len(recorded))   # the same 60 targets
