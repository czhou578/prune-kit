import base64
import json
import struct

import pytest

from prune_kit import capture, sessions


def npy(rows, descr="<i4", code="i"):
    """A base64 ``.npy`` array (version 1.0) of ``rows[token][layer][slot]``."""
    shape = (len(rows), len(rows[0]), len(rows[0][0]))
    head = repr({"descr": descr, "fortran_order": False, "shape": shape}).encode()
    head += b" " * (-(10 + len(head) + 1) % 64) + b"\n"
    flat = [e for token in rows for layer in token for e in layer]
    return base64.b64encode(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(head)) + head
                            + struct.pack(f"<{len(flat)}{code}", *flat)).decode()


def test_decode_reads_the_shape_and_the_values():
    values, shape = capture.decode(npy([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]]))
    assert shape == (3, 2, 2) and list(values) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]
    assert list(capture.decode(npy([[[300, 2]]], "<i8", "q"))[0]) == [300, 2]
    with pytest.raises(capture.CaptureError, match="not a .npy array"):
        capture.decode(base64.b64encode(b"junk junk junk").decode())
    with pytest.raises(capture.CaptureError, match="the type <f4"):
        capture.decode(npy([[[1, 2]]], "<f4", "f"))


def test_count_skips_the_rows_before_start_and_the_unfilled_rows():
    rows = [[[0, 1], [2, 3]],      # token 0
            [[1, 2], [0, 0]],      # token 1: layer 1 is not filled
            [[3, 1], [2, 0]]]      # token 2
    values, shape = capture.decode(npy(rows))
    assert capture.count(values, shape, experts=4) == ([[1, 3, 1, 1], [1, 0, 2, 1]], 1)
    assert capture.count(values, shape, start=1, experts=4) == ([[0, 2, 1, 1], [1, 0, 1, 0]], 1)
    assert capture.count(values, shape, start=3, experts=2) == ([[0, 0], [0, 0]], 0)


class Server:
    """A model server for the tests. A prompt is 4 prefix tokens, then one token for each message after the system
    prompt. Each token routes to the experts (token id % 4, 5) in its one layer."""

    def __init__(self, capture_flag=True):
        self.bodies, self.capture_flag = [], capture_flag

    def __call__(self, base, path, body):
        self.bodies.append((path, body))
        if path == "/tokenize":
            return {"tokens": [100, 101, 102, 103, 999]}
        if any(m.get("content") == "fail" for m in body["messages"]):
            raise OSError("the server sent 500")
        prompt = [100, 101, 102, 103] + [len(json.dumps(m)) for m in body["messages"] if m["role"] != "system"]
        tokens = prompt + list(range(body["max_tokens"]))
        choice = {"routed_experts": npy([[[t % 4, 5]] for t in tokens])} if self.capture_flag else {}
        return {"choices": [choice], "prompt_token_ids": prompt, "usage": {"completion_tokens": body["max_tokens"]}}


def data():
    seg = lambda *texts: [m for t in texts for m in ({"role": "user", "content": t},
                                                     {"role": "assistant", "content": "ok", "target": True})]
    return [sessions.session("a", "app", "claude_code", [seg("one"), seg("two", "three") + [{"role": "tool"}]],
                             False, "SYSTEM"),
            sessions.session("b", "app", "claude_code", [seg("fail")], False),
            sessions.session("h", "app", "claude_code", [seg("held")], True)]


def test_run_counts_the_prefix_once_and_continues_after_a_failure(tmp_path):
    server, out, log = Server(), tmp_path / "c/capture.jsonl", []
    assert capture.run("http://x", "m", data(), [{"type": "function"}], out, 6, post=server, log=log.append) == 1
    rows = [json.loads(line) for line in out.open()]
    assert [(r["session"], r["segment"], r["index"]) for r in rows] == [("a", 0, 1), ("a", 1, 3), ("b", 0, 1)]
    first, second, failed = rows
    assert (first["prompt_tokens"], first["prefix_tokens"], first["prefix_counted"]) == (6, 4, True)
    assert sum(first["counts"][0]) == 12 and first["counts"][0][5] == 6       # 6 tokens, 2 experts each
    assert second["prefix_counted"] is False and sum(second["counts"][0]) == 8  # 8 tokens, without the 4 of the prefix
    assert "OSError: the server sent 500" in failed["error"] and "counts" not in failed and len(log) == 3
    chat = server.bodies[0][1]
    assert chat["messages"][0] == {"role": "system", "content": "SYSTEM"} and chat["max_tokens"] == 1
    assert chat["messages"][-1] == {"role": "assistant", "content": "ok", "reasoning": ""} and chat["return_token_ids"]
    assert server.bodies[1][1]["messages"][1] == {"role": "user", "content": "."}
    # A second start does the failed segment only.
    data()[1]["segments"][0][0]["content"] = "fail"
    assert capture.run("http://x", "m", data(), None, out, 6, post=server, log=log.append) == 1
    assert len(out.read_text().splitlines()) == 4 and len(log) == 4


def test_generate_counts_the_new_tokens_and_the_extra_keys_reach_the_server(tmp_path):
    server = Server()
    extra = {"reasoning_effort": "medium"}
    capture.run("http://x", "m", data()[:1], None, tmp_path / "c.jsonl", 6, limit=2, extra=extra, generate=3,
                post=server, log=lambda line: None)
    first, second = (json.loads(line) for line in (tmp_path / "c.jsonl").open())
    assert first["gen_tokens"] == 3 and sum(first["gen_counts"][0]) == 2 * 3   # the context is a part of the prompt
    # The second segment has a tool message after its last target. That target is context now, and its token differs.
    assert sum(second["gen_counts"][0]) == 2 * (2 + 3)
    kinds = [(path, body.get("max_tokens")) for path, body in server.bodies[:3]]
    assert kinds == [("/v1/chat/completions", 1), ("/tokenize", None), ("/v1/chat/completions", 3)]
    assert server.bodies[0][1]["reasoning_effort"] == "medium" and "tools" not in server.bodies[0][1]
    assert server.bodies[1][1]["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert server.bodies[2][1]["add_generation_prompt"] and server.bodies[2][1]["messages"][-1]["role"] == "user"


def test_a_server_with_no_capture_flag_gives_a_clear_error(tmp_path):
    capture.run("http://x", "m", data()[:1], None, tmp_path / "c.jsonl", 6, post=Server(False), log=lambda line: None)
    row = json.loads((tmp_path / "c.jsonl").read_text().splitlines()[0])
    assert "boot the server with its capture flag" in row["error"]
