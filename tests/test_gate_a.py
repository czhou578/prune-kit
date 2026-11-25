"""Gate A (plans/PLAN_PRUNE_KIT.md §7): the preflight gives the numbers of the record, from the real checkpoints.

A test is skipped on a machine that does not hold its checkpoint. No test loads weights.
"""

import pytest

from prune_kit.facts import inspect
from prune_kit.resolve import ModelNotFound, resolve
from prune_kit.weights import WeightsError

GIB, MIB = 1 << 30, 1 << 20


def real(model):
    try:
        return inspect(resolve(model))
    except (ModelNotFound, WeightsError, OSError) as e:
        pytest.skip(f"{model} is not on this machine: {e}")


def test_qwen_k208_frees_the_recorded_bytes():
    """QWEN.md §3 of spark-moe-specialize: 256 experts, 8 per token, 77% of the weights; K 208 removes 3.17 GiB."""
    f = real("unsloth/Qwen3.6-35B-A3B-NVFP4-Fast")
    assert (len(f.experts), set(f.experts.values()), f.per_token) == (40, {256}, 8)
    assert f.routed_bytes / f.total_bytes == pytest.approx(0.77, abs=0.005)
    assert f.routed_bytes / (40 * 256) / MIB == pytest.approx(1.69, abs=0.005)
    assert f.freed(208) / GIB == pytest.approx(3.17, abs=0.01)
    assert [round(f.freed(k) / GIB, 2) for k in (192, 176)] == [4.22, 5.27]
    assert f.draft.experts == 256 and not f.router_bias


def test_inkling_gives_the_table_of_the_plan():
    """Plan §5.4: 128 experts in 24 layers, 4.86 MiB per expert, layers 6 to 17 run 3 times."""
    f = real("thinkingmachines/Inkling-2-Mini")
    assert (len(f.experts), set(f.experts.values()), f.per_token, f.layout) == (24, {128}, 8, "stacked")
    assert f.routed_bytes / GIB == pytest.approx(14.59, abs=0.005)
    assert f.routed_bytes / (24 * 128) / MIB == pytest.approx(4.86, abs=0.005)
    assert [round(f.freed(k) / GIB, 2) for k in (120, 112, 104, 96)] == [0.91, 1.82, 2.74, 3.65]
    assert f.loop == (6, 18, 3) and f.router_bias and f.draft.experts is None and f.draft.own_file
