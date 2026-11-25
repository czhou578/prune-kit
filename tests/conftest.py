import json
import struct

import pytest


def write_st(path, tensors):
    """Write a safetensors file with zero bytes as data. ``tensors`` is ``{name: (dtype, shape, size in bytes)}``."""
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, (dtype, shape, size) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


@pytest.fixture
def qwen_like(tmp_path):
    """2 layers with 4 experts each, one tensor set per expert (100 + 10 bytes), and a stacked draft with 6 experts."""
    t = {"model.embed.weight": ("BF16", (10, 4), 80), "mtp.layers.0.mlp.gate.weight": ("BF16", (6, 4), 48),
         "mtp.layers.0.mlp.experts.gate_up_proj": ("BF16", (6, 8, 4), 384)}
    for layer in range(2):
        p = f"model.language_model.layers.{layer}.mlp."
        t[p + "gate.weight"] = ("BF16", (4, 4), 32)
        t[p + "shared_expert.down_proj.weight_packed"] = ("U8", (4, 4), 16)
        for e in range(4):
            t[f"{p}experts.{e}.down_proj.weight_packed"] = ("U8", (4, 25), 100)
            t[f"{p}experts.{e}.down_proj.weight_scale"] = ("F8_E4M3", (10,), 10)
    write_st(tmp_path / "model-00001-of-00001.safetensors", t)
    config = {"model_type": "qwen3_5_moe", "quantization_config": {"quant_method": "compressed-tensors"},
              "text_config": {"num_experts": 4, "num_experts_per_tok": 2, "num_hidden_layers": 2}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.fixture
def inkling_like(tmp_path):
    """3 layers with 8 experts each in stacked tensors (640 + 8 bytes), a router bias, a loop, a dense draft file."""
    t = {"model.visual.blocks.0.mlp.weight": ("BF16", (8, 4), 64)}
    for layer in range(3):
        p = f"model.llm.layers.{layer}.mlp."
        t[p + "experts.w13_weight"] = ("U8", (8, 10, 8), 640)
        t[p + "experts.w13_weight_amax"] = ("F32", (8,), 8)
        t[p + "experts.w13_weight_outer_scale_denominator"] = ("F32", (1,), 4)
        t[p + "gate.weight"] = ("BF16", (8, 4), 64)
        t[p + "gate.bias"] = ("F32", (8,), 32)
    write_st(tmp_path / "model-00001-of-00001.safetensors", t)
    write_st(tmp_path / "mtp.safetensors", {"layers.0.mlp.down_proj.weight": ("BF16", (4, 4), 32)})
    config = {"model_type": "inkling2_loop", "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
              "text_config": {"n_routed_experts": 8, "num_experts_per_tok": 2,
                              "loop_config": {"core_start": 1, "core_end": 2, "num_passes": 3}}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path
