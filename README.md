# prune-kit

One pipeline that prunes a local model for one workflow.

## In simple words

- A local model holds parts that your work never uses. A prune removes them, and the model needs less memory.
- Two prunes by hand showed the method: DeepSeek V4 Flash (−27% of the loaded weights) and Qwen3.6-35B-A3B (−20%).
- prune-kit turns that method into commands, so that the next model needs no new project.
- A quick check comes first. It tells you how much memory each cut frees, and whether the pipeline can run.
- prune-kit never changes a live model server, and it never chooses the cut for you.

## Status

Stage I of [the plan](plans/PLAN_PRUNE_KIT.md) is built: the blueprint and the quick check. Stage II, the shared
stages, is in work. Stage III is not started.

| Part | State |
|---|---|
| [BLUEPRINT.md](BLUEPRINT.md): the stages, the gates and the rules of a prune | Written |
| `prune-kit stages`: the stages, and the command of each stage | Built |
| `prune-kit preflight MODEL`: the quick check, from the model files only | Built for safetensors checkpoints, EXL3 included |
| The workflow file, and `prune-kit preflight MODEL --workflow FILE` | Built |
| The data format: `sessions.jsonl`, the held-out split, the secret pass | Built. No command builds the sessions yet |
| The other stages | Not built. Do them by hand with the blueprint |

## Scope

- prune-kit prunes the routed experts of a mixture-of-experts (MoE) model. It does not prune a dense model.
- It does not train, and it does not quantize.
- It does not deploy. The switch of a live model server stays a step by hand.

## Install

```bash
uv sync
```

## Use

```bash
uv run prune-kit stages
```

```bash
uv run prune-kit preflight thinkingmachines/Inkling-2-Mini
```

`MODEL` is a folder, a Hugging Face name in a local cache, or a part of such a name. The preflight reads
`config.json` and the headers of the weight files. It loads no weights, so it runs in less than a second beside
the live model servers. It prints three parts:

- **The model:** the layers, the routed experts per layer and per token, and the size of one expert.
- **The cuts:** the memory that each K frees. `--k 208 192` sets the cuts.
- **Can the pipeline run?** One line for each part that a prune needs. `yes` is ready, `no` and `unknown` need
  work, and `info` is a fact that blocks nothing.

The verdict is `GO`, `GO WITH WORK` with the parts that need work, or `NO-GO` for a model with no routed experts.
The exit code is 0, 0 and 1. `--json` prints the same report as JSON.

- A dense model gets `NO-GO` and a list of the other levers. A folder of GGUF files gets `NO-GO`.
- A weight file with no read permission still counts: the preflight takes its tensor names from the index of the
  folder and its size from the file. The line `sizes` shows this case.

The preflight gives no estimate of how deep a cut can go. Only a measurement gives that depth: see §5.5 of the
plan.

### The workflow file

One YAML file describes one workflow: its data sources, its memory goal, and the limits of its gates.
[examples/coding-fast-route.yml](examples/coding-fast-route.yml) is the workflow of the Qwen prune. Each key except
`name` and `data.sources` has a default. A file with a fault gives one error that lists each fault.

```bash
uv run prune-kit preflight unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --workflow examples/coding-fast-route.yml
```

With a workflow file, the preflight adds the cut that reaches `memory.peak_gib`: the largest K that frees the need.
Set `memory.overhead_gib` to the measured KV pin plus the overhead of the server. Without it, the need holds the
weights only, and the preflight says that the need is a lower limit.

The preflight also reads `sessions.jsonl` in the data folder of the workflow (`data.folder`, default
`runs/data/<name>`). It prints the sessions, the segments, the tokens and the target turns of each source and split.
The line `data` compares the calibration tokens and the held-out turns with the minimums of the workflow. Below a
minimum, the verdict lists `data` as work.

- CAUTION: `sessions.jsonl` can hold private text of your transcripts. Keep it out of git: `runs/` is in
  `.gitignore`. The secret pass replaces only the strings that look like keys.
- A token count of the data is an estimate: the characters divided by 4.

## Tests

```bash
uv run pytest -q
```

The tests use small synthetic files in temporary folders. They load no model. `tests/test_gate_a.py` reads the
headers of the real checkpoints of this machine and compares the preflight with the recorded numbers of the two
prunes. Each of its tests is skipped where its checkpoint does not exist.
