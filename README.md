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
| `prune-kit data FILE`: the sessions of a workflow, the held-out split, the secret pass | Built |
| `prune-kit build MODEL`: the pruned folder, and its check against the source | Built for safetensors checkpoints. Not for EXL3 |
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

### The data of a workflow

```bash
uv run prune-kit data examples/coding-fast-route.yml
```

`prune-kit data FILE` reads each source of the workflow file and writes `sessions.jsonl` into the data folder. It
prints the table of the sources, and it compares the totals with the minimums. The exit code is 0 when the data
meets the minimums, and 1 when it does not. `--stats` reads the sessions that exist and builds nothing.

| Kind of source | Content | Keys |
|---|---|---|
| `claude_code` | The Claude Code transcripts, one session per transcript | `root`, `projects` (patterns), `map`, `budget` |
| `gateway_records` | The records of `model-gateway`, one session per conversation | `root`, `route`, `map`, `budget` |
| `repo` | Code files of a repository, each as a file read | `paths`, `files`, `budget` |
| `sessions` | A `sessions.jsonl` that exists, with its own split | `path`, `sources` |

- `budget` is the largest count of calibration tokens that the source gives. Each project gives one segment in turn.
- `map: hermes` renames the Claude Code tools to the Hermes tools, as the Qwen prune did.
- The Hermes sessions of the stock bot come in as `sessions`: the bot repo has the export tool for them.
- A session of a project in `held_out.projects` is held out. Of the other sessions, a hash holds out `share`.

The preflight also reads `sessions.jsonl` in the data folder of the workflow (`data.folder`, default
`runs/data/<name>`). It prints the sessions, the segments, the tokens and the target turns of each source and split.
The line `data` compares the calibration tokens and the held-out turns with the minimums of the workflow. Below a
minimum, the verdict lists `data` as work.

- CAUTION: `sessions.jsonl` can hold private text of your transcripts. Keep it out of git: `runs/` is in
  `.gitignore`. The secret pass replaces only the strings that look like keys.
- A token count of the data is an estimate: the characters divided by 4.

### The pruned folder

```bash
uv run prune-kit build unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --rank rank.json --k 208 --out runs/qwen-k208
```

`prune-kit build` writes a copy of the model with K experts in each layer. It keeps the first K experts of each
layer of the rank file, and it gives them the ids 0 to K-1. It copies bytes: it does no new quantization, and it
never changes the full model. Then it compares each kept tensor with its source, byte for byte.

- The output folder must not exist. The build stops before it writes, when the disk has too little free space.
- `--draft` sets the experts of the draft model: `rank` (the rank layer after the last layer of the model), `first`
  (the experts 0 to K-1), or `keep` (the whole draft: serve the draft from the full model).
- `--check` only compares a pruned folder that exists with its source.
- `scripts/gate_b.sh` builds K 208 of the full Qwen and compares each file with the folder of the prune by hand.
  It needs that folder, and about 19 GiB of disk for some minutes.

## Tests

```bash
uv run pytest -q
```

The tests use small synthetic files in temporary folders. They load no model. `tests/test_gate_a.py` reads the
headers of the real checkpoints of this machine and compares the preflight with the recorded numbers of the two
prunes. Each of its tests is skipped where its checkpoint does not exist.
