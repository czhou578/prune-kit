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
| `prune-kit capture MODEL` and `prune-kit rank CAPTURE`: the routed experts of the workflow, and their order | Built. The capture needs a server with its capture flag |
| `prune-kit build MODEL`: the pruned folder, and its check against the source | Built for safetensors checkpoints. Not for EXL3 |
| `prune-kit gate`: the replay gates `nll` and `tools`: record, score, compare | Built |
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

### The capture and the rank

- CAUTION: The capture needs the full model on a test server with the capture of the routed experts. On vLLM 0.33
  that is the flag `--enable-return-routed-experts`. A live worker does not have this flag. Do not stop a live
  worker for a capture without a plan for the window.

```bash
uv run prune-kit capture unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --workflow examples/coding-fast-route.yml --base-url http://127.0.0.1:8890
```

The capture sends each calibration segment of the workflow once. It writes one row of counts for each segment to
`runs/<model>/<workflow>/capture.jsonl`. A second start continues after the last good row.

- The shared prefix (the system prompt and the tools) counts once. A token that the server did not fill does not count.
- `--generate N` also counts the routing of N tokens that the model writes after each segment.
- `--body JSON` adds keys to each request, for example `{"reasoning_effort": "medium"}`.

```bash
uv run prune-kit rank runs/unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/coding-fast-route/capture.jsonl
```

The rank orders the experts of each layer by their count, and it writes `rank.json` beside the capture. For each K,
it prints the share of the routing that the cut removes. It also ranks two halves of the capture apart, and it
prints the share of the dropped experts that both halves drop. A low overlap shows that the capture is too small.
The rank of the capture of the Qwen prune is equal to the rank file of the record: a test compares them.

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

### The replay gate

The replay gate has three steps. Each step is one command, and a second start continues an output file.

1. Record: the full model writes one turn for each of 60 held-out targets.
2. Score: a model scores the recorded turns. Score the full model and each pruned point in the same window.
3. Compare: the totals of two score files, against the limits of the gates `nll` and `tools`.

```bash
uv run prune-kit gate --record --workflow examples/coding-fast-route.yml --base-url http://127.0.0.1:8889 --out runs/recorded.jsonl
```

```bash
uv run prune-kit gate --score --workflow examples/coding-fast-route.yml --base-url http://127.0.0.1:8889 --recorded runs/recorded.jsonl --out runs/score-k208.jsonl
```

```bash
uv run prune-kit gate --compare runs/score-full.jsonl runs/score-k208.jsonl --workflow examples/coding-fast-route.yml
```

- The exit code of `--compare` is 0 when each gate passes, and 1 when a gate fails.
- `--data DIR` gives a data folder in place of the folder of the workflow. Without a workflow file, the limits are
  the limits of the two prunes.
- The score of one turn moves by 2% to 5% between two runs on the live Qwen. The total of 60 turns moves much less.
  Compare totals only, and score the reference in the same window as the points.
- `scripts/gate_c.sh` scores the recorded turns of the Qwen prune with this scorer and with the scorer of the
  record, on the live Qwen. It stops no server.

## Tests

```bash
uv run pytest -q
```

The tests use small synthetic files in temporary folders. They load no model. `tests/test_gate_a.py` reads the
headers of the real checkpoints of this machine and compares the preflight with the recorded numbers of the two
prunes. Each of its tests is skipped where its checkpoint does not exist.
