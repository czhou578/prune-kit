# prune-kit

One pipeline that prunes a local model for one workflow.

## In simple words

- A local model holds parts that your work never uses. A prune removes them, and the model needs less memory.
- Two prunes by hand showed the method: DeepSeek V4 Flash (−27% of the loaded weights) and Qwen3.6-35B-A3B (−20%).
- prune-kit turns that method into commands, so that the next model needs no new project.
- A quick check comes first. It tells you how much memory each cut frees, and whether the pipeline can run.
- prune-kit never changes a live model server, and it never chooses the cut for you.

## Status

Stages I, II and III of [the plan](plans/PLAN_PRUNE_KIT.md) are built: the blueprint, the quick check, the shared
stages, and the measurements. Each gate of the project passes. The preflight gives the numbers of the two prunes by
hand (Qwen, DeepSeek). Two models ran through the pipeline in real windows on 2026-11-25: Qwen against its prune by
hand, and Inkling-2-Mini with no prune by hand. §6 of the plan has the results.

| Part | State |
|---|---|
| [BLUEPRINT.md](BLUEPRINT.md): the stages, the gates and the rules of a prune | Written |
| `prune-kit stages`: the stages, and the command of each stage | Built |
| `prune-kit preflight MODEL`: the quick check, from the model files only | Built for safetensors checkpoints, EXL3 included |
| The workflow file, and `prune-kit preflight MODEL --workflow FILE` | Built |
| `prune-kit data FILE`: the sessions of a workflow, the held-out split, the secret pass | Built |
| `prune-kit capture MODEL` and `prune-kit rank CAPTURE`: the routed experts of the workflow, and their order | Built. The capture ran on a real capture server in the probe |
| `prune-kit build MODEL`: the pruned folder, and its check against the source | Built for safetensors checkpoints. A server loaded pruned folders of Qwen (per-expert layout) and of Inkling (stacked layout). Not for EXL3 |
| `prune-kit gate`: the replay gates `nll` and `tools`: record, score, compare | Built |
| `prune-kit window`: test servers in one window, with the restore of the workers | Built. Real windows ran on Qwen and on Inkling on 2026-11-25 |
| `prune-kit probe MODEL`: the window file of a probe, and its report | Built for Qwen3.5 and Qwen3.6 MoE, and for Inkling 2 with a loop. A probe of each one ran |
| The gates `tasks`, `speed` and `memory`, and the verdict of a point in the report | Built. They ran in the gates window of 2026-11-25 |
| `prune-kit judge`: the gate `judge`, with the blind page for your own review | Built. DeepSeek judged two cuts of the full Qwen on 2026-11-25 |

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
- A model with a loop runs some layers more than one time (Inkling-2-Mini: 12 of its 24 layers run three times).
  Its server gives one row for each pass of a layer. The capture keeps each pass, and it writes the loop of the
  model into each row. The rank adds the passes of a layer: the passes share the weights, so one keep list serves
  each pass.

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

### A window

A window is a time in which test servers run in place of a live worker. `prune-kit window` runs the points of a
window file, one test server after the other, and it restores the machine at the end.

- CAUTION: A real window stops a live worker, and the worker gives no answers until the restore. Run each window
  with `--dry-run` first. The dry run prints each command, runs no command, and stops no worker.
- CAUTION: A wrong command in the machine file leaves a worker down. Check
  [examples/machine-spark.yml](examples/machine-spark.yml) against the machine before the first real window.

```bash
uv run prune-kit window examples/window-qwen-score.yml --machine examples/machine-spark.yml --stop qwen --start-after "2026-11-28 22:00" --boot-by "2026-11-29 01:00" --restore-at "2026-11-29 03:00" --dry-run
```

The rules of a window:

- A worker stops only when `--stop` names it, and only when its `/metrics` shows 0 running and 0 waiting requests.
- The three deadlines have no defaults. No point boots after `--boot-by`. Each client stops at `--restore-at`.
- The window refuses a stop that touches the `up` hours of a worker in the machine file.
- A boot that is not healthy in 15 minutes goes down. After two failed boots in a row, no more point boots.
- A watch reads the available memory and the free swap during each client. Below a limit, it stops the client.
- The window always ends with the restore of each worker that it stopped. Exit code 3 means that the restore failed.
- A stop of the runner is a failure of the window, and not the end of the restore: after Ctrl-C, `kill`, or the end
  of its terminal, the runner stops its client and its test server, and it restores the workers. A second stop has
  no effect.
- CAUTION: `kill -9` gives the runner no time. The workers then stay down, and you must start them by hand.
- A point can name a results file with its key `results`. The runner writes the result and the memory of the point
  to that file, and to the file of `--results`.
- The runner stops the whole process group of a test server, and it waits until the health URL is silent. A
  server that still answers would look like the next point.
- A window file can list folders under `release`. The runner asks the kernel to drop their cached pages before
  each boot and before the restore. On a machine with unified memory, a server start with CUDA needs free memory,
  and the page cache of large folders can make the start fail. The window file of a probe has this list.
- A start command that fails gets its second try after one minute.
- Each command of the machine file and of the window file runs with bash.

Start a real window in tmux under `job run`, with a `--desc` and a `--log`, so that it shows on the jobs page.

### The probe

The probe measures two or three cuts in one window, before a full capture. `prune-kit probe` only writes the window
file. It starts no server and stops no worker.

```bash
uv run prune-kit probe unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --workflow examples/coding-fast-route.yml --machine examples/machine-spark.yml
```

- The window has one point for the full model and one point for each cut. The cuts are `points` of the workflow
  file, or the first three default cuts.
- The full model boots with its capture flag. Its clients capture a sample of 400,000 tokens, record 20 held-out
  turns, score them, rank the sample, and build the pruned folder of each cut.
- Each cut boots, and one client scores the same 20 turns.
- The serve command comes from the profile of the model type, in `src/prune_kit/profiles/`. A model type with no
  profile gets an error. Qwen3.5 and Qwen3.6 MoE have a profile, and Inkling 2 with a loop has one.
- A profile can start its server in a container: its serve command then has `{mounts}`. The plan gives absolute
  paths, and the container gets the folder of the model and the run folder at the same paths. `down` of a profile
  is the command that removes the server after its stop.
- `replay` of a profile holds the settings of the replay gate for the model type: the tokens that start and end
  the reasoning, the number of tokens of the decision, the sampling, and the limit of a recorded turn
  (`max_tokens`, 800 with no setting). The gate commands of the window then have `--profile TYPE`. The report
  counts the recorded turns that stopped at the limit: such a turn has no decision to score.
- The command prints the `prune-kit window` command for the file. Run the dry run first.

After the window, `--report` prints the result. It needs no machine file. `--folder DIR` reads a probe folder
that moved.

```bash
uv run prune-kit probe unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --workflow examples/coding-fast-route.yml --report
```

- **The cost of each cut:** the NLL change against the full model of the same window, and the weights that the cut
  frees.
- **The band:** 20 turns give much noise. The band holds 90% of the results of new samples of the same turns. A
  cut is `pass` or `fail` only when its whole band is on one side of the limit. Each other cut is `near the limit`.
- **The estimate:** a line through the measured cuts gives the deepest K that holds the `nll` gate. The estimate
  is not a gate result.
- **The memory:** the peak of the full model in the window, from the results file of the window.

**The full gate is one more option.** `--full` puts the points of the full gate behind the points of the probe, in
the same window and with no restore between them. The full model boots again, captures the whole calibration split,
and records the turns of `gates.nll` (60). Then the same cuts boot with the rank of that capture. The files of the
full gate go to the folder `full` beside the folder `probe`, and `--report` then prints both results and compares
them: the full gate against the band of the probe, for each cut. With `--full`, `--gates` adds its clients to the
points of the full gate only.

**The full gate can also use a rank that exists.** `--rank FILE` gives the rank of a full capture: the window then
has no sample capture. `--turns 60` sets the turns of the full `nll` gate. `--gates tasks speed` adds two clients to
each point:

- `tasks`: the tool-call suite of `model-benchmarks` (85 tasks). The machine file names its folder under
  `tools.model_benchmarks`. A point passes with at most 3 tasks fewer than the full model.
- `speed`: `prune-kit gate --speed` measures the decode speed on eight fixed prompts. One prompt differs from the
  next by 10% and more, so the report also shows the median change for the same prompts. The test server of the Qwen
  profile has no draft model, so this speed is the speed with no draft. `--flags TEXT` and `--env NAME=VALUE` give
  each test server more flags and variables, for example those of its draft model. The profile shows the values.
  A draft model also uses memory: the gate `memory` then measures the test server with its draft.
- `memory` needs no client. The results file of the window gives the peak of each point, and the report compares
  it with `memory.peak_gib` of the workflow.

The report then prints a verdict for each cut:

| Verdict | Meaning |
|---|---|
| `pass` | Each gate that has a result passes, and `nll` passes |
| `your decision` | Only `nll` does not pass, and `tasks` and `judge` pass. This is the case of Qwen K 208 |
| `fail` | A gate of `tools`, `tasks`, `memory` or `judge` fails |
| `open` | `nll` does not pass, and `tasks` or `judge` has no result yet |

On the gate files of the Qwen prune, the report gives the numbers of the record: +4.15% at K 208 and +8.04% at
K 192. A test compares them.

### The judge

The judge compares the answers of the full model and of a pruned point to the same held-out requests. With
`--gates judge`, each cut of a window also records its own turns (`recorded-k<K>.jsonl`). After the window, a judge
model on a live server reads each pair:

```bash
uv run prune-kit judge runs/recorded.jsonl runs/recorded-k208.jsonl --workflow examples/coding-fast-route.yml --base-url http://127.0.0.1:8888 --out runs/judge-k208.json --page runs/pairs-k208.html
```

- The judge sees the two answers as A and B. A hash of the turn fixes which model is A. The judge does not see the
  reasoning of a model.
- A sign test compares the wins and the losses of the point. The gate finds a loss only when the point loses more
  often than it wins, and the test shows it (p below 0.05). A good result means "no loss". It never means a gain.
- `--page` writes the same pairs as one HTML page, with no model names. The key is at the end of the page.
- `--body JSON` adds keys to each judge request, for example `{"chat_template_kwargs": {"thinking": false}}`.
- Put the result into the folder of the run as `judge-k<K>.json`. The report of the run then shows it, and it uses
  it for the verdict.

## Tests

```bash
uv run pytest -q
```

The tests use small synthetic files in temporary folders. They load no model, send no request, and stop no
server. Some tests read files of this machine, and each of them is skipped where its files do not exist:

- `tests/test_gate_a.py` reads the headers of the real checkpoints and compares the preflight with the recorded
  numbers of the two prunes.
- `tests/test_rank.py` compares the rank of the capture of the Qwen prune with the rank file of the record.
- `tests/test_gate_c.py` compares the requests of the replay scorer with the requests of the scorer of the record.

Two gates of the plan are scripts, because they are long runs. Start each one under `job run`, as its first lines
show: `scripts/gate_b.sh` (about 19 GiB of disk for some minutes) and `scripts/gate_c.sh` (requests to the live Qwen
for about 20 minutes).
