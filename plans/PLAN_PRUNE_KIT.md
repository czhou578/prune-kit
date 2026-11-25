# Plan: prune-kit, one pipeline that prunes a local model for one workflow

Status: 2026-11-25. Stage I (§6) is built and merged, and gate A (§7) passes for Qwen, Inkling and DeepSeek. Stage
II is built in ten pull requests, with no window: gates B and C pass (§7). Stage III is not started: each of its
runs needs a window and your approval. The project name is a placeholder.

## In simple words

- You pruned two models by hand: DeepSeek V4 Flash and Qwen3.6-35B-A3B. Both prunes used the same steps: record
  which parts of the model your work uses, rank them, remove the cold parts, and test the result.
- prune-kit makes those steps one tool. Then the next model needs no new project.
- First, a quick check reads the files of a model. In seconds, it tells you how much memory each cut frees, and
  whether the pipeline can run on this model.
- Then a longer check loads the model once. It measures how deep the cut can go for your workflow.
- prune-kit never changes a live worker, and it never decides for you. It ends with a pruned folder and a report.

## 1. Verdict

The pipeline is feasible. It has four limits:

1. **The prune works on MoE models only.** The lever is the set of routed experts. A dense model has no routed
   experts, and the removal of layers failed on DeepSeek. For a dense model, the quick check answers "no prune
   without training" and lists the other levers.
2. **Each model family needs an adapter.** The method is the same for both models. The checkpoint layout, the
   server and the chat template are not. An adapter holds these three parts, and the rest of the tool is shared.
3. **The quick check gives the memory, not the depth.** The memory that a cut frees is exact arithmetic. The depth
   that a model tolerates is not: the Qwen plan expected that K 192 passes, and K 208 already failed. Only a
   measurement gives the depth.
4. **Each measurement needs a memory window.** The capture and the gates need the full model in memory. On this
   machine, 3 GiB is available with both workers up. So each window stops a worker, with your approval.

A stop point exists after the quick check (§6, stage I). If you prune only one more model, the blueprint and the
quick check give most of the value. The rest of the tool then costs more than a third prune by hand.

## 2. Facts

I measured these facts on this machine on 2026-11-24, or I read them from `~/Projects/dsv4-specialize`. This plan
calls that repo "the record".

| Fact | Value | Source |
|---|---|---|
| Memory | 121 GiB in total; 3 GiB available and 9 GiB of swap free with both workers up | `free -g` |
| Disk | 3.0 TB free | `df -h` |
| DeepSeek V4 Flash, stock | 43 layers, 216 experts per layer, 6 per token; 9.04 MiB per expert; the 43 layer files hold 81.96 GiB, about 86% of the 95.39 GiB that load | file sizes of `data/tp1`, `PRUNE.md` |
| DeepSeek, result | 95.39 GiB to 69.79 GiB (−27%): K 160 removes 2,408 experts (21.27 GiB), then the MD cut removes 486 more (4.29 GiB). K 144 fails with every rank | `README.md`, `PRUNE.md` |
| Qwen3.6-35B-A3B, full | 40 layers, 256 experts per layer, 8 per token; 1.69 MiB per expert; the experts hold 77% of the weights | `PLAN_QWEN_PRUNE.md` §1, `QWEN.md` §3 |
| Qwen, result | K 208 removes 3.17 GiB of experts (15% of 21.22 GiB). The NVFP4 draft brings the total to −20%. The NLL gate fails at +4.15%, and you chose K 208 after a blind judge | `QWEN.md` §3, §4 |
| Qwen, the forecast before the gate | "A likely result is that 192 passes and 160 fails." K 208 failed | `PLAN_QWEN_PRUNE.md` §1 |
| NLL cost per point of removed routing share | DeepSeek: 0.12 in the first step, then 0.51 and 0.53. Qwen: about 1.05 (4.2% for about 4% of the routing) | `PLAN_BETTER_RANK.md` §1 and its result, `PLAN_QWEN_PRUNE.md` §7.2 |
| Replay noise | Up to 0.56% for one DeepSeek replay; 0.1% between two windows of the full Qwen | `PLAN_QWEN_PRUNE.md` §4, `QWEN.md` §3 |
| vLLM overhead on Qwen | 6.2 to 7.7 GiB on top of the weights and the KV pin. No setting lowers it | `PLAN_QWEN_PRUNE.md` §7.4, §8 |
| Time of a window | The Qwen capture (2.44M tokens, 99 segments) took 79 minutes of downtime. Gate window 1 took 33 minutes | `PLAN_QWEN_PRUNE.md` §6.1, §7.1 |
| Stock DeepSeek checkpoint | On disk: 98.83 GiB in `~/Projects/DeepSeek-v4-Flash-One-DGX-Spark/data/tp1`. The layer files are `root:600` | `ls` |
| Full Qwen checkpoint | On disk: 22.01 GiB of tensors. Its files are links into the shared blob folder of the Hugging Face cache, so `du` without `-L` shows 6.7 MB. `data/qwen-k208/PRUNE.json` names it as the source | `du -L`, `prune_kit.facts.inspect` |
| Qwen gate files | On disk: `results/qwen-recorded.jsonl` (the 60 recorded turns), `results/qwen-fit-g208-score.jsonl`, `results/qwen-rank.json` | `ls` |
| Inkling-2-Mini | On disk, in the Hugging Face cache. 24 layers, and the middle 12 run three times. 128 experts per layer, 8 per token. The experts hold 14.59 GiB of 18.18 GiB (80%), 4.86 MiB per expert | `config.json`, the safetensors header |
| Inkling-2-Mini, layout | One stacked tensor set per layer (`mlp.experts.w13_weight`, `w2_weight`, with scale tensors), in NVFP4 and MXFP8. The router has a bias (`mlp.gate.bias`). The MTP draft is dense and is in its own file | the safetensors index |
| Inkling-2-Mini, server | A vLLM Docker image, not the 0.33 venv. It loads 18.52 GiB. A context of 262,144 tokens with the draft needs a KV pin of 20 GiB | `model-benchmarks/models/inkling2_mini_26b_a3b_thinkingmachines_nvfp4.yml` |
| vLLM 0.33, capture | The flag `--enable-return-routed-experts` is in the shared MoE layer (`fused_moe/routed_experts_capturer.py`). More than 40 model files use that layer | `~/venvs/vllm033` |
| vLLM 0.33, removed experts | A loader for a checkpoint that lists removed experts per layer (`fused_moe/expert_substitution.py`, config key `approximate_experts`). Not tested here | `~/venvs/vllm033` |
| Data format | `sessions.jsonl`: one session per line, with `id`, `source`, `skill`, `split`, `system_prompt` and `segments` | `data/qwen-data/sessions.jsonl` |
| Private imports | `capture.py` and `qwen_gate.py` import `model_eval/replay.py` of the private bot repo | the tool sources |
| agent-eval for each server | The option `--endpoint` is on the branch `agent-eval-15-endpoint`. The local `main` does not have it | `git log` |

## 3. The terms

| Term | Meaning |
|---|---|
| Workflow | One kind of work that a model does for you, for example the stock bot or the fast coding route. A workflow file describes it (§5.2) |
| Adapter | The part of prune-kit that knows one checkpoint layout, one server and one chat template (§5.3) |
| K | The number of routed experts that a layer keeps |
| Point | One candidate cut: a K, or a drop list with a different K in each layer |
| Preflight | The quick check, level 0. It reads files only (§5.4) |
| Probe | The longer check, level 1. It loads the model in one window and measures two or three points (§5.5) |
| Gate | A test with a limit that a point must pass (§5.6) |
| Window | A time in which prune-kit stops a worker, runs test boots, and restores the worker (§5.7) |
| Reference | The full model, measured in the same window as the points |

## 4. The blueprint: what the two prunes share

Both prunes ran these stages in this order. The last column shows where the model-specific work was.

| Stage | DeepSeek V4 Flash | Qwen3.6-35B-A3B | Shared or adapter |
|---|---|---|---|
| 1. Data | Bot sessions, 1.75M tokens, 34 segments | Transcripts, bot sessions and repositories, 2.44M tokens, 99 segments | Shared, with one data format |
| 2. Capture | A server patch and Model Runner V1 | One flag of vLLM 0.33 | Adapter (server) |
| 3. Rank | `capture.py --rank` | The same tool | Shared |
| 4. Screen | A mask of the router, built in a minute | Direct builds: a prune is a copy of bytes | Adapter (layout) |
| 5. Prune | `prune_experts.py`, EXL3 tensors, new ids | `qwen_prune.py`, plain tensors | Adapter (layout) |
| 6. Gates | Replay NLL, tool names, speed, full runs, judge, blind review | The same gates, plus a facts test and an image test | Shared, with a scorer hook in the adapter |
| 7. Decision | You chose K 160, then MD | You chose K 208, against the NLL gate | Yours |
| Extra work | A loader patch for a different K in each layer | A patch that keeps the expert count of the MTP draft | Adapter (server) |

The record also gives rules. prune-kit applies each rule in code, and `BLUEPRINT.md` states it for a prune by hand.

- **R1. Use the workload's own rank.** With the public rank, DeepSeek K 192 fails. With the bot rank, K 176 costs
  +0.29%.
- **R2. Screen with cheap points first.** Build a costly point only for the winner.
- **R3. Measure the reference in the same window.** A point within 0.5% of a limit gets a second replay, and the
  mean decides.
- **R4. Keep the gate order.** Run the replay first, then the speed, then the task runs, then the judge.
- **R5. Read a judge result as "no loss", not as a gain.**
- **R6. Test the speed with the draft model.** A pruned target can lower the acceptance of its draft.
- **R7. Record the free swap with the available memory.** Swap hides memory pressure on this machine.
- **R8. Write a new folder.** A weight tool never changes its source, and a check compares each kept tensor byte
  for byte.
- **R9. Hold out whole sessions.** The calibration never sees a held-out session or a held-out project.

## 5. The design

### 5.1 The commands

```
prune-kit preflight MODEL [--workflow FILE]    level 0: files only, seconds, no GPU
prune-kit data FILE                            build and check the sessions of a workflow
prune-kit probe MODEL --workflow FILE          level 1: one window, two or three measured points
prune-kit capture | rank | build | gate        the stages, one at a time
prune-kit report RUN                           the decision report of a run
```

`MODEL` is a folder, a Hugging Face name in a local cache, or a model config of `model-benchmarks`. prune-kit
searches `~/.cache/huggingface/hub`, each extra hub that the machine file names, and a plain path.

The data of a workflow goes to `runs/data/<workflow>/`, because it does not depend on the model. Each command for a
model writes its output to `runs/<model>/<workflow>/`. A second start continues after the last complete step, as
`capture.py` does today.

### 5.2 The workflow file

One YAML file describes one workflow. This example is the fast coding route:

```yaml
name: coding-fast-route
data:
  sources:
    - {kind: gateway_records, route: second-model}   # ~/.local/share/model-gateway/records
    - {kind: claude_code, projects: ["*"]}           # ~/.claude/projects
    - {kind: repo, paths: [~/Projects/model-benchmarks]}
  held_out: {share: 0.15, projects: [nanoGPT-inference]}
  min_calibration_tokens: 1000000
memory: {peak_gib: 26.5, context: 128000, margin_gib: 4}
gates:
  nll:   {limit_pct: 3.0, turns: 60}
  tools: {names_min_pct: 98, decisions_max_drop: 3}
  tasks: {sets: [dev, heldout], seeds: 3}
  judge: {turns: 30, model: deepseek}
points: auto            # or a list of K values
```

- The data builders write `sessions.jsonl` in the format of the record, so the old data loads unchanged.
- The sources are the builders of `qwen_data.py`, plus the records of `model-gateway`. The Hermes sessions come in
  as a `sessions.jsonl` that exists: the bot repo has their export tool, and that tool is specific to the bot.
- The output can hold secrets. It stays local and out of git, and a pass replaces strings that look like keys.

### 5.3 The adapter

An adapter is one Python class and one YAML profile. The class knows the layout. The profile holds the serve
command, the port, the health URL and the pinned server version.

| Method | Use |
|---|---|
| `inspect(folder)` | The facts of §5.4: layers, experts, experts per token, bytes per expert, the draft, the format |
| `prune(source, out, keep)` | A copy with the kept experts of each layer. It copies bytes and does no new quantization |
| `check(source, out)` | The byte-for-byte check of each kept tensor |
| `mask(source, out, drop)` | Optional: a copy in which the router never picks a dropped expert |
| `routing(reply)` | The routed experts of one reply, as an array of tokens, layers and experts |
| `scorer` | The request that scores a recorded turn, and the place where the turn starts in the prompt |

The first adapters:

| Adapter | Covers | Source of the code |
|---|---|---|
| `st_moe` | Plain safetensors with one tensor set per expert (Qwen) or one stacked set per layer (the Qwen draft, Inkling) | `qwen_prune.py`, made general |
| `dsv4_exl3` | The EXL3 checkpoint of DeepSeek V4 Flash | A wrap of `mask_checkpoint.py`, `topk_checkpoint.py` and `prune_experts.py` |
| `dense` | Each model with no routed experts | New: `inspect` only |

The chat template is real adapter work. In the Qwen capture window, each request failed at first: the Qwen template
refuses a message list with no user message, and the DeepSeek template accepts it.

### 5.4 The preflight (level 0)

The preflight reads `config.json` and the headers of the weight files. It loads no weights and needs no window.

**What it prints.** This is the output of `prune-kit preflight thinkingmachines/Inkling-2-Mini --k 120 112 104 96`
on 2026-11-24. The tool gives the cut table that I first computed by hand from the header.

```
model     ~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/ed40e7da…
type      inkling2_loop: MoE, stacked layout, NVFP4
weights   18.78 GiB in the folder; draft 0.59 GiB
experts   128 per layer, 8 per token, in 24 layers; 4.86 MiB per expert
routed    14.59 GiB (78% of the weights)

cut       freed        of the weights
K 120      0.91 GiB    5%
K 112      1.82 GiB    10%
K 104      2.74 GiB    15%
K 96       3.65 GiB    19%

can the pipeline run?
  routed experts  yes      128 per layer, 8 per token
  prune tool      no       prune-kit has no prune for the stacked layout yet
  capture         unknown  not proven for this model type: test the capture flag of its server
  mask            info     possible: the router has a bias
  draft model     info     dense: no expert count to keep; test its acceptance
  loop            info     layers 6 to 17 run 3 times: the counts of such a layer are the sum of the passes
  memory          info     the full model loads 18.78 GiB, and 4.1 GiB is available: needs a window
verdict   GO WITH WORK: prune tool, capture
```

The files do not hold the other levers. For this model, the KV pin is one: 20 GiB at 262,144 tokens with the
draft, so a smaller context can free more than the prune (`model-benchmarks`).

**The memory model.** The peak of a server is the loaded weights, plus the KV pin, plus the overhead of the server.
The weights that a point frees are exact. The overhead comes from the profile of the adapter, and the probe
measures it when no number exists.

**The verdicts.**

| Verdict | Meaning |
|---|---|
| GO | Each stage has its adapter part, and the server version matches the profile |
| GO WITH WORK | The model is MoE, and a listed part is missing. The list names each part |
| NO-GO | No routed experts, or a format that no tool can copy by bytes (for example GGUF) |

**With a workflow file**, the preflight adds two answers:

- The data: the tokens of the calibration split and of the held-out split, for each source, against the minimum.
- The need: the K that reaches `memory.peak_gib`, and that cut as a share of the experts. It prints the two known
  results beside it: Qwen held 19% with a cost, and DeepSeek held 31%.

**What the preflight cannot give** is the depth. Before a measurement, the only honest estimate is the range of the
two earlier prunes: 15% to 27% of the loaded weights. That range comes from two models.

### 5.5 The probe (level 1)

The probe measures. It runs in one window, on a test port, and it never uses a live port.

1. Boot the full model with the capture.
2. Capture a sample of the calibration split. The default is 400,000 tokens.
3. Record 20 held-out turns of the full model, and score them. This is the reference.
4. Rank the sample. Rank each half of the sample apart, and compare the two drop lists at each K. A low overlap
   shows that the sample is too small.
5. Build two or three points from the sample rank: masks where the adapter has a mask, and prunes where it has not.
6. Boot each point, and score the same 20 turns.
7. Stop the test server, and restore the worker.

**What it prints.** This example holds the recorded gate results of Qwen, in the format of the probe:

```
point    weights freed    NLL against the reference    gate nll (+3%)
K 208    3.17 GiB         +4.15%                       fail
K 192    4.22 GiB         +8.0%                        fail
K 176    5.27 GiB         +11.1%                       fail

estimate: the gate holds at about K 224 (2.11 GiB freed)
measured overhead of the server: 6.6 GiB at the peak
```

- The estimate comes from a line through the measured points. On DeepSeek the cost per point of routing share
  stayed at 0.51 to 0.53 over two steps, so a line is a fair first model.
- 20 turns give more noise than the 60 turns of the full gate. The probe prints a band, and a point near the limit
  goes to the full gate.
- My estimate for the time is 60 to 90 minutes for a model of the size of Qwen. It comes from the two Qwen windows
  of §2.

### 5.6 The gates and the report

The gates come from the two prunes. The limits are the defaults of a workflow file.

| Gate | Measure | Default limit |
|---|---|---|
| `nll` | The NLL of 60 held-out turns that the reference wrote | At most +3% |
| `tools` | On the same turns: the tool names and the first answer token | Names at least 98% of the reference; decisions down by at most 3 |
| `tasks` | The tool-call suite of `model-benchmarks` (85 tasks), and the task sets of `agent-eval` | At most 3 tasks fewer |
| `judge` | A blind judge on 30 held-out turns, with a sign test, and a page of blind pairs for you | No loss that the test shows |
| `speed` | The decode speed with the draft, against the reference | Report |
| `memory` | The peak at the context of the workflow | The margin of the workflow |

The report gives one row per point, with each gate and the evidence files. A point has one of three results:

- **Pass:** each gate passes.
- **Your decision:** `nll` fails, and `tools`, `tasks` and `judge` pass. This is the Qwen K 208 case.
- **Fail:** each other case.

prune-kit recommends a point. You choose it.

### 5.7 The window runner

The runner is a Python port of the rules of `scripts/run_lib.sh`. A machine file holds the facts of one machine:
the workers with their stop, start and health commands, the hours in which a worker must be up, and the memory
limit of the watch.

- CAUTION: A window stops a live worker, and the worker gives no answers until the restore. The runner stops a
  worker only when you name it on the command line: `--stop deepseek`.
- The runner stops a worker only when its `/metrics` shows 0 running and 0 waiting requests.
- The deadlines have no defaults: `--start-after`, `--boot-by` and `--restore-at`.
- A boot that is not healthy in 15 minutes goes down. After two failed boots in a row, the runner restores.
- A watch reads the available memory and the free swap. Below the limit, it stops the client.
- On each other failure, the runner does the safe action: the restore.
- Each window runs in tmux under `job run`, with a `--desc` and a `--log`, so that it shows on the jobs page.

### 5.8 What prune-kit reuses, and what it does not do

| Part | Project | Use |
|---|---|---|
| The weight tools, the capture and the gate | `dsv4-specialize` | Copied into prune-kit with their tests. That repo is a finished record and does not change |
| The task gate | `agent-eval`, `agent-eval-heldout` | Called through `--endpoint` |
| The tool-call suite and the speed | `model-benchmarks` | Called in external mode |
| The jobs page | `machine-dash` | `job run`, `job step`, `job set --phase` |
| The data of the local routes | `model-gateway` | The records folder is a data source |

prune-kit does not do these things:

- **The deploy.** The switch of a live worker stays a step by hand, with its own plan and your approval.
- **Training.** A heal of a prune with LoRA belongs to `night-train` (its method M1).
- **New quantization, and the serving settings.** The preflight only lists them as other levers.

## 6. The steps and the pull requests

Each pull request changes at most 400 lines and passes its tests alone. The line counts are my estimates.

**Stage I: the blueprint and the preflight. No GPU and no window.**

| PR | Content | Lines |
|---|---|---|
| 1 | The repo: `pyproject.toml`, the command entry, `README.md`, and `BLUEPRINT.md` with the stages and the rules of §4 | 300 |
| 2 | The model resolver and `inspect` of `st_moe`, with tests on small synthetic checkpoints | 350 |
| 3 | `prune-kit preflight`: the memory table, the verdict list, the text and JSON output. Gate A for Qwen and Inkling | 350 |
| 4 | `inspect` of `dsv4_exl3` and of `dense`. Gate A for DeepSeek | 300 |

Stop point: you decide if the work goes on.

**Stage II: the shared stages. No window.** Built on 2026-11-25. Three steps are two pull requests each, because
one pull request holds at most 400 lines. The line counts are the real counts.

| PR | Content | Lines |
|---|---|---|
| 5 | The workflow file: the schema, the loader, the checks, and the cut that reaches the memory goal in the preflight | 359 |
| 6 | The data format, the held-out split, the secret pass, and the data line of the preflight | 289 |
| 7 | The data sources and `prune-kit data`: Claude Code transcripts, gateway records, repositories, sessions that exist | 346 |
| 8 | `prune` and `check` of `st_moe`, for the per-expert and the stacked layout | 367 |
| 8b | `prune-kit build`, the gate B script, and the new state of the prune tool in the preflight | 128 |
| 9 | The capture client, with no import from the bot repo | 266 |
| 9b | The rank, the overlap of two halves, `prune-kit capture` and `prune-kit rank` | 262 |
| 10 | The replay scorer: record, score, compare, `prune-kit gate`, and the gate C script | 393 |
| 11 | The window runner, the machine file and the window file, with dry-run tests | 381 |
| 11b | `prune-kit window`, the dry run, and the example files | 139 |
| 12 | The test of gate C with no server, and this status | 63 |

What stage II did not prove, because the proof needs a window:

- The capture client did not run against a real capture server. Its tests use a server of the test. The rank is
  proven: the rank of the capture of the Qwen prune is equal to the rank file of the record.
- The window runner did not run a real window. The commands of `examples/machine-spark.yml` are not tested, and the
  end of the `up` hours of DeepSeek in that file is an assumption.
- prune-kit has no new dependency for the capture: it reads the `.npy` array of the server with the standard
  library. PyYAML is the only dependency.

**Stage III: the measurements. Each run needs a window.**

| PR | Content | Lines |
|---|---|---|
| 12 | `prune-kit probe`: the sample, the points, the line, the report | 350 |
| 13 | The gates `tasks`, `speed` and `memory`, the judge, and the report with the blind page | 400 |
| 14 | The third model: its profile, the missing adapter parts, and gate D | 300 |

## 7. The gates of the project

| Gate | Test | Needs |
|---|---|---|
| A | The preflight gives the recorded numbers: Qwen K 208 frees 3.17 GiB; DeepSeek K 160 frees 21.27 GiB in 28,896 tensors, and the MD cut frees 4.29 GiB; Inkling gives the table of §5.4 | Files only. The tool gives 3.16 GiB for Qwen K 208, and the record rounds to 3.17 GiB |
| B | `st_moe` builds K 208 from the full Qwen with `results/qwen-rank.json`. Each tensor is the same as in `data/qwen-k208`, the folder of the live server, byte for byte | **Pass, 2026-11-25** (`scripts/gate_b.sh`). The five weight files and the index are equal, byte for byte: 101,316 tensors, 18.84 GiB. `config.json` is equal without `quantization_config`: a later step of the prune by hand changed those keys for the 4-bit draft |
| C | The new scorer sends the requests of the scorer of the record, byte for byte. On the live Qwen, its NLL of `results/qwen-recorded.jsonl` is equal to `results/qwen-fit-g208-score.jsonl` inside the noise of two runs | **Pass, 2026-11-25** (`tests/test_gate_c.py`, `scripts/gate_c.sh`). The 121 requests are equal. The token counts of the 60 turns are equal. NLL per token: record 0.3274; old scorer today 0.3282; new scorer 0.3303 and 0.3283 in two runs. Two runs of the same scorer differ by 0.62%. The first limit of this gate was 0.2%: that limit was below the noise, so I changed the gate |
| D | A third model runs from the preflight to the report. The probe estimate and the full gate agree within the band of the probe | Windows, with your approval for each stop |

Gate D is the test of the claim "each model". Inkling-2-Mini is a hard case: a different layout, a different
server, and layers that run three times.

## 8. Risks

1. **The estimate comes from two models only.** The preflight states a range, and the probe measures. Each model that
   runs through prune-kit adds a data point to the range.
2. **A server update breaks an adapter.** Each profile pins a server version, and the preflight compares it with
   the installed version.
3. **A new workflow has no data.** Without recorded traffic, the rank is the rank of other work. The preflight
   reports the tokens against the minimum, and it stops the pipeline below the minimum.
4. **A gate is not your view.** Qwen K 208 fails `nll`, and you see no loss. So the report shows the evidence, and
   the decision stays with you.
5. **A window stops a live worker.** The runner needs your `--stop`, has deadlines with no defaults, and restores
   on each failure. The stock bot needs DeepSeek from 05:00 on weekdays.
6. **The data holds secrets.** It stays local and out of git.
7. **A general tool costs more than one prune by hand.** The stop point after stage I limits that cost.

## 9. Open points to check during the work

1. Does the capture flag work with the model code of Inkling in its Docker image?
2. Does `approximate_experts` of vLLM 0.33 load quantized experts? If it does, a different K in each layer needs no
   loader patch for the models of the shared MoE layer.
3. For a layer that runs three times, is the rank better per pass or per layer?
4. Does the share of a layer that one token uses predict the depth? DeepSeek uses 2.8% (6 of 216), Qwen 3.1%
   (8 of 256) and Inkling 6.25% (8 of 128). The record says that a small model loses more. Inkling is the third
   data point.
5. Can a mask work with no copy of the checkpoint, as a setting of the server?
6. When does `--endpoint` of `agent-eval` reach `main`? PR 13 needs it.

## 10. Your decisions

1. **The name and the place.** I used `~/Projects/prune-kit` as a new repo. Choose another name at any time before
   PR 1.
2. **The first scope.** My recommendation: MoE models on vLLM only.
3. **The third model.** My recommendation: Inkling-2-Mini, because it is on disk and it differs from both earlier
   models.
4. **The stop point.** My recommendation: build stage I, use the preflight on your next candidate model, and then
   decide on stages II and III.
5. **The default limits of the gates.** The table of §5.6 holds the limits of the two prunes. Change them here if
   your next workflow needs other limits.
