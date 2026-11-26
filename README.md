# prune-kit

One pipeline that prunes a local mixture-of-experts (MoE) model for one workflow.

> **For MoE models only.** A MoE model sends each token to a few of its many experts. prune-kit removes the experts
> that your work uses least. A dense model has no such experts, so prune-kit cannot prune it. The quick check tells
> you in less than a second: it prints `NO-GO` for a dense model.

## In simple words

- A local MoE model holds experts that your work almost never uses. A prune removes them, and the model needs less memory.
- Two prunes by hand showed the method: DeepSeek V4 Flash (−27% of the loaded weights) and Qwen3.6-35B-A3B (−20%).
- prune-kit turns that method into commands, so that the next model needs no new project.
- A quick check comes first. It tells you how much memory each cut frees, and whether the pipeline can run.
- prune-kit never changes a live model server, and it never chooses the cut for you.

## Status

Stages I, II and III of [the plan](plans/PLAN_PRUNE_KIT.md) are built, and each gate of the project passes. Two
models ran through the pipeline in real windows on 2026-11-25: Qwen3.6-35B-A3B and Inkling-2-Mini.
[docs/reference.md](docs/reference.md) shows the state of each part.

## Results

These numbers are from real windows on one machine (2026-11-25), with coding sessions as the workflow. Each cut is
compared with the full model of the same window. §6 of the plan has each table and its limits.

| Model | Cut | Memory peak | `nll` (limit +3%) | `tasks` (of 85) | `judge` (30 blind pairs) | Verdict |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | Full model | 30.9 GiB | Reference | 54 | Reference | |
| | K 208 of 256 | 28.0 GiB | +5.04%: fail | 52: pass | No loss | fail (memory goal of 26.5 GiB) |
| | K 192 of 256 | 26.4 GiB | +6.36%: fail | 52: pass | No loss | your decision |
| Inkling-2-Mini | Full model | 43.7 GiB | Reference | 49 | Reference | |
| | K 112 of 128 | 42.7 GiB | +2.14%: pass | 51: pass | No loss | pass |

- A prune of the experts frees a small part of the memory. The +3% limit of `nll` holds a cut of about 12% to 15%
  of the experts of these two models.
- A cut can fail `nll` and still show no loss in `tasks` and `judge`. The report then says `your decision`.
- prune-kit ran on two model types and on one machine. Each new model type needs a profile and a test.

## Scope

- **MoE models only.** prune-kit prunes the routed experts of a mixture-of-experts model. It does not prune a dense
  model.
- The weights must be safetensors files. A folder of GGUF files gets `NO-GO`.
- It does not train, and it does not quantize.
- It does not deploy. The switch of a live model server stays a step by hand.

## Requirements

- Linux, Python 3.11 or later, and [uv](https://docs.astral.sh/uv/).
- The preflight needs only the files of a model: a folder, or a model in the local Hugging Face cache. It needs no
  GPU and no server.
- The capture, the gates and the probe need a vLLM server that returns the routed experts (vLLM 0.33, with the flag
  `--enable-return-routed-experts`), and the memory for the full model.
- The probe needs a profile for the model type. Qwen3.5 and Qwen3.6 MoE have one, and Inkling 2 with a loop has one.

### What belongs to the machine of the author

- The files in `examples/` describe one NVIDIA DGX Spark with 121 GiB of unified memory: its paths, its two live
  model servers and their hours. Write your own machine file and your own workflow file from them.
- "The record" and "the prune by hand" are the two earlier prunes. They are in the public repo
  [dsv4-specialize](https://github.com/czhou578/dsv4-specialize).
- The gate `tasks` runs the tool-call suite of the public repo
  [model-benchmarks](https://github.com/czhou578/model-benchmarks).
- `model-gateway` and the command `job` are private tools of the author. The source kind `gateway_records` reads the
  records of that gateway. `job` only shows a run on a page: each script also runs with no `job` command.

## Install

```bash
uv sync
```

## Use

Start with the quick check. It reads only the files of a model, and it loads no weights.

```bash
uv run prune-kit preflight thinkingmachines/Inkling-2-Mini
```

`MODEL` is a folder, a Hugging Face name in a local cache, or a part of such a name. The verdict is `GO`,
`GO WITH WORK` with the parts that need work, or `NO-GO` for a model with no routed experts.

`prune-kit stages` prints the stages of a prune. [docs/reference.md](docs/reference.md) describes each command.

| Stage | Command | Question |
|---|---|---|
| preflight | `prune-kit preflight MODEL` | How much memory does each cut free, and can the pipeline run on this model? |
| data | `prune-kit data FILE` | Which recorded sessions are the workflow, and which sessions are held out? |
| capture | `prune-kit capture MODEL` | Which experts does the model route the workflow to? |
| rank | `prune-kit rank CAPTURE` | Which experts of each layer does the workflow use least? |
| screen | By hand | Which cuts pass a short replay? |
| prune | `prune-kit build MODEL` | Does the pruned folder hold the kept experts, byte for byte? |
| gates | `prune-kit gate`, `prune-kit judge` | Does the cut pass the replay, the tools, the tasks, the judge, the speed and the memory? |
| decision | By hand | Which cut do you choose? |

Two more commands run the measurements on test servers:

- `prune-kit probe MODEL` writes the window file of a probe. A probe measures two or three cuts in one window, before
  a full capture.
- `prune-kit window FILE` runs a window file. A window is a time in which test servers run in place of a live model
  server. The runner restores each live model server at the end.

- CAUTION: A real window stops a live model server, and that server gives no answers until the restore. Run each
  window with `--dry-run` first.
- CAUTION: The data of a workflow (`sessions.jsonl`) can hold private text of your transcripts. Keep it out of git:
  `runs/` is in `.gitignore`.

## Tests

```bash
uv run pytest -q
```

The tests load no model, send no request, and stop no server. Some tests read files of the machine of the author,
and each of them is skipped where its files do not exist: a new clone gives 138 passed and 7 skipped.

## License

[MIT](LICENSE)
