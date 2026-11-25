# The blueprint of a prune

This page is the method of the two prunes of `spark-moe-specialize` (DeepSeek V4 Flash and Qwen3.6-35B-A3B), as one
checklist. Use it for a prune by hand, or to read what a prune-kit command does. `prune-kit stages` shows which
stages have a command.

## In simple words

- A mixture-of-experts (MoE) model holds many experts in each layer, and each token uses only a few of them.
- Your work uses some experts often and other experts almost never.
- A prune records which experts your work uses, and removes the cold experts from the files.
- The model then needs less memory. Tests against the full model show what the cut costs.
- You choose the cut. A test result is evidence, not the decision.

## Before you start

- [ ] The model has routed experts. A dense model has no expert lever, and the removal of layers failed on DeepSeek.
- [ ] You can name the workflow: the kind of work that the model does for you.
- [ ] Recorded traffic of that workflow exists. The two prunes used 1.75M and 2.44M tokens.
- [ ] You have a memory goal: the peak that the model can use, at the context that you need.
- [ ] The full model fits in memory alone. The capture and the reference need it.

## The stages

Do the stages in this order. Do not start a stage before the stage above it meets its exit.

### 0. Preflight

- **Do:** read `config.json` and the headers of the weight files. Count the layers, the experts per layer and the
  experts per token. Compute the bytes of one expert.
- **Exit:** a table of the memory that each K frees, and a list of the parts that the model needs: a prune tool for
  its layout, a capture on its server, and a rule for its draft model.

### 1. Data

- **Do:** build the sessions of the workflow in one format. Split them into a calibration set and a held-out set.
- **Exit:** the calibration set has enough tokens, and the held-out set has at least 60 target turns.
- The output can hold secrets. Keep it local and out of git.

### 2. Capture

- **Do:** boot the full model with the capture of the routed experts. Send the calibration set.
- **Exit:** each segment has its counts for each layer, and no row is empty. A shared prefix counts once.
- On vLLM 0.33 the capture is the flag `--enable-return-routed-experts`. DeepSeek needed a server patch.

### 3. Rank

- **Do:** order the experts of each layer by their use in the capture.
- **Exit:** one rank file. Record the share of the routing that each K step removes.

### 4. Screen

- **Do:** build a cheap copy of each candidate point, and run the replay on it. A mask of the router is the
  cheapest copy. Where a prune is a copy of bytes, a prune is cheap enough.
- **Exit:** the replay result of each point, against the reference of the same window.

### 5. Prune

- **Do:** build the folder of the chosen point. Copy the bytes of the kept experts, give them new ids, and change
  the router rows and the config in the same order.
- **Exit:** a check compares each kept tensor with its source, byte for byte. A test boot gives the replay result
  of the screen, inside the noise.

### 6. Gates

Run the gates in this order. A point that fails a gate does not run the next gate, unless you decide it.

| Gate | Measure | Limit of the two prunes |
|---|---|---|
| Replay NLL | The NLL of 60 held-out turns that the reference wrote | At most +3% |
| Tools | On the same turns: the tool names, and the first answer token | Names at least 98%; decisions down by at most 3 |
| Tasks | A suite of tool-call tasks or agent tasks | At most 3 tasks fewer |
| Speed | The decode speed with the draft model | Report |
| Memory | The peak at the context of the workflow | The margin of the memory goal |
| Judge | A blind judge and your own blind review of pairs of answers | No loss that the test shows |

### 7. Decision

- **Do:** read the report. Choose a point, or choose no prune.
- **Exit:** a record of the choice with its date and its reason. Then plan the deploy as its own step.

## The rules

Each rule comes from a result of the two prunes.

- **R1. Use the workload's own rank.** With the public rank, DeepSeek K 192 failed (+3.31%). With the rank of the
  bot's own traffic, K 176 cost +0.29%.
- **R2. Screen with cheap points first.** A mask builds in a minute. Build a costly point only for the winner.
- **R3. Measure the reference in the same window.** One replay moves up to 0.56% between runs. A point within 0.5%
  of a limit gets a second replay, and the mean decides.
- **R4. Keep the gate order.** A broad NLL loss shows in long sessions before it shows in a short suite.
- **R5. Read a judge result as "no loss", not as a gain.** Most wins against an old baseline come from newer data.
- **R6. Test the speed with the draft model.** A smaller K lowered the acceptance of the DeepSeek draft, and the
  decode got slower. Fewer experts per token gave the speed back.
- **R7. Record the free swap with the available memory.** The kernel moves idle server pages to swap before the
  available memory falls.
- **R8. Write a new folder.** A weight tool never changes its source.
- **R9. Hold out whole sessions.** The calibration set never sees a held-out session or a held-out project.

## What the two prunes found

| | DeepSeek V4 Flash | Qwen3.6-35B-A3B |
|---|---|---|
| Experts per layer, experts per token | 216, 6 | 256, 8 |
| The point that you chose | 140 to 160 per layer, 5 per token | 208 per layer |
| Loaded weights | 95.39 GiB to 69.79 GiB (−27%) | 21.22 GiB to 17.01 GiB (−20%, with the NVFP4 draft) |
| Replay NLL of the chosen point | Inside the +3% gate | +4.15%: fails the gate; a blind judge finds no loss |
| The floor | K 144 fails with each rank | The gate holds only above about K 224 |

- **The depth is not known before a measurement.** The plan of the Qwen prune expected that K 192 passes. K 208
  already failed.
- **A small model loses more.** At the same share of the routing, Qwen lost about twice as much NLL as DeepSeek.
- **More cuts did not work without training:** fewer experts per token, the removal of whole layers, and cold
  cuts past the chosen point.
