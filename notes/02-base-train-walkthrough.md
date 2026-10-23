# 02 — `scripts/base_train.py`, walked through

> Written 2026-10-23 at commit `92d63d4`. The file is a script, not a library: it runs top to bottom in 7
> phases. The real training loop is lines 416–594; everything before it is setup.
> Notes marked **Verified** were checked against the other source files after the session.

## Phase 0 — Imports and flags (L1–81)

- **L15** `expandable_segments:True` — a PyTorch GPU-allocator setting that reduces memory fragmentation. It has
  to be set before `import torch`, which is why it sits above the other imports.
- **L24–36** The imports are the pipeline: `GPT` (model), `dataloader` (feeds tokens), `Engine` (generates
  samples), `evaluate_bpb` and `evaluate_core` (the two metrics), `checkpoint_manager` (save/load).
- **L41–80** argparse flags, in five groups:
  - **Architecture** — `--depth` is the main dial. `--aspect-ratio` sets `model_dim = depth × 64`. `--head-dim`
    is 128.
  - **Training horizon** — three ways to say how long to train, with precedence `--num-iterations` >
    `--target-flops` > `--target-param-data-ratio`.
  - **Optimization** — one learning rate per parameter group, plus weight decay, warmup and warmdown.
  - **Eval and logging** — how often to run each check.
  - **FP8** — 8-bit matmuls for H100-class GPUs.
  - Heads-up: the `--window-pattern` help text (L54) says "S=half context". The code gives a **quarter**
    (see the corrections in [01-fundamentals.md](01-fundamentals.md)).

## Phase 1 — Device and distributed setup (L85–117)

- **L86** `compute_init` returns `ddp, ddp_rank, ddp_local_rank, ddp_world_size, device`. With
  `torchrun --nproc_per_node=8`, this script runs as 8 identical processes, one per GPU, and each one knows its
  rank (0–7).
  - **Verified** (`nanochat/common.py:174-209`): it seeds torch with 42, so weight init is reproducible. On
    CUDA it enables TF32 matmuls. It only creates a process group when launched by torchrun **and** on CUDA;
    on CPU you always get a single process.
- **L87** `master_process = rank == 0`. Only rank 0 logs, prints (`print0`) and samples. All ranks still take
  part in the evals (Phase 6).
- **L88–96** Look up the GPU's peak FLOPS so MFU can be computed later. MFU is the fraction of the hardware's
  theoretical speed you actually use. On CPU it is set to `inf`, so MFU prints as 0.
- **L99–100** `DummyWandb` is a no-op logger, so later code can call `wandb_run.log(...)` unconditionally.
- **L102–117** Check for Flash Attention 3. Without it the code falls back to PyTorch SDPA, which emulates
  sliding windows with a slow explicit mask, hence the warning to use `--window-pattern L`.

## Phase 2 — Tokenizer and model construction (L121–162)

- **L121–124** Load the tokenizer. The model needs `vocab_size` to size its embedding table.
- **L129–143** `build_model_meta` shows how one `--depth` number determines the whole model:
  - `model_dim = depth × 64`, rounded up to a multiple of 128 so `head_dim == 128` exactly.
  - `num_heads = model_dim / 128`. Example: depth 20 → dim 1280 → 10 heads.
  - `n_kv_head = num_heads` (L138), so GQA is off.
  - `torch.device("meta")` builds the model with shapes only and no memory allocated.
- **L146–151** A three-step init:
  1. Build on the meta device (free, so parameter counts can be inspected cheaply).
  2. `to_empty(device)` allocates real but uninitialised memory on the target device.
  3. `init_weights()` fills it in.

  This avoids allocating the model twice, and avoids a slow init on CPU followed by a copy to the GPU.
- **L154–162** Resume support. With `--resume-from-step`, load the saved weights (`load_state_dict(...,
  assign=True)`) **and** the optimizer state. That is why a resume continues smoothly instead of restarting
  momentum from zero.

## Phase 3 — FP8 and compile (L168–246)

- **L168–192** With `--fp8`, each big `Linear` is swapped for a `Float8Linear`, so matmuls run in 8-bit floats
  (roughly 2× throughput on H100s). The filter skips layers whose dims aren't divisible by 16 (a hardware
  requirement) or are smaller than 128 (too small to benefit).
- **L195–240** `disable_fp8` is a context manager. During evaluation it temporarily swaps the FP8 layers back to
  normal `Linear`, so the eval numbers stay accurate. The weights are shared, not copied, so it costs no memory.
- **L245–246** `orig_model` keeps the uncompiled model, used for checkpoints (compiled models prefix their
  state_dict keys with `_orig_mod.`) and for sampling (input shapes vary there). `torch.compile(model,
  dynamic=False)` fuses operations into faster kernels. `dynamic=False` is safe because training batches
  always have the same shape.
  - **Verified**: the optimizer's step functions are compiled too, by decorator (`nanochat/optim.py:23` and
    `:111`). On a machine without a C++ compiler, `TORCHDYNAMO_DISABLE=1` turns **every** `torch.compile`
    into a no-op. That is how the CPU runs in these notes work (see [03-cpu-run-40-steps.md](03-cpu-run-40-steps.md)).

## Phase 4 — Hyperparameters from scaling laws (L252–304)

The most interesting part. It works out how long to train, how big the batch should be, and what the learning
rate and weight decay should be. The reference point is a hand-tuned d12 model: you only ever tune d12 by hand,
and everything else is extrapolated (muP-style).

| Step | Lines | What |
|---|---|---|
| Count params and FLOPs | 252–258 | `num_scaling_params()`, `estimate_flops()` |
| Training horizon | 263–269 | `target_tokens = ratio × (transformer matrices + lm_head)`. With `--target-param-data-ratio=8`, a model with 1B scaling params trains on 8B tokens. Embeddings are left out because that gives the cleanest scaling laws (`dev/LOG.md`, Jan 27 2026). |
| Reference run | 272–274 | The d12 token budget (`D_REF`) and best batch size (`B_REF = 2^19 ≈ 524K tokens`), both measured empirically |
| Batch size | 279–284 | `B ∝ D^0.383` (Power Lines paper). Bigger runs get bigger batches, rounded to a power of 2. Skipped if you pass `--total-batch-size`. |
| LR scaling | 287–294 | `lr ∝ √(B / B_ref)`. A bigger batch has less gradient noise, so you can take bigger steps. |
| Weight decay scaling | 302 | `λ = λ_ref · √(B/B_ref) · (D_ref/D)`, keeping the "T_epoch" quantity constant. The comments admit this is AdamW theory applied to Muon on faith. |

Note: `target_tokens` is computed from the ratio even when `--num-iterations` overrides the horizon, so the
weight-decay scaling still uses it.

## Phase 5 — Optimizer, data and schedules (L308–386)

- **L308–316** `setup_optimizer` (in `gpt.py:419`) builds the Muon + AdamW combination. Every learning rate
  is multiplied by `batch_lr_scale`.
- **L324** `GradScaler` is only for fp16, whose narrow range lets gradients underflow to zero. bf16 has fp32's
  range, so it doesn't need one.
- **L331–333** The data loaders yield `(x, y)`, where `y` is `x` shifted one token to the left, so `y[t]` is the
  "next token" label for `x[t]`. `next(train_loader)` is called once up front to prime the pipeline.
  - **Verified** (`nanochat/dataloader.py`):
    - Train split = every shard except the last; val = the last shard (L38).
    - Each document is tokenized with `<|bos|>` prepended. Rows of `T+1` tokens are filled by **best-fit
      packing**: repeatedly insert the largest buffered document that still fits, and when nothing fits, crop
      the shortest one to fill the row exactly (L122–151). No padding. At T=2048 about 35% of tokens are
      cropped away (docstring).
    - `x = row[:-1]`, `y = row[1:]` (L154–155).
    - Multi-GPU: rank r reads row groups r, r+N, r+2N, ... (L62–68).
    - On CUDA, one pinned-memory, non-blocking host→device copy per batch.
- **L340–357** Resolve `num_iterations` by the precedence above. The usual path is `target_tokens //
  total_batch_size`. Then print total tokens, tokens-per-param and total FLOPs.
- **L360–369** LR schedule, three phases:
  1. Linear warmup over `--warmup-steps` (40). Full LR on an untrained model can blow up.
  2. Constant at full strength.
  3. Linear decay over the last 65% (`--warmdown-ratio`) down to 5% of peak (`--final-lr-frac`).

  The warmup check comes first, so a run of 40 steps or fewer is *all* warmup.
- **L372–382** Muon momentum ramps 0.85 → 0.97 over a fixed 400 steps, holds, then falls to 0.90 during warmdown.
  - Quirk for short runs: the ramp branch (`it < 400`) is checked first, so when warmdown starts before step
    400 (runs under ~1,140 steps), momentum keeps ramping up and then *jumps* down at step 400. In the 500-step
    CPU run it goes 0.970 at step 399 → 0.922 at step 400. Real runs have thousands of steps and never hit this.
- **L385–386** Weight decay follows a cosine from full strength down to 0.

All three schedules are functions of the step number. Inside the loop they overwrite each param group's
values before every step.

## Phase 6 — The training loop (L392–594)

### Loop state (L392–413)
Either a fresh start or values restored from a checkpoint.

**Gradient accumulation** (L407–413). Suppose the target batch is 1M tokens per step, but one GPU only fits
16 × 2048 tokens at once. Run several forward/backward passes, let the gradients add up, then take a single
optimizer step:

```
tokens per micro-batch (all GPUs) = device_batch_size × seq_len × n_gpus
grad_accum_steps                  = total_batch_size / that
```

Example: 8 GPUs at 16 × 2048 → 262,144 tokens per micro-batch, so a 1,048,576-token batch needs 4 accumulation
steps. Mathematically the same as one big batch.

### Periodic duties at the top of each iteration (L417–499)
`while True` runs `num_iterations + 1` times so the final step (`last_step`) can still evaluate and save.

- **Validation bpb (L421–436)** — run the model on held-out data. `model.eval()`, FP8 off. All ranks
  participate, because the sums are combined across GPUs.
  - **Verified** (`nanochat/loss_eval.py:9-65`): bpb = (sum of per-token losses in nats) ÷ (ln 2 × total
    bytes of those target tokens). Special tokens like `<|bos|>` have byte length 0 and are left out. Because
    it is normalised by bytes, models with different tokenizers can be compared.
- **CORE (L442–453)** — downstream tasks (`nanochat/core_eval.py`, from the DCLM paper). Uses `orig_model`,
  because every example has a different length and would trigger recompiles.
- **Sampling (L457–474)** — complete 7 fixed prompts ("The capital of France is…") greedily
  (`temperature=0`). A sanity check: the loss can look fine while the output is garbage. Rank 0 only.
- **Checkpoint (L477–499)** — save weights, optimizer state, the dataloader position and loop state, so a
  resume continues exactly where it stopped.
  - **Verified** (`nanochat/checkpoint_manager.py:41-57`): rank 0 writes `model_<step>.pt` and
    `meta_<step>.json`. **Every** rank writes its own `optim_<step>_rank<r>.pt`, because optimizer state is
    sharded. Sizes in the d4 CPU run: model 147 MB = 36.7M params × 4 bytes (fp32). Optimizer 281 MB, mostly
    AdamW's two moment buffers for the 33.5M embedding parameters (33.5M × 2 × 4 bytes ≈ 268 MB).

### The step itself (L508–543)

```python
for micro_step in range(grad_accum_steps):
    loss = model(x, y)                 # forward pass
    loss = loss / grad_accum_steps     # so accumulated grads form an average, not a sum
    loss.backward()                    # backward: gradients ADD into each param's .grad
    x, y, ... = next(train_loader)     # fetch the next batch
```

`next(train_loader)` sits right after `backward()`. CUDA calls are asynchronous, so the GPU is still working
through the backward pass while the CPU loads and tokenizes the next batch. This hides data-loading time.

Then the update:

1. **L520–527** Compute the three schedule values for this step and write them into each param group. This is
   how schedules work here: the optimizer knows nothing about them, you just overwrite `lr`, `momentum` and
   `weight_decay` before each step. `initial_lr` was stored in `setup_optimizer`.
2. **L528–539** `optimizer.step()` applies the updates. The `scaler` branch is the fp16-only path. With
   multiple GPUs, the "did any rank see inf/NaN" flag must be agreed across ranks, or they would diverge.
3. **L540** `zero_grad(set_to_none=True)` clears the gradients. Forget this and gradients pile up across steps.
   `set_to_none` frees the memory instead of filling it with zeros.
4. **L541–542** `.item()` copies a scalar from GPU to CPU, and `synchronize()` waits for the GPU. Both make
   `dt` measure the real step time rather than the time to *queue* the work.

**Where do gradients get averaged across GPUs?** Not in this file, and there is no `DistributedDataParallel`
wrapper. **Verified**: `MuonAdamW.step` (`nanochat/optim.py:429`) does it. Small tensors are `all_reduce`d
(averaged); large ones are `reduce_scatter`ed so each GPU receives the averaged gradient for **its slice**,
updates only that slice, then `all_gather`s the updated weights back to everyone.

### Logging (L548–580)
- **L548–550** The raw loss is noisy from batch to batch, so it is smoothed with an exponential moving
  average. The average starts at 0, so early values are biased low; dividing by `1 − β^(step+1)` corrects
  that. Adam does the same trick internally.
- **L552–554** `tok/sec` is throughput. `mfu` is achieved FLOPs ÷ peak FLOPs. `dev/LOG.md` records around
  35–46% on H100 runs. Low MFU means the GPU is waiting on something.
- **L555–565** Time and ETA only count after step 10, because the first steps include compile time and warmup.
- **L568** Log to wandb every 100 steps.

### The garbage-collector hack (L586–594)
Python's GC kept scanning for reference cycles, costing about 500 ms at random moments, which shows up as
spikes in step time. So after the first step the script does:
- `gc.collect()` to clear setup garbage,
- `gc.freeze()` to exclude every surviving object from future scans,
- `gc.disable()` to turn it off entirely, with a manual collect every 5000 steps.

That is only reasonable because the loop creates almost no cyclic garbage.

## Phase 7 — Wrap-up (L597–604)
Print peak memory and total time, finish the wandb run, tear down the process group.

## The whole thing in one picture

```
CLI args → pick device/ranks → tokenizer → build model on meta → allocate → init
        → (fp8 swap) → torch.compile
        → count params → target_tokens → batch size → LR/WD scaling
        → optimizer, dataloaders, schedules
        → loop:
            [periodic: val bpb, CORE, samples, checkpoint]
            repeat grad_accum_steps: forward → loss → backward
            set lr/momentum/wd from schedules → optimizer.step → zero_grad
            log
```

## Takeaways

- **One knob, everything derived.** Tune once at d12; the code extrapolates the rest.
- **Training is a loop.** The mathematical core is `loss = model(x, y); loss.backward(); optimizer.step()`. The
  other ~590 lines are engineering: scale, precision, resumability and monitoring.
- **Evaluation is separate from the training loss.** Val bpb, CORE and sample text each catch different failure
  modes.
