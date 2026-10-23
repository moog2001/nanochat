# 08 — The optimizers: AdamW and Muon

> Companion experiment: [experiments/optimizer_by_hand.py](experiments/optimizer_by_hand.py)
> (`.venv\Scripts\python.exe -m notes.experiments.optimizer_by_hand`, with `TORCHDYNAMO_DISABLE=1`).
> Code: `nanochat/optim.py` and `GPT.setup_optimizer` (`nanochat/gpt.py:419-457`) at commit `92d63d4`.
> Outputs are from runs on 2026-10-23.

Backprop gives a gradient for every weight: the direction that increases the loss fastest. The optimizer
decides **how far to move each weight**, using that gradient plus some memory of past gradients. Plain SGD
moves `lr × gradient`. Momentum moves along a running average of gradients, which smooths out noise. Adam and
Muon go further and **rescale** the step.

## 1. AdamW (`adamw_step_fused`, `optim.py:24-63`)

For each weight, keep two running averages:

```
m = beta1 * m + (1 - beta1) * g        # average gradient       ("which way")
v = beta2 * v + (1 - beta2) * g²       # average squared grad   ("how big, usually")
m̂, v̂ = m / (1 - beta1^t), v / (1 - beta2^t)    # bias correction: both start at 0
p = p * (1 - lr * wd)                   # decoupled weight decay (the "W" in AdamW)
p = p - lr * m̂ / (sqrt(v̂) + eps)
```

Dividing by `sqrt(v̂)` gives each weight a step size of about `lr`, **whatever scale its gradients have**.
The first step shows this most clearly (experiment section 1):

```
gradient          [ 0.3000, -0.0010,  5.0000, -0.0200]
update (nanochat) [-0.0100,  0.0100, -0.0100,  0.0100]     <- identical by hand
-lr * sign(grad)  [-0.0100,  0.0100, -0.0100,  0.0100]
```

A gradient of 5.0 and one of 0.001 produce the same step. Adam's first step is **sign descent**. Later steps
are smoother because `m` and `v` average over time. This per-weight normalisation is why Adam copes with
parameters whose gradients differ by orders of magnitude, like embeddings vs. scalar gates.

nanochat details:
- `beta1` is **0.8**, a shorter memory than the textbook 0.9, with beta2 between 0.95 and 0.995 per group, and
  `eps = 1e-10`.
- Everything runs in one `torch.compile`d function, so there's no Python overhead between the operations.
  Hyperparameters are passed as 0-D CPU tensors (`optim.py:262-272`), so changing the learning rate every step
  doesn't trigger a recompile.

## 2. Muon (`muon_step_fused`, `optim.py:112-180`)

Muon is for the **2D weight matrices inside the blocks** (attention q/k/v/proj, MLP fc/proj). Its core idea:
a weight matrix's gradient is itself a matrix, and usually a few directions dominate it. Muon replaces the
update with the **nearest orthogonal matrix**, which keeps the directions and makes every one of them the same
size.

Experiment section 2 builds a gradient-like 64×128 matrix with four strong directions (50, 10, 3, 1) and 60 weak
ones (0.05), and runs nanochat's Polar Express iteration on it:

```
after 0 iterations: largest singular values [0.979, 0.196, 0.059, 0.020], smallest 0.001
after 1 iterations: largest singular values [1.421, 1.012, 0.470, 0.158], smallest 0.008
after 2 iterations: largest singular values [1.710, 1.620, 0.627, 0.581], smallest 0.032
after 5 iterations: largest singular values [1.122, 1.111, 1.080, 1.005], smallest 0.839

cosine similarity of the 5-step result with the exact U V^T: 0.9976
share of the raw update along the strongest direction:  95.8%
share after orthogonalization:                            2.7%   (perfectly equal would be 1/64 = 1.6%)
```

- A raw gradient step would spend 95.8% of its "length" on one direction. After orthogonalization the weak
  directions get roughly the same step as the strong ones, so the model learns them too.
- **Polar Express** (`optim.py:100-108, 143-154`) is a polynomial iteration, `X ← aX + (bA + cA²)X` with
  `A = XXᵀ`. Each step pushes all singular values toward 1. The five coefficient triples are tuned for exactly
  5 steps, which is why the intermediate steps overshoot (1.7 after two) and only the fifth lands near 1. It
  uses only matrix multiplies, so it runs fast in bf16 on GPUs, unlike an exact SVD.

The full Muon step, in order:

| Lines | Step | What it does |
|---|---|---|
| 130–133 | Nesterov momentum | Average gradients over steps (`momentum` from the schedule, 0.85 → 0.97 → 0.90). |
| 138–141 | MuonEq | Rescale each row to the mean row norm, so the matrix entering orthogonalization is better conditioned. |
| 143–154 | Polar Express | Orthogonalize (section 2 above). |
| 158–161 | Muon+ | Snap the overall size to √min(rows, cols), the size of an exactly orthogonal matrix. |
| 163–174 | NorMuon | Per-row (or per-column) adaptive scaling from a running average of squared updates (beta2 = 0.9), Adam-like. Orthogonalized updates can have uneven per-neuron scales. |
| 176–180 | cautious weight decay + update | Decay only the weights where the update already pushes toward zero (below). |

**Cautious weight decay** (experiment section 3). The update is `params -= lr * update`. Decay is applied only
where `update` and `params` have the **same sign**, i.e. where the gradient step already shrinks the weight:

```
params [[ 0.5, -0.5],     update [[ 0.1,  0.1],     decayed? [[ True, False],
        [ 0.5, -0.5]]             [-0.1, -0.1]]              [False,  True]]
```

Where the gradient wants the weight to grow, decay would fight it, so it is skipped.

Two more details:
- Tall matrices get a bigger step: `lr × sqrt(max(1, rows/cols))` (`optim.py:393`). The MLP's `c_fc` weight is
  (1024, 256), so it steps 2× larger than the square attention matrices.
- **Never use Muon for embeddings, the output head, or 0-D/1-D parameters** (`optim.py:198-200`). An embedding
  "matrix" is really 32,768 separate vectors, and only the rows for tokens in the batch get gradients, so
  orthogonalizing it as a whole makes no sense. Those go to AdamW.

## 3. Who gets which optimizer: the real groups (experiment section 4)

The d4 model's groups, with the learning rates the CPU runs used:

```
adamw lr 0.00122   1 tensor   8,388,608 params  (32768, 256)  lm_head.weight         betas (0.8, 0.96)
adamw lr 0.04593   1 tensor   8,388,608 params  (32768, 256)  transformer.wte.weight betas (0.8, 0.995)
adamw lr 0.02296   2 tensors 16,777,216 params  (32768, 256)  value_embeds.*         betas (0.8, 0.995)
adamw lr 0.00044   1 tensor           4 params  (4,)          resid_lambdas          betas (0.8, 0.95)
adamw lr 0.04419   1 tensor           4 params  (4,)          x0_lambdas             betas (0.96, 0.95)
adamw lr 0.20000   3 tensors         26 params  (1, 24)...    smear_gate, smear_lambda, backout_lambda
muon  lr 0.00177   2 tensors         96 params  (4, 12)       ve_gate
muon  lr 0.00177  16 tensors  1,048,576 params  (256, 256)    attn c_q, c_k, c_v, c_proj (x4 layers)
muon  lr 0.00177   4 tensors  1,048,576 params  (256, 1024)   mlp c_proj
muon  lr 0.00177   4 tensors  1,048,576 params  (1024, 256)   mlp c_fc
```

How each learning rate was built (`base_train.py:287-316`, `gpt.py:432-452`):

- **Batch scaling**, all groups but one: × √(4096 / 524,288) = 0.0884. The CPU batch is 128× smaller than the
  reference.
- **Width scaling**, AdamW groups: × √(768 / 256) = 1.732, because the AdamW learning rates were tuned at width
  768. E.g. `lm_head`: 0.008 × 0.0884 × 1.732 = 0.00122.
- Per-group tweaks: `resid_lambdas` × 0.01, `value_embeds` × 0.5.
- **The smear/backout group is hard-coded at `lr=0.2`** (`gpt.py:444`). It gets neither the batch nor the
  width scaling, so in the CPU runs it learned about 11× (1 / 0.0884) faster *relative to everything else* than
  it would at the reference batch size. That fits [04](04-cpu-run-500-steps.md): `smear_lambda` had already shot up to 2.73
  by step 40. That's a reading of the code, not something I tested.
- Muon groups are formed **by shape** (`gpt.py:447-452`), so same-shaped matrices can be stacked and processed
  in one batched call.

## 4. Multiple GPUs: the optimizer does the communication (`optim.py:184-259`, `428-459`)

There is no PyTorch DDP. `MuonAdamW.step()` works in three phases so communication overlaps compute:

1. **Launch all gradient reductions** asynchronously. Small tensors (< 1,024 elements) are `all_reduce`d:
   averaged, with everyone getting the full result. Large ones are `reduce_scatter`ed: averaged, and each GPU
   receives only **its slice**.
2. **Each GPU updates only its slice**, and keeps optimizer state (m, v, momentum buffers) only for that slice.
   This is "ZeRO-2": optimizer memory is split N ways. Then it launches `all_gather` to share the updated
   weights.
3. **Wait for the gathers** and copy the results into place.

On one device (this machine) all communication is skipped and the code reduces to plain full-tensor updates
(`optim.py:279-281, 298-301`).

## 5. Schedules come from outside

The optimizer itself has no schedule. Before every step `base_train.py:520-527` overwrites `lr` (warmup →
constant → warmdown), Muon's `momentum` (0.85 → 0.97 → 0.90), and Muon's `weight_decay` (cosine to 0). See
[02](02-base-train-walkthrough.md), Phase 5. That includes the momentum jump at step 400 in short runs, which
[04](04-cpu-run-500-steps.md) tests.

## Exercises

1. In section 1, run a second Adam step with a different gradient. Is it still pure sign descent? (No: `m` and
   `v` now average the two gradients.)
2. In section 2, set `steps=3` or `steps=7`. Why does going *past* 5 not help? (The coefficient list has only
   5 entries; look at `polar_express_coeffs[:steps]`.)
3. Why would a learning rate that ignores batch-size scaling matter more for a tiny CPU batch than for the
   speedrun's 1M-token batch? (Compare the scale factors 0.088 and √2.)
