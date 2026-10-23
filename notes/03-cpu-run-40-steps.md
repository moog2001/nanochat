# 03 — First CPU run: a d4 model for 40 steps

> 2026-10-23. Raw output: [logs/cpu-run-d4-40steps.log](logs/cpu-run-d4-40steps.log).
> Headline: validation bpb fell 3.14 → 2.34 in under a minute, but the samples are still "the the the".

## What was set up on this machine

| Thing | Where | Notes |
|---|---|---|
| Data | `~/.cache/nanochat/base_data_climbmix/` (439 MB) | `-n 4` gave 4 training shards plus the validation shard, which is always the last one |
| Tokenizer | `~/.cache/nanochat/tokenizer/` | Trained on 200M characters (BPE training itself took 6.45 s). `runs/runcpu.sh` uses 2B; I cut it 10× to save time. `tok_eval` was not run. |
| Checkpoint | `~/.cache/nanochat/base_checkpoints/d4/` (409 MB) | `model_000040.pt` 147 MB, `optim_000040_rank0.pt` 281 MB, `meta_000040.json` |

Two Windows-specific gotchas, both confirmed in this session:

1. **No C++ compiler (`cl`)**, so `torch.compile` crashes at the first forward pass with
   `InductorError: RuntimeError: Compiler: cl is not found.` Fix: `TORCHDYNAMO_DISABLE=1`, which turns every
   `torch.compile` into a no-op, including the decorated optimizer kernels in `nanochat/optim.py`. It costs
   speed, not correctness. (Installing the Visual Studio C++ build tools would likely make compile work;
   untested.)
2. **Console code page cp949.** When Python's output is piped or redirected, it encodes with cp949, which can't
   encode the startup banner's `█` and crashes with `UnicodeEncodeError`. Fix: `PYTHONIOENCODING=utf-8`. Not
   needed when printing straight to an interactive terminal.

## Commands

PowerShell (your main shell). The `$env:` settings last for the session:

```powershell
cd C:\moog\studies\nanochat
$env:PYTHONIOENCODING = "utf-8"
$env:TORCHDYNAMO_DISABLE = "1"
.venv\Scripts\python.exe -m nanochat.dataset -n 4
.venv\Scripts\python.exe -m scripts.tok_train --max-chars=200000000
.venv\Scripts\python.exe -m scripts.base_train --depth=4 --head-dim=64 --window-pattern=L --max-seq-len=512 --device-batch-size=4 --total-batch-size=4096 --eval-every=20 --eval-tokens=16384 --core-metric-every=-1 --sample-every=40 --num-iterations=40
```

Git Bash equivalent: `export PYTHONIOENCODING=utf-8 TORCHDYNAMO_DISABLE=1`, then the same commands with
`.venv/Scripts/python.exe`. Data and tokenizer only need to be done once.

## Reading the setup lines

```
Model config: n_layer 4, n_head 4, n_embd 256
```
Depth 4 × aspect ratio 64 = 256 dims, ÷ head-dim 64 = 4 heads (`build_model_meta`, `base_train.py:129-143`).

```
wte 8.4M | value_embeds 16.8M | lm_head 8.4M | transformer_matrices 3.1M | total 36.7M
```
A tiny transformer under a huge vocabulary. The three tables with 32,768 rows hold 91% of the parameters. That is
why `get_scaling_params` (`base_train.py:263`) counts only `transformer_matrices + lm_head` (11.5M here): it is
the cleaner measure of model size.

```
Scaling LRs by 0.0884 for batch size 4,096 (reference: 524,288)
```
The batch is 128× smaller than the reference, so every LR is multiplied by `√(1/128) = 0.0884`
(`base_train.py:293`). Small batches give noisy gradients, so steps must be smaller.

```
Scaling the LR for the AdamW parameters ∝1/√(256/768) = 1.732051
```
A second, separate correction (`gpt.py:433`): AdamW learning rates were tuned at width 768. This model is 3×
narrower, so they are scaled **up** by √3.

```
Tokens : Scaling params ratio: 0.01
```
163,840 training tokens against 11.5M scaling params. Real runs use 8–12 tokens per parameter, so this is
roughly 1000× undertrained.

```
Total batch size 4,096 => gradient accumulation steps: 2
```
4 sequences × 512 tokens = 2,048 per micro-batch, so 2 forward/backward passes fill one optimizer step
(`base_train.py:407-413`).

## Reading the training log

**`loss: 10.396724` at step 0.** That is `ln(32768) = 10.3972`, the loss of a model guessing uniformly among
all 32,768 tokens. It confirms the init (`gpt.py:221`): `lm_head` starts with std 0.001, so every logit is
near zero and softmax is flat. **Rule: the first-step loss of an LLM should be about ln(vocab_size). If it
isn't, something is wrong.**

**`lrm: 0.03, 0.05, 0.07 …`** The warmup in `get_lr_multiplier` is `(step+1)/40`. `--warmup-steps` defaults to
40 and we trained 40 steps, so the whole run was warmup. It never reached the constant or warmdown phases.

**Flat for ~15 steps, then falling fast** (10.39 → 10.31 around steps 15–18, then about 0.1 per step). The LR
started tiny (`lrm` 0.03–0.3 times an already scaled-down base), so the first steps barely moved anything.

**A wobble at step 19** (10.26 → 10.31). One noisy batch showing through. The printed `loss` is the debiased
moving average (`base_train.py:548-550`), not the raw per-step loss.

**`dt ≈ 1.7 s`, `tok/sec ≈ 2,350`.** CPU, no compile. The real speedrun processes millions of tokens per second
on 8 H100s.

**`bf16_mfu: 0.00`, `Peak memory usage: 0.00MiB`.** CPU placeholders: peak FLOPs is set to `inf`
(`base_train.py:95`) and the memory query is a CUDA-only stub.

**`epoch: 1 pq: 0 rg: 1`.** The dataloader's position: epoch, parquet file index, row group within it. Saved in
the checkpoint so a resume continues at the same place in the data.

**`total time: 0.00m` for the first 11 steps.** The timer only starts after step 10 (`base_train.py:555`), so
the ETA is blank until then.

## Validation bpb: 3.14 → 3.05 → 2.34

Bits per byte is the loss re-expressed per byte of original text (`nanochat/loss_eval.py`).

- **Step 0 decodes the tokenizer.** A uniform guess over 32,768 tokens costs log2(32768) = **15 bits per token**.
  15 ÷ 3.14 ≈ **4.8 bytes per token**, so this tokenizer packs about 4.8 bytes of text into each token on the
  validation shard. (A uniform guess over the 256 byte values would score 8 bpb. Guessing at the token level is
  already better, because each token covers several bytes.)
- **Where it should end up:** GPT-2-grade nanochat models reach about 0.72 (README leaderboard). At 2.34 we are
  at the "learned which words are common" stage.

## The samples

```
The capital of France is the the the the the the ...
```

Expected after 40 steps. At a loss of about 8.8 nats the model has learned one thing: "the" is the most common
token. Greedy decoding (`temperature=0`) always takes the top token, so it repeats "the" forever. This is why
the script samples fixed prompts every N steps: the loss can look fine while generation is broken, and the
same prompts let you watch quality change over time. The 500-step run in
[04-cpu-run-500-steps.md](04-cpu-run-500-steps.md) shows what comes next.

## What this run didn't show

- The constant-LR and warmdown phases, because the whole run was warmup.
- CORE (`--core-metric-every=-1`, turned off to keep the run short).
- Mid-run checkpoints. It saved once at the end.
- Anything multi-GPU: one process, so all the distributed code paths were skipped.
