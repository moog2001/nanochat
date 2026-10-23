# nanochat study notes

Notes from learning ML / LLM training with this repo as the textbook. Started 2026-10-23 at commit `92d63d4`.
Everything here is local to this machine (Windows 11, CPU-only PyTorch). Guides cite `file:line` in the repo.

## Guides, in reading order

| # | Guide | What it covers | Hands-on |
|---|---|---|---|
| 01 | [Fundamentals](01-fundamentals.md) | What an LM is, training, the Transformer (RMSNorm, attention, RoPE, MLP), Muon/AdamW, the full pipeline, scaling laws | – |
| 02 | [base_train.py walkthrough](02-base-train-walkthrough.md) | The pretraining script phase by phase, with verified details from the files it calls | – |
| 03 | [First CPU run: 40 steps](03-cpu-run-40-steps.md) | Machine setup, Windows gotchas, annotated log | [log](logs/cpu-run-d4-40steps.log) |
| 04 | [CPU run: 500 steps](04-cpu-run-500-steps.md) | All 3 LR phases, samples improving, causality before/after, a controlled A/B experiment | [log](logs/cpu-run-d4-500steps.log), [momentum_ab.py](experiments/momentum_ab.py), [compare_logs.py](experiments/compare_logs.py) |
| 05 | [Tokenizer](05-tokenizer.md) | BPE, what the vocabulary learned, numbers/Korean, special tokens, the SFT loss mask | [tokenizer_tour.py](experiments/tokenizer_tour.py) |
| 06 | [Data pipeline](06-data-pipeline.md) | Shards → documents → best-fit packed (x, y) batches, measured crop waste | [inspect_data.py](experiments/inspect_data.py) |
| 07 | [GPT forward pass](07-gpt-forward-pass.md) | Shapes through the model, attention/RMSNorm/RoPE by hand, causality test | [attention_by_hand.py](experiments/attention_by_hand.py) |
| 08 | [Optimizers](08-optimizers.md) | AdamW (first step = sign descent), Muon's orthogonalization, cautious weight decay, the real param groups, ZeRO-2 | [optimizer_by_hand.py](experiments/optimizer_by_hand.py) |
| 09 | [Inference](09-inference.md) | Next-token probabilities, temperature/top-k, greedy loops, KV cache, the tool-use state machine | [sampling.py](experiments/sampling.py) |
| 10 | [SFT](10-sft.md) | How chat finetuning differs from pretraining, the data mixture, GSM8K tool calls, MMLU formatting | [sft_data.py](experiments/sft_data.py) |
| 11 | [Evaluation](11-evaluation.md) | bpb, CORE (scoring by loss), ChatCORE, centered accuracy | [core_scoring.py](experiments/core_scoring.py) |
| 12 | [RL](12-rl.md) | GRPO-as-REINFORCE on GSM8K, advantages, why RL needs SFT first, pass@k | [rl_toy.py](experiments/rl_toy.py) |

New to all this? Read 01, then 07, then 02 and 03–04. After that, follow the pipeline: 05 → 06 → 08 → 09 → 10 → 11 → 12.

## Where we left off (2026-10-23)

Done: guides 01–12, two CPU training runs (40 and 500 steps) plus a controlled A/B rerun, a real CORE
evaluation (d4-500 scores 0.043 vs GPT-2's 0.257), and an experiment script for every topic from tokenizer to
RL. Not done: SFT and RL weren't actually run (they need more data downloads, and the base model is tiny). The
exercises at the end of each guide are untried.

Suggested next steps, roughly in order:
1. Run `attention_by_hand` and `tokenizer_tour` yourself and do their exercises. Reading output you produced
   beats reading mine.
2. Do one guide's "Exercises" per sitting. Most need only a one-line edit to an experiment.
3. A longer CPU run, e.g. the 500-step command with `--num-iterations=2000` (~1 hour), to see how far a d4 gets.
4. Try SFT on `d4-500` (sketch in [10](10-sft.md) section 6) to watch a base model learn the chat format.
5. Fix the Windows calculator bug in `nanochat/engine.py` ([09](09-inference.md) section 5). It's a good
   first code change.
6. Read `dev/LOG.md`, the repo's lab notebook of experiments, including the ones that didn't work.

## Running things on this machine

From the repo root in PowerShell:

```powershell
$env:PYTHONIOENCODING = "utf-8"   # needed when output is piped/redirected (console code page is cp949)
$env:TORCHDYNAMO_DISABLE = "1"    # no C++ compiler (cl.exe) here, so torch.compile must be off
.venv\Scripts\python.exe -m notes.experiments.attention_by_hand
```

- Experiments run as modules (`-m notes.experiments.<name>`), like the repo's own `scripts`, because the
  `nanochat` package isn't installed into the venv. Running them by file path fails to import it.
- Tiny training run (≈2.4K tokens/s, ~1.7 s per step at these settings):
  `.venv\Scripts\python.exe -m scripts.base_train --depth=4 --head-dim=64 --window-pattern=L --max-seq-len=512 --device-batch-size=4 --total-batch-size=4096 --eval-every=100 --eval-tokens=16384 --core-metric-every=-1 --sample-every=100 --num-iterations=500 --model-tag=d4-500`
- Details and reasons: [03-cpu-run-40-steps.md](03-cpu-run-40-steps.md).

## What's on disk (outside the repo)

`~/.cache/nanochat/` (`C:\Users\UOS\.cache\nanochat\`):

| Path | Size | What |
|---|---|---|
| `base_data_climbmix/` | 439 MB | 4 training shards + the validation shard |
| `tokenizer/` | 0.5 MB | BPE tokenizer trained on 200M chars, plus `token_bytes.pt` |
| `base_checkpoints/d4/` | 409 MB | the 40-step model |
| `base_checkpoints/d4-500/` | 409 MB | the 500-step model (used by the experiments) |
| `base_checkpoints/d4-500-momfix/` | 409 MB | the A/B variant from guide 04 |
| `eval_bundle/` + `eval_bundle.zip` | 162 + 25 MB | CORE task data (the zip isn't needed once unpacked) |
| `base_eval/` | tiny | CORE results CSV |
| `task_data/` | 2.3 MB | GSM8K train + MMLU dev, downloaded by `sft_data.py` |

Each checkpoint is `model_*.pt` (147 MB) + `optim_*_rank0.pt` (281 MB, needed only to resume training) +
`meta_*.json`. To free about 0.8 GB, delete `base_checkpoints/d4/` and `base_checkpoints/d4-500-momfix/`.
Nothing depends on them; the experiments use `d4-500`.

This `notes/` folder isn't tracked by git. To hide it from `git status`, add a line `notes/` to
`.git/info/exclude` (local-only; not done yet).

## Corrections log

Things I said in chat that turned out wrong when checked against the code. Details are in each guide.

- The sliding-window "S" layers see **512** tokens of a 2048 context (a quarter), not ~768. The comment at
  `gpt.py:302` and the `--window-pattern` help text are stale.
- SFT trains on a SmolTalk + MMLU + GSM8K **mixture**, not only SmolTalk.
- GQA is supported but **off** in default training (`n_kv_head = n_head`).
- The dataloader's "~35% cropped at T=2048" doesn't match this data: about 12% at T=2048 and 35% at T=512
  (measured in 06).

## Things noticed in the repo

Code quirks found while writing these guides. None affect real GPU runs much, but they're good reading
practice, and a couple would make easy first contributions:

| Where | What | Guide |
|---|---|---|
| `nanochat/engine.py:30-42` | The calculator timeout uses `signal.SIGALRM`, so tool use raises `AttributeError` on Windows | [09](09-inference.md) |
| `nanochat/engine.py:302-352` | The self-test calls `torch.cuda.synchronize()`, so it can't run CPU-only | [09](09-inference.md) |
| `nanochat/gpt.py:302`, `scripts/base_train.py:54` | Comment says 2048 → 768 and help text says "half"; the code gives a quarter (512) | [01](01-fundamentals.md) |
| `nanochat/dataloader.py:8` | "~35% cropped at T=2048"; this data measures ~12% (35% at T=512) | [06](06-data-pipeline.md) |
| `scripts/base_train.py:372-382` | In runs under ~1,140 steps, Muon momentum jumps 0.970 → 0.922 at step 400. A controlled A/B showed it's harmless (−0.0007 bpb, within noise) | [04](04-cpu-run-500-steps.md) |
| `nanochat/gpt.py:444` | The smear/backout LR is hard-coded (0.2) and skips the batch/width LR scaling | [08](08-optimizers.md) |

## Resources

- DeepWiki for this repo: https://deepwiki.com/karpathy/nanochat (recommended in the README; it rate-limited my fetch)
- Karpathy, "Let's build GPT: from scratch, in code, spelled out" (YouTube)
- Vaswani et al. 2017, "Attention Is All You Need"; Hoffmann et al. 2022, Chinchilla (arXiv 2203.15556)
- Muon: https://kellerjordan.github.io/posts/muon/ (linked from `nanochat/optim.py`)
- DCLM paper for the CORE metric: arXiv 2406.11794
- The repo's own lab notebook: `dev/LOG.md`, which records experiments that worked and that didn't
