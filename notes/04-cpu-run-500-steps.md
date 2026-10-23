# 04 — CPU run: the d4 model for 500 steps

> 2026-10-23. Raw log: [logs/cpu-run-d4-500steps.log](logs/cpu-run-d4-500steps.log). Same settings as
> [03](03-cpu-run-40-steps.md) except `--num-iterations=500 --eval-every=100 --sample-every=100 --model-tag=d4-500`.
> 13.5 minutes of training, 2.05M tokens (0.18 tokens per scaling parameter, still ~50× short of a real run).

## 1. The learning-rate schedule, now visible in full

| step | 0 | 10 | 20 | 30 | 39 | 40–175 | 180 | 200 | 300 | 400 | 499 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lrm` | 0.03 | 0.28 | 0.53 | 0.78 | 1.00 | 1.00 | 0.99 | 0.93 | 0.63 | 0.34 | 0.05 |

Warmup (steps 0–39), constant (40–175), then linear warmdown over the last 65% to 5% of the peak. Exactly what
`get_lr_multiplier` (`base_train.py:360-369`) computes for 500 iterations: warmdown = round(0.65 × 500) = 325
steps, starting after step 175.

## 2. Loss and validation bpb

| step | 0 | 20 | 30 | 40 | 50 | 100 | 150 | 200 | 300 | 400 | 499 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train loss (smoothed) | 10.40 | 10.29 | 9.69 | 8.68 | 8.01 | 7.00 | 6.69 | 6.63 | 6.24 | 6.18 | 6.07 |

| step | 0 | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|
| val bpb | 3.142 | 2.059 | 1.938 | 1.877 | 1.829 | 1.807 |
| improvement | | −1.083 | −0.121 | −0.061 | −0.048 | −0.022 |

What to notice:

- **The big drop is right after warmup.** Steps 20–100 take the loss from 10.3 to 7.0. That's the model
  learning word frequencies and the most common pairs. After that every 100 steps buys less, the classic shape
  of a loss curve: fast at first, then slow.
- **Loss and bpb are the same quantity in different units.** bpb ≈ loss ÷ (ln 2 × bytes per token) =
  6.07 ÷ (0.693 × 4.69) ≈ 1.87, close to the measured 1.81. They differ because val bpb leaves out `<|bos|>`
  targets and uses different text than the smoothed training loss.
- **Same start as the 40-step run.** Steps 0–10 match the 40-step run to 6 decimals (10.396724, 10.396898, …).
  That's the fixed seed (42) and identical data order. They drift apart slightly from about step 12, because the
  weight-decay cosine (`base_train.py:385`) depends on the total number of steps (40 vs 500). Determinism is
  what makes the A/B experiment in section 6 possible.

## 3. The samples, every 100 steps (greedy decoding)

```
step 100  The capital of France is a few of the world of the of the of the of the of the of
step 200  The capital of France is the of the of the of the of the of the of the of the of
step 300  The capital of France is the most of the same time. The first of the same time, the same
step 400  The capital of France is a very common in the most of the most of the most of the most of
step 500  The capital of France is a very common part of the most of the most of the most of the most

step 300  If yesterday was Friday, then tomorrow will be a few times. I have a few times, but I have a few times
step 500  The opposite of hot is a good way to the right way to make a few of the same time.
step 500  My favorite color is a lot of people who are not a lot of people who are not a lot

step 200  If 5*x + 3 = 13, then x is 20 20 20 20 20 20 20 20
step 300  If 5*x + 3 = 13, then x is 2020, 2020, 2020, 2020,
step 400  If 5*x + 3 = 13, then x is a 2. 2. 2. 2. 2.
step 500  If 5*x + 3 = 13, then x is a 1.2.2. 1.2. 2.
```

- **From word salad to phrases.** At step 100 it's "of the of the". By 300–500 it produces real English chunks:
  "at the same time", "a very common part of", "a lot of people who are not". Grammar arrives before meaning.
- **Repetition loops** are what greedy decoding (`temperature=0`) does with a weak model. Once a phrase is the
  most likely continuation of itself, the model is stuck. Sampling with temperature breaks loops
  ([09](09-inference.md) will show this).
- **The math prompt** shows tokenization at work ([05](05-tokenizer.md)): "20" is a very common two-digit
  token, because years split as 20|19, 20|26. So step 200 answers "20", and step 300 produces whole years,
  "2020". Step 500 has picked up that a short number often follows "x is". It has no idea about algebra.
  Scale check: this run used 7.55e7 FLOPs/token × 2.048M tokens ≈ 1.5e14 FLOPs. The README calls the GPT-2-grade
  speedrun a "4e19 FLOPs" model, about **260,000×** more compute.

## 4. Does the model use context now? (causality test, [07](07-gpt-forward-pass.md) section 7)

The same edit as before: change `' the'` → `' dog'` at position 5 of "The cat sat on the mat and looked at the
door", then measure how much the logits change at each later position.

| position | 5 (edited) | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|
| 40 steps | 1.14 | 0.77 | 0.018 | 0.021 | 0.010 | 0.011 | 0.007 |
| 500 steps | 5.46 | 1.00 | 1.53 | 0.78 | 1.61 | 1.29 | 0.81 |

Positions 0–4 stay exactly 0 in both (causality). After 40 steps the edit barely reaches past the next token:
the model was close to a bigram model. After 500 steps the edit changes predictions all the way to the end of
the sentence, so the model is now using attention to look back.

The learned scalars point the same way (read straight from the checkpoints):

| | `smear_lambda` | attention `c_proj` weight std (layers 0–3) |
|---|---|---|
| step 40 | 2.73 | 0.0016, 0.0016, 0.0016, 0.0017 |
| step 500 | 0.46 | 0.0060, 0.0060, 0.0068, 0.0066 |

Early on, the model leaned heavily on **smear**, the cheap "mix in the previous token" shortcut (`gpt.py:476-480`).
Then it turned smear down as the attention output projections, which start at exactly zero, grew about 4×. A
plausible reading: grab the easy bigram signal first, then learn real context. (Two snapshots only, so treat it
as a story the numbers are consistent with, not a proof.)

## 5. A code quirk this run exercises: the Muon momentum jump

`get_muon_momentum` (`base_train.py:372-382`) checks the 400-step warm-up branch before the warmdown branch. In
a run shorter than ~1,140 steps, warmdown has already started by step 400, so momentum ramps 0.85 → 0.970 until
step 399, then **jumps** to 0.922 at step 400 ([02](02-base-train-walkthrough.md), Phase 5).

The smoothed loss right after step 400:

```
step 395-404:  6.18 6.20 6.17 6.18 6.19 6.18 6.15 6.15 6.16 6.14
step 405-414:  6.17 6.22 6.21 6.28 6.30 6.30 6.31 6.29 6.26 6.24
step 415-425:  6.25 6.26 6.27 6.26 6.27 6.24 6.24 6.21 6.19 6.21 6.17
```

A bump of about +0.16 starts at step 405 and takes ~20 steps to recover. The printed loss is a moving average,
so a cause at step 400 would show up a few steps late. But the log has other bumps of +0.05 to +0.07 from
ordinary batch noise, so the timing alone proves nothing. Section 6 tests it.

## 6. A/B experiment: is the jump to blame?

**Design.** [experiments/momentum_ab.py](experiments/momentum_ab.py) runs `scripts/base_train.py` with one change,
patched in memory (the repo file is untouched): from step 400 on, momentum glides smoothly from 0.97 to 0.90
instead of jumping to 0.922. Same flags, seed and data order. Raw log:
[logs/cpu-run-d4-500steps-momfix.log](logs/cpu-run-d4-500steps-momfix.log). Compared with
[experiments/compare_logs.py](experiments/compare_logs.py):

```
identical loss up to step 400; first difference at step 401

 step         A (jump)  B (smooth)  B - A
  404    6.1441    6.1444  +0.0002
  406    6.2172    6.2175  +0.0003
  408    6.2820    6.2822  +0.0002
  410    6.2995    6.2997  +0.0001
  411    6.3089    6.3089  -0.0000      <- peak of the bump, in BOTH runs
  415    6.2547    6.2549  +0.0001
  420    6.2434    6.2431  -0.0003
  425    6.1748    6.1746  -0.0002
mean (B - A) over steps 401-499: -0.0012

val bpb at step 500:  A 1.806699   B 1.805992   (-0.0007)
```

**Verdict: the jump is not to blame.** The bump happens identically without it, differing by at most 0.0003.
Both runs train on exactly the same batches, so the bump comes from **the data**: the batches around steps
405–412 are harder text. The printed loss is measured on the current batch, so harder text shows up as higher
loss whatever the optimizer does. The smooth schedule ends 0.0007 bpb better, a difference too small to
distinguish from noise with a single run per arm.

**Lessons.**
- **Timing is not causation.** The bump started right after a real code quirk, and it still had nothing to do
  with it.
- **Determinism makes A/B tests cheap and clean.** Identical up to step 400 means any later difference can only
  come from the change. Without a fixed seed and data order you would need many runs per arm to see anything.
- **A tiny effect needs repeated runs.** To claim the 0.0007 improvement is real, you'd rerun each arm with
  several seeds and compare the spread. That is the discipline behind the README leaderboard and `dev/LOG.md`.
