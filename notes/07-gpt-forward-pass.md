# 07 — Inside the GPT forward pass

> Companion experiment: [experiments/attention_by_hand.py](experiments/attention_by_hand.py). Run it from the repo root:
> `.venv\Scripts\python.exe -m notes.experiments.attention_by_hand` (with `$env:TORCHDYNAMO_DISABLE="1"` set).
> All outputs below come from running it on 2026-10-23. Code refs are `nanochat/gpt.py` at commit `92d63d4`.
> Prerequisite: section 3 of [01-fundamentals.md](01-fundamentals.md).

The model is one function: token ids `(B, T)` in, logits `(B, T, vocab)` out. B is the batch size (how many
sequences at once), T the sequence length, C the model width (`n_embd`).

## 1. The forward pass, top to bottom (`GPT.forward`, L459–524)

Shapes for two models: the d4 from the CPU runs, and the d24 that `runs/speedrun.sh` trains (per GPU).

| Step | Code | d4 shape (B=1, T=12) | d24 shape (B=16, T=2048) |
|---|---|---|---|
| token ids | `idx` | (1, 12) | (16, 2048) |
| embed + norm | `x = norm(wte(idx))` L471–473 | (1, 12, 256) | (16, 2048, 1536) |
| smear | L476–480 | same | same |
| 4 / 24 blocks | L499–504 | same | same |
| backout + final norm | L506–508 | same | same |
| logits | `lm_head(x)` L512 | (1, 12, 32768) | (16, 2048, 32768) |
| loss (training only) | `cross_entropy` L520 | scalar | scalar |

The width C stays the same all the way through the blocks. That is the **residual stream**. Only the two ends
change shape: the embedding maps ids to vectors, `lm_head` maps vectors to vocabulary scores. The d24 logits
tensor holds 16 × 2048 × 32768 ≈ 1.07 billion numbers (4.3 GB in fp32), which is why the comment at L512 calls
it a "very big tensor".

Step by step:

1. **Rotary slice** (L467–468). Grab the precomputed cos/sin for positions `T0 … T0+T`. `T0` is 0 in
   training; during generation it is how many tokens are already in the KV cache.
2. **Embed and normalise** (L471–473). Look up each token's vector, then RMS-normalise it.
3. **Smear** (L476–480). Mix a gated fraction of the *previous* token's embedding into each position:
   `x[t] += gate[t] * x[t-1]`. A cheap way to give every position bigram information ("what came just before
   me") before any attention happens. `smear_lambda` starts at 0, so this is switched off at init and learned.
4. **The blocks** (L499–504). Before block *i*: `x = resid_lambdas[i]*x + x0_lambdas[i]*x0`. That rescales
   the stream and blends a little of the original embedding `x0` back in. Alternate layers also get a **value
   embedding** `ve`: a second embedding table whose vectors are added to the attention values (L94–97).
5. **Backout** (L506–507). Subtract a learned fraction of the stream as it was at the middle layer
   (`n_layer // 2`). The comment says this removes low-level features before predicting.
6. **Head** (L510–515). Final norm, `lm_head`, crop the vocabulary padding, cast to fp32, then **softcap**:
   `15 * tanh(logits / 15)` keeps every logit inside (−15, 15).
7. **Loss** (L517–521). With `targets`, return cross-entropy. `ignore_index=-1` means targets set to −1 are
   skipped; SFT uses this to train only on assistant tokens. Without targets, return the logits.

## 2. A Block (L144–153)

```python
x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
x = x + self.mlp(norm(x))
```

Pre-norm residual: normalise a *copy* for the sublayer, add the result back to the unnormalised stream.

## 3. Causal self-attention (L84–128)

Shapes for the d4 (C=256, 4 heads of 64):

| Line | Code | Shape |
|---|---|---|
| L89–91 | `q = c_q(x).view(B, T, n_head, head_dim)`, same for k, v | (1, 12, 4, 64) |
| L94–97 | value residual: `v = v + gate * ve`, gate ∈ (0, 3) per head | (1, 12, 4, 64) |
| L100–101 | RoPE on q and k | same |
| L102–104 | QK norm, then ×1.2 each | same |
| L110 | `flash_attn_func(q, k, v, causal=True, window_size=...)` | (1, 12, 4, 64) |
| L126–127 | glue heads back together, `c_proj` | (1, 12, 256) |

The `c_q` hook in the experiment shows (1, 12, 256) out. `.view` reshapes that into 4 heads × 64 without
copying. Each head runs attention independently, and `c_proj` mixes the heads' outputs back together.

On CPU, `flash_attn_func` falls back to PyTorch's `scaled_dot_product_attention`
(`nanochat/flash_attention.py:115-136`), transposing to the `(B, H, T, D)` layout PyTorch expects.

### The math with real numbers (experiment section 1)

One head, 4 tokens, head_dim 2. Each row is one token.

```
scores = Q @ K^T / sqrt(d)           after the causal mask
[[0.707, 0.000, 0.707, 0.000],       [[0.707,  -inf,  -inf,  -inf],
 [0.000, 0.707, 0.707, 1.414],        [0.000, 0.707,  -inf,  -inf],
 [0.707, 0.707, 1.414, 1.414],        [0.707, 0.707, 1.414,  -inf],
 [1.414, 0.000, 1.414, 0.000]]        [1.414, 0.000, 1.414, 0.000]]
```

- Row *i* says how much token *i* "likes" each token *j*. The mask sets every *j > i* (the future) to −∞.
- Token 1 likes token 3 most (1.414), but token 3 is in its future, so the mask removes it.

```
weights = softmax(row-wise)          output = weights @ V      (V rows: [10,0] [0,10] [5,5] [1,1])
[[1.000, 0.000, 0.000, 0.000],       [[10.000, 0.000],
 [0.330, 0.670, 0.000, 0.000],        [ 3.302, 6.698],
 [0.248, 0.248, 0.503, 0.000],        [ 5.000, 5.000],
 [0.402, 0.098, 0.402, 0.098]]        [ 6.131, 3.087]]
```

- `exp(−∞) = 0`, so masked positions get exactly zero weight, and each row sums to 1.
- **Token 0 can only see itself**, so its output is exactly its own value `[10, 0]`.
- Every output is a weighted average of the value vectors of the tokens it may see.
- Checked against PyTorch: `max |by hand - PyTorch SDPA| = 4.8e-07` (float rounding).

## 4. RMSNorm (experiment section 2)

```
x                          rms     x / rms                    nanochat norm(x)
[3.0, 4.0, 0.0, 0.0]       2.500   [1.200, 1.600, 0.0, 0.0]   [1.200, 1.600, 0.0, 0.0]
[0.1, 0.2, 0.3, 0.4]       0.274   [0.365, 0.730, 1.095, 1.461]  (same)
```

Two vectors at very different scales both come out with RMS exactly 1.000. The *direction* is kept and the
*size* is reset. Without this, the residual stream's magnitude would drift as layers keep adding to it.
nanochat's `norm` (L42) has no learnable scale, unlike textbook LayerNorm.

## 5. RoPE (experiment section 3, `apply_rotary_emb` L57–65)

RoPE splits each head vector into pairs of numbers and rotates each pair by an angle of `position × frequency`.
Frequencies range from fast (1 radian per position) to very slow (`1/100000^(…)`), like the hands of a clock
running at different speeds (L276–282).

The rotated dot product depends only on the **distance** between positions:

```
plain q.k (no rotation): -0.5261
q at position  3, k at position  1 (distance 2): score +1.7298
q at position 10, k at position  8 (distance 2): score +1.7298
q at position 40, k at position 38 (distance 2): score +1.7298
q at position 10, k at position  1 (distance 9): score +1.2168
q at position 50, k at position 41 (distance 9): score +1.2168
rotation keeps lengths: |q| = 3.3953  |rotated q| = 3.3953
```

So attention learns patterns like "look 2 tokens back" that work anywhere in the sequence. Rotations don't
change vector length, so RoPE adds position information without distorting magnitudes.

## 6. The MLP (L131–141)

`c_fc` (256 → 1024), `relu(x)²`, `c_proj` (1024 → 256). The experiment's hooks show (1, 12, 1024) in the
middle. The nonlinearity is what lets the network compute anything a stack of matrix multiplies couldn't.
Without it, the whole MLP would collapse into one linear map.

## 7. Causality, tested on the real model (experiment section 4)

(The numbers below are from the 40-step `d4` checkpoint, the only one that existed when this was written. The
experiment now picks `d4-500` if present, so a rerun shows the 500-step numbers that are compared in
[04](04-cpu-run-500-steps.md) section 4. To reproduce these, set `tag = "d4"` in the script.)

Edit one token in the middle of a sentence and compare the logits at every position:

```
tokens A: ['<|bos|>', 'The', ' cat', ' sat', ' on', ' the', ' mat', ' and', ' looked', ' at', ' the', ' door']
tokens B: ['<|bos|>', 'The', ' cat', ' sat', ' on', ' dog', ' mat', ...]   (position 5 edited)

d4 after 40 steps:
position  0-4: max |logits A - logits B| = 0.000000
position  5:   1.137737   <- edited here
position  6:   0.771335
position  7:   0.018057
position  8-11: 0.007 to 0.021
```

- **Positions before the edit are bit-for-bit identical.** That is causality: a prediction at position *t* can
  only depend on tokens 0…*t*, so there's no peeking at the future, even through the extras like smear.
- **What changes after the edit tells you how much the model uses context.** At 40 steps, position 6 (right
  after the edit) changes a lot and later positions barely change. This barely trained model predicts mostly
  from the current and previous token, close to a bigram model. The 500-step comparison is in
  [04-cpu-run-500-steps.md](04-cpu-run-500-steps.md).

## 8. All shapes (experiment section 5, d4)

```
transformer.wte (embedding)    in (1, 12)            -> out (1, 12, 256)
h[0].attn.c_q                  in (1, 12, 256)       -> out (1, 12, 256)
h[0].attn.c_k                  in (1, 12, 256)       -> out (1, 12, 256)
h[0].attn.c_v                  in (1, 12, 256)       -> out (1, 12, 256)
h[0].attn.c_proj               in (1, 12, 256)       -> out (1, 12, 256)
h[0].mlp.c_fc                  in (1, 12, 256)       -> out (1, 12, 1024)
h[0].mlp.c_proj                in (1, 12, 1024)      -> out (1, 12, 256)
lm_head                        in (1, 12, 256)       -> out (1, 12, 32768)
```

## 9. Check your understanding: count the parameters yourself

The d4 run printed `transformer_matrices: 3,145,824`, `value_embeds: 16,777,216`, `scalars: 34`, and
`Estimated FLOPs per token: 7.549819e+07`. Can you reproduce them? Answers below.

<details>
<summary>Answers</summary>

- **Per block**: attention has `c_q, c_k, c_v, c_proj`, each 256×256 → 4 × 65,536 = 262,144. The MLP has
  256×1024 + 1024×256 = 524,288. Total 786,432 per block (12·C²), × 4 blocks = **3,145,728**.
- **ve_gate**: `has_ve` (L53) is true for layers 1 and 3 when n_layer=4. Each `ve_gate` is `Linear(12, 4)`
  = 48 → 96. 3,145,728 + 96 = **3,145,824** ✓.
- **value_embeds**: 2 layers × 32,768 × 256 = **16,777,216** ✓.
- **scalars**: resid_lambdas 4 + x0_lambdas 4 + smear_gate 24 + smear_lambda 1 + backout_lambda 1 = **34** ✓.
- **FLOPs per token** (`estimate_flops`, L319): matmul params = 3,145,824 + lm_head 8,388,608 + smear_gate 24 =
  11,534,456, × 6 = 69,206,736. Attention adds 12 × 4 heads × 64 × 512 context = 1,572,864 per layer, × 4 =
  6,291,456. Total **75,498,192 ≈ 7.5498e7** ✓.

</details>

## Exercises

1. In section 1 of the experiment, set `is_causal=False` in the PyTorch call and drop the mask. What does token 0
   output now?
2. Change `head_dim` in section 3 to 64. Does the distance property still hold?
3. Rerun section 4 on the 500-step model and compare how far the edit's effect reaches.
