# 01 — ML and nanochat fundamentals

> Written 2026-10-23 while studying nanochat at commit `92d63d4`. Line numbers point at that commit.
> Ordered so each section builds on the one before. Corrections made after checking the code are listed at the end.

## 1. What a language model is

A language model is a function: given a sequence of tokens, output a probability distribution over the
**next** token. That is the whole job. "Chatting" is a loop: sample a token from that distribution, append
it to the input, repeat. The loop is `GPT.generate` (`nanochat/gpt.py:527`).

- **Token** — a chunk of text, usually a word or a piece of one. `scripts/tok_train.py` trains the tokenizer
  (BPE, byte-pair encoding). nanochat's vocabulary has 32,768 tokens (2^15).
- **Embedding** — a lookup table from token id to a vector of numbers (`wte`, `gpt.py:174`). Everything after
  this point is math on vectors.
- **Logits** — raw scores, one per vocabulary entry. Softmax turns them into probabilities. `lm_head`
  (`gpt.py:177`) produces them.

## 2. What "training" means

The model is a large pile of numbers called **parameters** (or weights). Training adjusts them so the model
predicts text better:

1. Take a chunk of text, e.g. `"The cat sat on the"`.
2. Run it through the model. At every position it predicts the next token.
3. **Loss** — compare each prediction with the real next token using cross-entropy (`gpt.py:520`).
   Cross-entropy is −ln(probability the model gave the correct token), averaged over positions. It is high when
   the model gave the right answer low probability.
4. **Backprop** — PyTorch works out, for every weight, which direction to nudge it to reduce the loss. That
   direction is the **gradient**.
5. **Optimizer** — applies the nudges. Plain SGD does `w -= lr * grad`. Adam adds momentum and a per-weight
   step size.
6. Repeat, for billions of tokens.

Terms you will keep seeing:

- **Learning rate (lr)** — the step size.
- **Batch size** — how much data the gradient is averaged over per step. nanochat counts it in tokens.
- **Epoch** — one pass over the dataset. LLM pretraining usually does less than one.
- **bf16 / fp8** — low-precision number formats. They make GPUs faster at the cost of some numerical care.
  On CPU nanochat uses plain fp32 (the run log prints `COMPUTE_DTYPE: torch.float32`).

## 3. The Transformer, piece by piece

The model is a stack of identical **Blocks** (`gpt.py:144`):

```python
x = x + self.attn(norm(x), ...)   # tokens exchange information
x = x + self.mlp(norm(x))         # each token processes on its own
```

### Residual stream (the `x = x + ...`)
Think of `x` as a shared notepad that runs through the whole network. Each layer reads it, computes something,
and **adds** its result back; it never overwrites the notepad. This is what makes deep networks trainable:
gradients flow straight down the `+` path. It is also why `c_proj` is initialised to zeros (`gpt.py:230`):
every block starts as a no-op and learns to contribute.

### LayerNorm / RMSNorm
Activations drift to huge or tiny magnitudes as they pass through layers, which makes training unstable.

- **LayerNorm** subtracts the mean, divides by the standard deviation, then applies a learned scale and shift.
- **RMSNorm** (what nanochat uses, `gpt.py:42`) skips the mean subtraction: `x / sqrt(mean(x²))`. Simpler, and
  works as well. nanochat also drops the learned scale ("no learnable params in rmsnorm", `gpt.py:9`).

The norm is applied *before* attention and the MLP ("pre-norm"), the standard modern arrangement.

### Causal self-attention (`gpt.py:67`)
The one mechanism that moves information **between** tokens. For each token, three linear layers (`c_q`,
`c_k`, `c_v`) make three vectors from the same input:

- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I contain?"
- **Value (V)** — "what I hand over if you attend to me."

Token *i*'s attention weights over other tokens are `softmax(Q_i · K_j / √d)`. Its output is the weighted
average of the V vectors.

- **Causal** — token *i* may only look at tokens *j ≤ i*. It can't peek at the future, which would be cheating
  when the task is predicting the future. This is also what lets you train on every position of a sequence at
  once.
- **Multi-head** — the vector is split into `n_head` independent groups. Each head can learn a different
  pattern ("look at the previous word", "find the matching bracket", ...).

Things in this file you will meet in any modern LLM:

- **RoPE, rotary embeddings** (`gpt.py:57`, frequencies at `gpt.py:270`) — attention by itself has no sense of
  word order. RoPE rotates Q and K by an angle that depends on position, so the Q·K dot product depends on the
  *relative* distance between tokens.
- **QK norm** (`gpt.py:102`) — normalises Q and K so attention scores can't blow up.
- **GQA, grouped-query attention** (`n_kv_head`) — several query heads share one K/V head, which shrinks the KV
  cache. The model supports it, but `scripts/base_train.py:138` sets `n_kv_head = n_head`, so it is **off** in
  the default training runs.
- **KV cache** — when generating, the K and V of past tokens never change, so you store them instead of
  recomputing them. That is the `flash_attn_with_kvcache` call at `gpt.py:114`.
- **Sliding window** (`window_pattern="SSSL"`, `gpt.py:287`) — "S" layers only see the last **quarter** of the
  context: 512 tokens at the default 2048 context. Every 4th layer, and always the last layer, sees the full
  context. Cheaper, and works about as well.
- **Flash Attention** — a fused GPU kernel that computes the same attention math without ever materialising
  the T×T score matrix. Purely a speed and memory trick. Without FA3 (e.g. on CPU) nanochat falls back to
  PyTorch's SDPA and emulates sliding windows with an explicit mask (`nanochat/flash_attention.py:77-110`),
  which is slow, so CPU runs use `--window-pattern=L` (full context everywhere).

### MLP (`gpt.py:131`)
Expand the vector 4×, apply a nonlinearity (`relu(x)²`), project back down. Attention mixes information
**across** tokens; the MLP processes **each** token on its own, and is thought to be where much of the
"knowledge" is stored. Per block the MLP has 8·d² weights against attention's 4·d², so it holds about two
thirds of each block's parameters. (In tiny models the embedding tables dominate the *total* count: in the d4
CPU run, 91% of all parameters were `wte` + `value_embeds` + `lm_head`.)

### The two ends
Token ids → embedding table → N blocks → final norm → `lm_head` → a score for every vocabulary entry. The
`softcap` line (`gpt.py:515`) squashes logits with `tanh` into [−15, 15] so they can't get extreme.

### nanochat's extras
`resid_lambdas`, `x0_lambdas`, `smear`, `backout` and `value_embeds` are small tricks from the modded-nanogpt
speedrun community. Each buys a little more quality per unit of compute. Skip them on a first read: the core
architecture is complete without them.

## 4. The optimizer: Muon + AdamW

`setup_optimizer` (`gpt.py:419`) splits parameters into groups and gives them two different optimizers
(`nanochat/optim.py`):

- **AdamW** (`optim.py:24`) for embeddings, `lm_head` and the scalar parameters. Standard Adam with decoupled
  weight decay. Note the betas, e.g. `(0.8, 0.96)`: beta1 is lower than the textbook 0.9.
- **Muon** (`optim.py:112`) for the 2D weight matrices inside the blocks. In plain words: take the
  momentum-smoothed gradient matrix and replace it with the *nearest orthogonal matrix* — keep its directions,
  equalise their strengths — then step. nanochat computes this with 5 iterations of a polynomial ("Polar
  Express", a tuned variant of the Newton–Schulz iteration; `ns_steps=5`). It then applies a per-row/column
  adaptive scale (NorMuon) and **cautious weight decay**: weights are only decayed where the decay agrees
  with the update direction (`optim.py:179`). In practice Muon trains these matrices faster than Adam.
- **Multi-GPU**: gradient averaging happens **inside this optimizer**, not through PyTorch's DDP wrapper.
  Optimizer state is sharded across GPUs (ZeRO-2 style): each GPU updates only its slice of the parameters,
  then they exchange the results (docstring at `optim.py:184-259`).

## 5. The pipeline (`runs/speedrun.sh`)

| Stage | Script | What happens |
|---|---|---|
| Data | `nanochat/dataset.py` | Download web-text shards (NVIDIA ClimbMix). `-n N` fetches N training shards **plus** one validation shard, always the last one. |
| Tokenizer | `scripts/tok_train.py` | Learn the BPE vocabulary from the data. |
| **Pretraining** | `scripts/base_train.py` | Next-token prediction over billions of tokens of web text. Most of the compute. The result, a "base model", autocompletes text but doesn't follow instructions. |
| Base eval | `scripts/base_eval.py` | CORE score (few-shot tasks from the DCLM paper, `nanochat/core_eval.py`) and bits per byte. |
| **SFT** (supervised finetuning) | `scripts/chat_sft.py` | Keep training on a mixture of conversations: SmolTalk (~460K general chats), MMLU auxiliary-train (multiple choice) and GSM8K (grade-school math) (`chat_sft.py:162`). The loss counts only the assistant's tokens: `render_conversation` (`nanochat/tokenizer.py:140`) builds a mask, and masked targets become −1, which cross-entropy ignores (`chat_sft.py:285-290`). This turns the autocompleter into a chatbot. |
| Chat eval | `scripts/chat_eval.py` | MMLU, ARC, GSM8K, HumanEval (in `tasks/`). |
| RL (optional) | `scripts/chat_rl.py` | On GSM8K: sample several answers per question, reward the correct ones, push probability toward the better-than-average answers. Called "GRPO" but simplified to something close to REINFORCE: no KL penalty, no PPO clipping, advantage = reward − mean reward (`chat_rl.py:1-17`). |
| Serve | `scripts/chat_cli.py`, `nanochat/engine.py` | Inference with the KV cache. |

**Pretraining vs finetuning** use the same loss and the same machinery. What differs is the data (raw web text
vs curated conversations), the loss mask (SFT trains only on assistant tokens), and the scale.

## 6. Scaling, and what "doing AI work" looks like

- **`--depth` is the single dial.** Width, heads, learning rates, batch size and training length are derived
  from it. This works because of **scaling laws**: loss falls predictably, as a power law, with compute,
  parameters and data.
- **Compute-optimal (Chinchilla)**: for a fixed compute budget there is a best split between model size and
  training tokens. `--target-param-data-ratio` sets tokens per parameter: the default in `base_train.py` is 12,
  the speedrun uses 8, and the Chinchilla paper's figure is about 20.
- **`estimate_flops`** (`gpt.py:319`): training costs about `6 × params` FLOPs per token (2 for the forward pass,
  4 for the backward pass), plus a term for attention.
- **Metrics**: validation **bpb** (bits per byte, lower is better) is the loss normalised per byte of text, so it
  is comparable across tokenizers. **CORE** is a suite of downstream tasks; GPT-2's CORE is 0.256525, which is
  the speedrun's finish line (`README.md`).
- **MFU** (model FLOPs utilisation) is the fraction of the GPU's peak speed you actually use. `dev/LOG.md`
  records around 35–46% on H100 runs.
- **The research loop** is what the README leaderboard is: change one thing (architecture, optimizer, data),
  train a small model, check whether the metric improved at equal compute. Most day-to-day work is this, plus
  debugging numerics and making sure the GPU never sits idle.

## Suggested reading order in the repo

1. `nanochat/gpt.py` top to bottom: `Block`, `CausalSelfAttention` and `MLP` first, then `GPT.forward`.
2. `scripts/base_train.py` for the training loop. Walkthrough in [02-base-train-walkthrough.md](02-base-train-walkthrough.md).
3. `nanochat/dataloader.py` and `nanochat/tokenizer.py`.
4. `scripts/chat_sft.py` to see how SFT differs from pretraining (loss masking).
5. `nanochat/engine.py` for generation and the KV cache.
6. `nanochat/optim.py` last; it is the densest.

Outside material: Karpathy's "Let's build GPT: from scratch, in code, spelled out" video covers sections 1–3
hands-on. Then "Attention Is All You Need" (2017) and the Chinchilla paper (arXiv 2203.15556).

## Corrections vs. the chat version

Checked against the code after the session; these differ from what I first said:

- **Sliding window size.** I said "~768 tokens", copied from the comment at `gpt.py:302` ("2048 -> 768"). The
  formula on that line, `-(-2048 // 4 // 128) * 128`, gives **512**. The `--window-pattern` help text in
  `base_train.py:54` ("S=half context") is also wrong. The docstring at `gpt.py:296` ("quarter context") is the
  one that matches the code. Lesson: comments go stale, so check the arithmetic.
- **GQA** is off by default (`base_train.py:138`). I hadn't said so.
- **SFT data** is a SmolTalk + MMLU + GSM8K mixture, not just `tasks/smoltalk.py`.
- **"The MLP holds ~2/3 of the parameters"** is true per block. Small models are dominated by embedding tables.
- **Muon** orthogonalises with Polar Express coefficients, a tuned Newton–Schulz variant, and adds NorMuon and
  cautious weight decay.
