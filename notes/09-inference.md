# 09 — Inference: from logits to text (`nanochat/engine.py`, `scripts/chat_cli.py`)

> Companion experiment: [experiments/sampling.py](experiments/sampling.py)
> (`.venv\Scripts\python.exe -m notes.experiments.sampling`). Code at commit `92d63d4`. Outputs are from the
> 500-step d4 model on 2026-10-23.

Generation is a loop: run the model, get a probability for every possible next token, **choose** one, append
it, repeat. All the interesting decisions are in "choose", and all the speed tricks are in "run the model again".

## 1. What the model actually outputs (experiment section 1)

For the prompt `The capital of France is`, the logits at the last position become probabilities:

```
18.34%  ' a'        1.36%  ' not'       1.23%  ' that'
13.37%  ' the'      1.25%  ' also'      1.11%  ' in'
 3.58%  ' an'       1.24%  ' used'      0.83%  ' found'
 1.93%  ' to'
' Paris' is 1 token(s); its probability: 0.0023%, rank 2505 of 32,768
entropy of this distribution: 8.35 bits (uniform over the vocab would be 15.00)
```

- The model is a **distribution**, not an answer. "Generation" is choosing from it.
- 8.35 bits of entropy means it is about as uncertain as a uniform pick among 2^8.35 ≈ 330 tokens. It has
  learned grammar ("is" → " a" / " the"), not geography. ' Paris' ranks 2,505th.

## 2. Temperature and top-k (`sample_next_token`, `engine.py:140-156`)

```python
if temperature == 0.0:  return argmax(logits)                  # greedy
if top_k:               keep only the k largest logits          # drop the long tail
probs = softmax(logits / temperature); sample from probs
```

Dividing logits by T before the softmax reshapes the distribution (experiment section 2):

```
T=0.5: top-3 ' a' 61.5%, ' the' 32.7%, ' an' 2.3% | entropy 1.42 bits    (sharper: safer, more repetitive)
T=1.0: top-3 ' a' 18.3%, ' the' 13.4%, ' an' 3.6% | entropy 8.35 bits    (the model's own distribution)
T=1.5: top-3 ' a' 2.3%,  ' the' 1.9%,  ' an' 0.8% | entropy 13.23 bits   (flatter: more random)
top-k=50 keeps only the 50 most likely tokens; their share of the probability: 57.3%
```

Top-k matters more than it looks: **43% of the probability sits outside the top 50**, spread over 32,000+
mostly bad tokens. Sampling at T=1 without top-k regularly lands in that tail.

## 3. What the choices do to text (experiment section 3, 24 new tokens, 3 samples each)

```
greedy (temperature 0):            (all 3 samples identical)
    ' a very common part of the most of the most of the most of the most of the most of the most of the'

temperature 0.6, top-k 50 (chat_cli's default):
    " a very little few things that they are the best of the most popular in the following to the whole thing's. The"
    ' the largest in the 2017\nThe 1919 in a 2020, which has been used in the'
    ' a broad method of a broad range of the species. The study of the cell energy from the energy of the 20'

temperature 1.0, no top-k:
    ' rebaning/hiz do not have become a slaughter.\nThe Department of work chainsco as we finally want to target non'
    ' attacked with floss orons such as growth, extends limited center margin. These vegetable doubts can lead to ...'
```

- **Greedy is deterministic and loops.** The most likely continuation of "of the most" is "of the most".
- **T=0.6 + top-k 50** gives varied text that stays locally grammatical. That's why `chat_cli.py` defaults to
  it (`-t 0.6 -k 50`, L18–19).
- **T=1.0 without top-k** samples from the long tail and produces made-up words.
- Evaluations use greedy (`chat_eval.py` default `-t 0.0`), so results are reproducible.

## 4. The KV cache (`KVCache`, `engine.py:82-137`; used at `gpt.py:111-123`)

The naive loop (`GPT.generate`, `gpt.py:527`) reruns the **whole** sequence through the model for every new
token. But the keys and values of past tokens never change, so the Engine stores them per layer and feeds the
model **one new token** per step:

1. **Prefill**: run the prompt once with batch 1 and fill the cache (`engine.py:194-206`).
2. **Replicate** the cache for `num_samples` rows (`KVCache.prefill`, `engine.py:123-137`). The prompt is
   processed once even when sampling many answers.
3. **Decode**: each step forwards only `(B, 1)` new tokens; attention reads the cached K/V
   (`flash_attn_with_kvcache`).

Experiment section 4 checks that both paths produce **identical tokens**, and times them: 9.4× faster for 400
tokens (details at the end of section 6).

Cache size: per token, every layer stores K and V for every KV head: `n_layer × 2 × n_kv_head × head_dim ×
bytes`. The d4 in fp32 is 4 × 2 × 4 × 64 × 4 = **8,192 bytes/token**. The speedrun's d24 in bf16 is 24 × 2 × 12 ×
128 × 2 = 147,456 bytes/token, about **302 MB for one 2,048-token conversation**. That's why GQA, which reduces
`n_kv_head`, and sliding windows matter for serving.

Smear needs a cache too: the model mixes in the *previous token's* embedding, so the cache also remembers it
(`KVCache.prev_embedding`, `gpt.py:481-492`).

## 5. The tool-use state machine (`Engine.generate`, `engine.py:175-275`)

After each sampled token, the Engine checks for special tokens:

- `<|python_start|>` → start collecting tokens as an expression.
- `<|python_end|>` → decode the expression, run `use_calculator` (`engine.py:46-79`). If there's a result,
  **force** `<|output_start|>`, the result's tokens and `<|output_end|>` into the sequence over the next steps.
  The model doesn't sample those: `token_masks` marks them 0 = forced.
- `<|assistant_end|>` or `<|bos|>` → this row is finished.

That's the inference side of what SFT trained ([10](10-sft.md)). The model learned to *write* `2*12` and stop,
and the Engine supplies `24`.

The calculator is deliberately restricted: pure arithmetic without `**`, or `.count()` on string literals,
with a blacklist of dangerous names, `eval` with no builtins, and a 3-second timeout.

### ⚠ The calculator doesn't work on Windows (experiment section 5)

```
use_calculator('2+3*4') raised AttributeError: module 'signal' has no attribute 'alarm'
use_calculator("'strawberry'.count('r')") raised AttributeError: module 'signal' has no attribute 'alarm'
```

The timeout uses `signal.SIGALRM` / `signal.alarm` (`engine.py:30-31, 42`), which exist only on Unix. On
Windows, setting the alarm fails inside the `try`, then the `except` block calls `signal.alarm(0)`, which fails
again and escapes. Any generation where a model calls the calculator would crash with this error on this
machine: an SFT/RL model in `chat_cli`, or GSM8K in `chat_eval`. Base models practically never emit
`<|python_start|>`, so nothing in these notes was affected. A fix would guard the alarm calls with
`hasattr(signal, "SIGALRM")`, or use a thread-based timeout. Not changed, since it's repo code.

Related: `engine.py`'s own self-test (`python -m nanochat.engine`, L302–352) calls `torch.cuda.synchronize()`
and can't run on a CPU-only install. Experiment section 4 does the same naive-vs-Engine check instead.

## 6. How the chat CLI uses all this (`scripts/chat_cli.py`)

- It keeps the whole conversation as one token list:
  `[<|bos|>, <|user_start|>, …, <|user_end|>, <|assistant_start|>, …]`.
- Each turn appends the user message and `<|assistant_start|>`, streams up to 256 tokens from
  `engine.generate`, and makes sure the reply ends with `<|assistant_end|>` (L94–96).
- Every turn re-prefills the entire conversation. The cache isn't kept between turns.
- It loads an `sft` model by default (`-i sft`), but `-i base` works too, because `load_model` accepts
  base/sft/rl. Trying it on the base model shows exactly why SFT exists:

```
> .venv\Scripts\python.exe -m scripts.chat_cli -i base -g d4-500 -p "What is the capital of France?"
Assistant:  is the most common reason for you are using a large bit of a few years, the best way to understand
the right amount of time, and the same way, the most of the time of the way to the way, the next time, ...
(continues until the 256-token limit)
```

  The base model has never seen `<|user_start|>`/`<|assistant_start|>` used as a conversation, so it just
  continues the text. It never produces `<|assistant_end|>`, so the reply only stops at `max_tokens`.

### KV cache timings (experiment section 4, CPU otherwise idle)

```
 64 new tokens: naive GPT.generate 0.34s | Engine with KV cache 0.14s | 2.3x faster | identical tokens: True
400 new tokens: naive GPT.generate 8.58s | Engine with KV cache 0.91s | 9.4x faster | identical tokens: True
```

The naive loop reprocesses the whole sequence at every step, so its total cost grows roughly with the
**square** of the length. With the cache each step handles one token, so total cost grows roughly **linearly**.
6× more tokens cost the naive loop 25× more time and the Engine 6.5× more. For short outputs the gap is small,
because fixed per-step overhead dominates.

## Exercises

1. Change the prompt in section 1 to `"Paris is the capital of"`. Does ' France' rank higher than ' Paris' did?
2. Try T=0.8 with top-k 10 in section 3. Where does text quality break down as you raise T?
3. Why can the Engine run the prompt once and then fork 3 samples, while the naive `GPT.generate` can't?
