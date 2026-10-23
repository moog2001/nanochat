"""
Inference by hand: next-token probabilities, temperature, top-k, greedy vs sampled text, and the KV cache.
Companion to notes/09-inference.md.

Run from the repo root (loads base checkpoint d4-500 if present, else d4):
    .venv\Scripts\python.exe -m notes.experiments.sampling
"""
import os
import time
import torch
import torch.nn.functional as F

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine, use_calculator

def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

ckpt_root = os.path.join(get_base_dir(), "base_checkpoints")
tag = "d4-500" if os.path.isdir(os.path.join(ckpt_root, "d4-500")) else "d4"
model, tokenizer, _ = load_model("base", torch.device("cpu"), phase="eval", model_tag=tag)
bos = tokenizer.get_bos_token_id()
prompt = "The capital of France is"
prompt_ids = tokenizer.encode(prompt, prepend=bos)

# -----------------------------------------------------------------------------
header(f"1) What the model ({tag}) thinks comes after {prompt!r}")
with torch.no_grad():
    logits = model(torch.tensor([prompt_ids]))[0, -1]  # logits for the NEXT token, shape (vocab,)
probs = F.softmax(logits, dim=-1)
top = torch.topk(probs, 10)
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"   {p:6.2%}  {tokenizer.decode([i])!r}")
paris = tokenizer.encode(" Paris")
print(f"' Paris' is {len(paris)} token(s); its probability: {probs[paris[0]].item():.4%}, "
      f"rank {(probs > probs[paris[0]]).sum().item() + 1} of {probs.numel():,}")
entropy_bits = -(probs * probs.clamp_min(1e-12).log2()).sum().item()
print(f"entropy of this distribution: {entropy_bits:.2f} bits (uniform over the vocab would be 15.00)")

# -----------------------------------------------------------------------------
header("2) Temperature reshapes the distribution: softmax(logits / T)")
for T in (0.5, 1.0, 1.5):
    p_T = F.softmax(logits / T, dim=-1)
    top3 = torch.topk(p_T, 3)
    shown = ", ".join(f"{tokenizer.decode([i])!r} {p:.1%}" for p, i in zip(top3.values.tolist(), top3.indices.tolist()))
    ent = -(p_T * p_T.clamp_min(1e-12).log2()).sum().item()
    print(f"T={T}: top-3 {shown} | entropy {ent:.2f} bits")
print("top-k=50 keeps only the 50 most likely tokens; their share of the probability:",
      f"{torch.topk(probs, 50).values.sum().item():.1%}")

# -----------------------------------------------------------------------------
header("3) Greedy vs sampled continuations (Engine, 24 new tokens)")
engine = Engine(model, tokenizer)
settings = [("greedy (temperature 0)", dict(temperature=0.0)),
            ("temperature 0.6, top-k 50 (chat_cli default)", dict(temperature=0.6, top_k=50)),
            ("temperature 1.0, no top-k", dict(temperature=1.0))]
for name, kw in settings:
    print(f"\n{name}:")
    results, _ = engine.generate_batch(prompt_ids, num_samples=3, max_tokens=24, seed=42, **kw)
    for r in results:
        print("   ", repr(tokenizer.decode(r[len(prompt_ids):])))

# -----------------------------------------------------------------------------
header("4) The KV cache: same tokens, less work")
for n_new in (64, 400):
    t0 = time.time()
    naive = list(model.generate(prompt_ids, max_tokens=n_new, temperature=0))  # recomputes the whole sequence each step
    t_naive = time.time() - t0
    t0 = time.time()
    cached, _ = engine.generate_batch(prompt_ids, num_samples=1, max_tokens=n_new, temperature=0.0)
    t_cached = time.time() - t0
    cached_new = cached[0][len(prompt_ids):]
    print(f"{n_new} new tokens: naive GPT.generate {t_naive:.2f}s | Engine with KV cache {t_cached:.2f}s "
          f"| {t_naive / t_cached:.1f}x faster | identical tokens: {naive[:len(cached_new)] == cached_new}")
print(f"KV cache memory for this model: {model.kv_bytes_per_token():,} bytes per token "
      f"(n_layer x 2 [K and V] x n_kv_head x head_dim x 4 bytes fp32)")

# -----------------------------------------------------------------------------
header("5) The calculator tool on this machine")
for expr in ["2+3*4", "'strawberry'.count('r')"]:
    try:
        print(f"use_calculator({expr!r}) -> {use_calculator(expr)!r}")
    except Exception as e:
        print(f"use_calculator({expr!r}) raised {type(e).__name__}: {e}")
