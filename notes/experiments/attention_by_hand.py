"""
Attention, RMSNorm, RoPE and causality: computed by hand, checked against PyTorch / nanochat.
Companion to notes/07-gpt-forward-pass.md.

Run from the repo root:
    .venv\Scripts\python.exe -m notes.experiments.attention_by_hand
Sections 4-5 load a base checkpoint (d4-500 if present, else d4) from ~/.cache/nanochat.
"""
import os
import math
import torch
import torch.nn.functional as F

from nanochat.gpt import apply_rotary_emb, norm

torch.set_printoptions(precision=3, sci_mode=False)
torch.manual_seed(0)

def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

# -----------------------------------------------------------------------------
header("1) Causal self-attention for ONE head: 4 tokens, head_dim 2")
# Each row is one token (positions 0..3). Hand-picked so the numbers are readable.
q = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., 0.]])
k = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 2.]])
v = torch.tensor([[10., 0.], [0., 10.], [5., 5.], [1., 1.]])
T, d = q.shape

scores = q @ k.T / math.sqrt(d)  # (T, T): row i = how much token i "likes" each token j
print("scores = Q @ K^T / sqrt(d)\n", scores)
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))  # True where j <= i
masked = scores.masked_fill(~causal, float("-inf"))
print("after the causal mask (future positions -> -inf)\n", masked)
weights = masked.softmax(dim=-1)  # exp(-inf) = 0, and each row sums to 1
print("attention weights = row-wise softmax\n", weights)
out = weights @ v  # each output row is a weighted average of the value rows
print("output = weights @ V\n", out)
ref = F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None], is_causal=True)[0, 0]
print("max |by hand - PyTorch SDPA| =", (out - ref).abs().max().item())

# -----------------------------------------------------------------------------
header("2) RMSNorm: divide each vector by its root-mean-square")
x = torch.tensor([[3., 4., 0., 0.], [0.1, 0.2, 0.3, 0.4]])  # same "shape", very different scales
rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
by_hand = x / rms
print("x\n", x)
print("rms per row\n", rms.squeeze(-1))
print("x / rms (by hand)\n", by_hand)
print("nanochat norm(x)\n", norm(x))
print("max |by hand - norm(x)| =", (by_hand - norm(x)).abs().max().item(), "(F.rms_norm adds a tiny eps)")
print("RMS of each output row:", norm(x).pow(2).mean(dim=-1).sqrt())

# -----------------------------------------------------------------------------
header("3) RoPE: after rotation, the q.k score depends only on the DISTANCE i - j")
head_dim = 8
# Same recipe as GPT._precompute_rotary_embeddings (nanochat/gpt.py:270), base 100000
inv_freq = 1.0 / (100000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
freqs = torch.outer(torch.arange(64, dtype=torch.float32), inv_freq)  # (position, pair)
cos, sin = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]

def rotate(vec, pos):
    # vec is (B=1, T=1, H=1, head_dim); pick the cos/sin row for this position
    return apply_rotary_emb(vec, cos[:, pos:pos + 1], sin[:, pos:pos + 1])

qv = torch.randn(1, 1, 1, head_dim)
kv = torch.randn(1, 1, 1, head_dim)
print(f"plain q.k (no rotation): {(qv * kv).sum().item():+.4f}")
for i, j in [(3, 1), (10, 8), (40, 38), (10, 1), (50, 41)]:
    s = (rotate(qv, i) * rotate(kv, j)).sum().item()
    print(f"q at position {i:2d}, k at position {j:2d} (distance {i - j}): score {s:+.4f}")
print("rotation keeps lengths: |q| =", f"{qv.norm().item():.4f}", " |rotated q| =", f"{rotate(qv, 37).norm().item():.4f}")

# -----------------------------------------------------------------------------
from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import load_model

ckpt_root = os.path.join(get_base_dir(), "base_checkpoints")
tag = "d4-500" if os.path.isdir(os.path.join(ckpt_root, "d4-500")) else "d4"
model, tokenizer, _ = load_model("base", torch.device("cpu"), phase="eval", model_tag=tag)

header(f"4) Causality in the real model ({tag}): edit one token, see which positions change")
bos = tokenizer.get_bos_token_id()
ids_a = tokenizer.encode("The cat sat on the mat and looked at the door", prepend=bos)
edit_pos = 5
ids_b = list(ids_a)
ids_b[edit_pos] = tokenizer.encode(" dog")[0]
print("tokens A:", [tokenizer.decode([t]) for t in ids_a])
print("tokens B:", [tokenizer.decode([t]) for t in ids_b], f"(position {edit_pos} edited)")
with torch.no_grad():
    logits_a = model(torch.tensor([ids_a]))[0]  # (T, vocab)
    logits_b = model(torch.tensor([ids_b]))[0]
for pos in range(len(ids_a)):
    diff = (logits_a[pos] - logits_b[pos]).abs().max().item()
    note = "  <- edited here" if pos == edit_pos else ""
    print(f"position {pos:2d}: max |logits A - logits B| = {diff:.6f}{note}")

# -----------------------------------------------------------------------------
header(f"5) Tensor shapes through one forward pass ({tag}, batch 1)")
records = []
def shape_hook(name):
    def fn(module, inputs, output):
        records.append((name, tuple(inputs[0].shape), tuple(output.shape)))
    return fn
block0 = model.transformer.h[0]
watch = {
    "transformer.wte (embedding)": model.transformer.wte,
    "h[0].attn.c_q": block0.attn.c_q,
    "h[0].attn.c_k": block0.attn.c_k,
    "h[0].attn.c_v": block0.attn.c_v,
    "h[0].attn.c_proj": block0.attn.c_proj,
    "h[0].mlp.c_fc": block0.mlp.c_fc,
    "h[0].mlp.c_proj": block0.mlp.c_proj,
    "lm_head": model.lm_head,
}
handles = [m.register_forward_hook(shape_hook(name)) for name, m in watch.items()]
with torch.no_grad():
    model(torch.tensor([ids_a]))
for h in handles:
    h.remove()
cfg = model.config
print(f"config: n_layer={cfg.n_layer} n_head={cfg.n_head} n_embd={cfg.n_embd} "
      f"head_dim={cfg.n_embd // cfg.n_head} vocab={cfg.vocab_size} | T={len(ids_a)} tokens")
for name, shape_in, shape_out in records:
    print(f"{name:30s} in {str(shape_in):18s} -> out {shape_out}")
