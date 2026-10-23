"""
The two optimizers by hand: an AdamW step, Muon's orthogonalization, cautious weight decay, and the real
parameter groups of the d4 model.
Companion to notes/08-optimizers.md.

Run from the repo root:
    $env:TORCHDYNAMO_DISABLE = "1"   (the optimizer kernels are decorated with torch.compile)
    .venv\Scripts\python.exe -m notes.experiments.optimizer_by_hand
"""
import os
import torch

from nanochat.optim import adamw_step_fused, polar_express_coeffs

torch.set_printoptions(precision=4, sci_mode=False)
torch.manual_seed(0)

def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

# -----------------------------------------------------------------------------
header("1) One AdamW step, by hand and with nanochat's fused kernel")
p = torch.tensor([1.0, -2.0, 0.5, 3.0])
g = torch.tensor([0.3, -0.001, 5.0, -0.02])  # gradients of very different sizes
lr, beta1, beta2, eps, wd = 0.01, 0.8, 0.95, 1e-10, 0.0

# by hand, following optim.py:44-59
m = (1 - beta1) * g                        # first moment, starts from 0
v = (1 - beta2) * g * g                    # second moment, starts from 0
m_hat, v_hat = m / (1 - beta1), v / (1 - beta2)  # bias correction for step 1
by_hand = p - lr * m_hat / (v_hat.sqrt() + eps)

p_fused = p.clone()
exp_avg, exp_avg_sq = torch.zeros(4), torch.zeros(4)
t = lambda x: torch.tensor(float(x))
adamw_step_fused(p_fused, g, exp_avg, exp_avg_sq, t(1), t(lr), t(beta1), t(beta2), t(eps), t(wd))
print("gradient        ", g)
print("update by hand  ", by_hand - p)
print("update (nanochat)", p_fused - p)
print("-lr * sign(grad)", -lr * g.sign())
print("=> Adam's first step moves every weight by ~lr, whatever the gradient's size (it is sign descent).")

# -----------------------------------------------------------------------------
header("2) Muon's orthogonalization: equalize the singular values of the update")
def polar_express(G, steps=5):
    """The orthogonalization loop of muon_step_fused (optim.py:143-154), for one wide matrix."""
    X = G / (G.norm() * 1.01 + 1e-6)
    for a, b, c in polar_express_coeffs[:steps]:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X

# a gradient-like matrix: a few strong directions plus weak ones (rank-4 signal + noise)
U, _ = torch.linalg.qr(torch.randn(64, 64))
V, _ = torch.linalg.qr(torch.randn(128, 128))
s = torch.tensor([50.0, 10.0, 3.0, 1.0] + [0.05] * 60)
G = U @ torch.diag(s) @ V[:64]  # (64, 128)
for steps in (0, 1, 2, 5):
    X = G / G.norm() if steps == 0 else polar_express(G, steps)
    sv = torch.linalg.svdvals(X)
    print(f"after {steps} iterations: largest singular values {sv[:4].tolist()}, smallest {sv[-1].item():.3f}")
Uf, Sf, Vf = torch.linalg.svd(G, full_matrices=False)
exact = Uf @ Vf  # the exact orthogonal factor U V^T
X5 = polar_express(G, 5)
print(f"\ncosine similarity of the 5-step result with the exact U V^T: "
      f"{(X5 * exact).sum().item() / (X5.norm() * exact.norm()).item():.4f}")
print("share of the raw update along the strongest direction:  "
      f"{(s[0] ** 2 / (s ** 2).sum()).item():.1%}")
print("share after orthogonalization (every direction equal): "
      f"{(torch.linalg.svdvals(X5)[0] ** 2 / torch.linalg.svdvals(X5).pow(2).sum()).item():.1%}")

# -----------------------------------------------------------------------------
header("3) Cautious weight decay (optim.py:179-180)")
params = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
update = torch.tensor([[0.1, 0.1], [-0.1, -0.1]])  # the step is params -= lr * update
mask = (update * params) >= 0
print("params\n", params)
print("update (params -= lr * update)\n", update)
print("decay applied where the update already shrinks the weight (same sign)\n", mask)

# -----------------------------------------------------------------------------
header("4) The real parameter groups of the d4 model, with the CPU run's learning rates")
from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import load_model
ckpt_root = os.path.join(get_base_dir(), "base_checkpoints")
tag = "d4-500" if os.path.isdir(os.path.join(ckpt_root, "d4-500")) else "d4"
model, _, _ = load_model("base", torch.device("cpu"), phase="train", model_tag=tag)
batch_lr_scale = (4096 / 2**19) ** 0.5  # what base_train.py computed for --total-batch-size=4096
opt = model.setup_optimizer(
    unembedding_lr=0.008 * batch_lr_scale, embedding_lr=0.3 * batch_lr_scale,
    scalar_lr=0.5 * batch_lr_scale, matrix_lr=0.02 * batch_lr_scale, weight_decay=0.236238,
)
names = {p: n for n, p in model.named_parameters()}
for group in opt.param_groups:
    n_params = sum(p.numel() for p in group["params"])
    first = names[group["params"][0]]
    extra = f"betas {group['betas']}" if group["kind"] == "adamw" else f"momentum {group['momentum']}, beta2 {group['beta2']}"
    print(f"{group['kind']:5s} lr {group['lr']:.5f}  {len(group['params']):2d} tensor(s) {n_params:>10,d} params  "
          f"shape {tuple(group['params'][0].shape)}  e.g. {first} | {extra}")
