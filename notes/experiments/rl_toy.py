"""
The policy-gradient update of scripts/chat_rl.py on a toy problem: a "policy" that answers a 4-way question.
Same recipe: sample 16 answers, reward 1 if correct, advantage = reward - mean reward, loss = -(logp * advantage).
Companion to notes/12-rl.md.

Run from the repo root (no model or data needed):
    .venv\Scripts\python.exe -m notes.experiments.rl_toy
"""
import torch

CORRECT = 2  # answer "C"
NUM_SAMPLES = 16  # chat_rl.py --num-samples

def train(initial_logits, steps=40, lr=0.5, seed=0, label=""):
    torch.manual_seed(seed)
    logits = torch.tensor(initial_logits, requires_grad=True)
    opt = torch.optim.SGD([logits], lr=lr)
    print(f"\n{label}")
    for step in range(steps):
        probs = logits.softmax(-1)
        samples = torch.multinomial(probs.detach(), NUM_SAMPLES, replacement=True)  # rollouts
        rewards = (samples == CORRECT).float()                                       # task.reward(...)
        advantages = rewards - rewards.mean()                                        # chat_rl.py:143-144
        logp = torch.log_softmax(logits, dim=-1)[samples]
        loss = -(logp * advantages).sum() / NUM_SAMPLES                              # chat_rl.py:264-272
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5 == 0 or step == steps - 1:
            print(f"  step {step:2d}: P(correct) {probs[CORRECT].item():.3f} | {int(rewards.sum())}/16 samples correct"
                  f" | advantages {sorted(set(round(a, 3) for a in advantages.tolist()))}")

# 1) the model sometimes gets it right: RL sharpens that
train([0.0, 0.0, 0.0, 0.0], label="1) uniform start, P(correct) = 0.25")

# 2) the model (almost) never gets it right: every group is all-zero reward, so every advantage is 0
train([2.0, 2.0, -4.0, 2.0], label="2) P(correct) = 0.0008 at start: almost never sampled")

# 3) one group by hand: how the advantage spreads over the tokens of each sampled answer
print("\n3) one group of 4 sampled answers (token-level view)")
rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
advantages = rewards - rewards.mean()
lengths = [5, 9, 7, 4]  # tokens in each sampled answer
for r, a, n in zip(rewards.tolist(), advantages.tolist(), lengths):
    print(f"  reward {r:.0f} -> advantage {a:+.2f} applied to each of its {n} tokens")
print(f"  loss = -sum(logp_t * advantage) / {sum(lengths)} valid tokens (token-level normalization, DAPO style)")
print("  if all 4 had the same reward, every advantage would be 0 and this question would teach nothing")
