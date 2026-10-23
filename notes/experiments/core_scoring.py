"""
How CORE scores a multiple-choice question without generating any text: compare the model's loss on each
answer option, pick the lowest. Uses the exact functions of nanochat/core_eval.py.
Companion to notes/11-evaluation.md.

Run from the repo root:
    .venv\Scripts\python.exe -m notes.experiments.core_scoring
"""
import os
import torch

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import render_prompts_mc, batch_sequences_mc, stack_sequences, forward_model

questions = [
    {"query": "The capital of France is", "choices": ["Paris", "London", "banana"], "gold": 0},
    {"query": "Ice is made of frozen", "choices": ["water", "fire", "music"], "gold": 0},
    {"query": "Question: What do bees make?\nAnswer:", "choices": ["honey", "steel", "glass"], "gold": 0},
]

ckpt_root = os.path.join(get_base_dir(), "base_checkpoints")
for tag in ("d4", "d4-500"):
    if not os.path.isdir(os.path.join(ckpt_root, tag)):
        continue
    model, tokenizer, meta = load_model("base", torch.device("cpu"), phase="eval", model_tag=tag)
    print()
    print("=" * 72)
    print(f"model {tag} (step {meta['step']})")
    print("=" * 72)
    correct = 0
    for item in questions:
        prompts = render_prompts_mc(item, " ", [])  # 0-shot: "<query> <choice>"
        tokens, starts, ends = batch_sequences_mc(tokenizer, prompts)  # answer span = after the common prefix
        input_ids = stack_sequences(tokens, tokenizer.get_bos_token_id())
        losses, _ = forward_model(model, input_ids)  # per-token loss of predicting the next token
        # losses[i, t] is the loss for predicting token t+1, so the answer span starts at index start-1
        mean_losses = [losses[i, s - 1:e - 1].mean().item() for i, (s, e) in enumerate(zip(starts, ends))]
        pick = mean_losses.index(min(mean_losses))
        correct += pick == item["gold"]
        scored = "  ".join(f"{c!r} {l:.2f}" for c, l in zip(item["choices"], mean_losses))
        print(f"{item['query']!r:45}  mean loss per choice: {scored}  -> picks {item['choices'][pick]!r}"
              f" {'(correct)' if pick == item['gold'] else '(wrong)'}")
    print(f"{correct}/{len(questions)} correct (random guessing: 1/3 each)")
