"""
What SFT actually trains on: real GSM8K and MMLU examples rendered into tokens with their loss masks.
Companion to notes/10-sft.md.

Run from the repo root (downloads the small GSM8K-train and MMLU-dev splits into ~/.cache/nanochat/task_data):
    .venv\Scripts\python.exe -m notes.experiments.sft_data
"""
from nanochat.tokenizer import get_tokenizer
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU

tok = get_tokenizer()

def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

def show_masked(ids, mask):
    """Decode the conversation, wrapping the trained-on (mask=1) spans in [[ ]]."""
    groups, current, current_mask = [], [], mask[0]
    for token_id, m in zip(ids, mask):
        if m != current_mask:
            groups.append((current_mask, current))
            current, current_mask = [], m
        current.append(token_id)
    groups.append((current_mask, current))
    print("".join(f"[[{tok.decode(g)}]]" if m else tok.decode(g) for m, g in groups))
    print(f"-> {len(ids)} tokens, {sum(mask)} trained on ({sum(mask) / len(ids):.0%})")

# -----------------------------------------------------------------------------
header("1) A GSM8K training example: calculator annotations become tool calls")
gsm = GSM8K(subset="main", split="train")
print(f"GSM8K train: {len(gsm):,} problems")
conv = gsm[0]
print("\nuser:", repr(conv["messages"][0]["content"]))
print("\nassistant content parts:")
for part in conv["messages"][1]["content"]:
    print(f"   {part['type']:13s} {part['text']!r}")
print("\nrendered for SFT ([[ ]] = trained on):")
ids, mask = tok.render_conversation(conv)
show_masked(ids, mask)
print("\nthe graded answer is what follows '####':", repr(conv["messages"][1]["content"][-1]["text"][-12:]))

# -----------------------------------------------------------------------------
header("2) An MMLU example: multiple choice, answered with a single letter")
mmlu = MMLU(subset="all", split="dev")
conv = mmlu[0]
print(f"MMLU dev: {len(mmlu)} questions; this one is from {conv['subject']!r}")
print("\nuser message (tasks/common.py render_mc):")
print(conv["messages"][0]["content"])
print("\nassistant message:", repr(conv["messages"][1]["content"]))
ids, mask = tok.render_conversation(conv)
print("\nrendered for SFT:")
show_masked(ids, mask)
print("\nwhy no space before the letter in '=A': the answer token must be the same id as in the prompt")
for s in ["A", " A", "=A"]:
    print(f"   {s!r:6} -> {tok.encode(s)}")
