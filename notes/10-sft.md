# 10 — SFT: turning the autocompleter into a chat assistant (`scripts/chat_sft.py`)

> Companion experiment: [experiments/sft_data.py](experiments/sft_data.py)
> (`.venv\Scripts\python.exe -m notes.experiments.sft_data`; downloads the small GSM8K-train and MMLU-dev splits).
> Code: `scripts/chat_sft.py`, `tasks/*.py`, `nanochat/tokenizer.py` at commit `92d63d4`. Outputs are from 2026-10-23.
> SFT itself was **not** run on this machine. This guide is from reading the code plus rendering real examples.

A base model continues text. Ask it a question and it may continue with more questions. **Supervised
finetuning (SFT)** keeps training the same model with the same next-token loss, but on *conversations*, and
**only on the assistant's tokens**. Afterwards, `<|user_start|>…<|user_end|><|assistant_start|>` reliably leads
to an assistant-style answer ending in `<|assistant_end|>`.

## 1. What changes compared to pretraining

| | Pretraining (`base_train.py`) | SFT (`chat_sft.py`) |
|---|---|---|
| Starts from | random init | the base checkpoint, **including its optimizer momentum** (L139–149) |
| Data | ClimbMix web text | a mixture of conversations (section 2) |
| Loss on | every token | assistant tokens only; everything else has target −1 (L285–296) |
| Packing | best-fit, **crop** to fill rows | best-fit, **pad** with `<|bos|>` (targets −1), so nothing is ever discarded (L180–298) |
| Length | set by scaling laws | one pass over the mixture (`consumed >= dataset_size`, L276) |
| LR schedule | warmup / constant / warmdown by step | by *progress* 0→1: 80% of the base LR (`--init-lr-frac`), constant for the first half, linear to 0 (L304–314) |
| Weight decay | cosine to 0 | 0 (pretraining already ended at 0, L131–132) |
| Muon momentum | 0.85 → 0.97 → 0.90 | 0.85 → 0.95 over 300 steps (L317–320) |
| Extra eval | CORE | ChatCORE (section 5) |
| Output | `base_checkpoints/` | `chatsft_checkpoints/` |

Batch size, sequence length and learning rates are **inherited from the base checkpoint's metadata** unless
overridden (L96–115). Reloading the optimizer state needs one fix: pretraining's warmdown left every LR near
zero, so the fresh SFT learning rates are restored after loading (L134–147).

## 2. The data mixture (`chat_sft.py:161-173`, `tasks/`)

```python
train_tasks = [
    SmolTalk(split="train"),                                   # ~460K general conversations
    *[MMLU("all", "auxiliary_train") for _ in range(3)],       # ~100K multiple-choice rows, 3 epochs
    *[GSM8K("main", "train") for _ in range(4)],               # 7,473 math problems, 4 epochs
]
```

- **`TaskMixture`** (`tasks/common.py:129-161`) concatenates the tasks and shuffles all (task, index) pairs with
  a fixed seed, so every batch mixes all three. Passing a task several times **oversamples** it (the comment
  calls this a "fun trick"). Here it's 3× MMLU and 4× GSM8K.
- **SmolTalk** teaches general conversation. **MMLU** teaches the multiple-choice format. **GSM8K** teaches
  step-by-step math *and tool use* (`--gsm8k-epochs` help text).
- Datasets come from the HuggingFace hub through `load_hub_dataset` (`tasks/common.py:45-82`), cached under
  `~/.cache/nanochat/task_data/`.

## 3. A math problem with tool calls (experiment section 1)

GSM8K's reference answers contain calculator annotations like `<<2*12=24>>`. `tasks/gsm8k.py:51-84` turns each
one into a **python** part (the expression) and a **python_output** part (the result):

```
text          'Mimi has 2 x 12 = '
python        '2*12'
python_output '24'
text          '24 sea shells.\nKyle has 24 x 2 = '
python        '24*2'
python_output '48'
...
text          '16 sea shells.\n#### 16'
```

Rendered for training (`[[ ]]` = in the loss):

```
<|bos|><|user_start|>Mimi picked up 2 dozen seashells ... How many seashells did Leigh have?<|user_end|>
<|assistant_start|>[[Mimi has 2 x 12 = <|python_start|>2*12<|python_end|>]]<|output_start|>24<|output_end|>
[[24 sea shells.\nKyle has 24 x 2 = <|python_start|>24*2<|python_end|>]]<|output_start|>48<|output_end|>
[[48 sea shells.\nLeigh has 48 / 3 = <|python_start|>48/3<|python_end|>]]<|output_start|>16<|output_end|>
[[16 sea shells.\n#### 16<|assistant_end|>]]
-> 129 tokens, 61 trained on (47%)
```

The model learns to **write the expression** and **stop** with `<|python_end|>`. It never learns to produce
the `24` itself: at inference time the Engine runs the expression and *forces* the result tokens into the
sequence ([09](09-inference.md)). A small model can't multiply reliably, but it can learn when to ask a
calculator. The final answer after `####` is what gets graded.

## 4. A multiple-choice question (experiment section 2)

```
Multiple Choice question: Psychological egoism is:
- an ethical theory about how we ought to behave.=A
- a generalization concerning the way people tend to behave.=B
- a claim about human nature and the ways people are capable of behaving.=C
- none of the above.=D

Respond only with the letter of the correct answer.
```

The assistant's whole reply is `C`. Only **2 of 84 tokens** are trained on: `C` and `<|assistant_end|>`.

Two deliberate formatting choices (`render_mc`, `tasks/common.py:187-206`):
- **The letter comes after the choice** ("…behaving.=C"). The comment says small models bind the letter to the
  choice better that way.
- **No space before the letter.** `'A'` is token 65, `' A'` is token 328, and `'=A'` tokenizes as `[61, 65]`.
  The answer `A` is therefore the *same token* as the label in the prompt, which makes the "copy the right
  letter" job easier for a small model.

## 5. ChatCORE: the SFT scoreboard (`chat_sft.py:356-389`)

Every `--chatcore-every` steps it runs five tasks: ARC-Easy, ARC-Challenge, MMLU (multiple choice) and GSM8K,
HumanEval (generative). Each score is **centered**: `(accuracy − baseline) / (1 − baseline)`, with baseline
0.25 for 4-way multiple choice (random guessing) and 0 for generative tasks. ChatCORE is the mean, so 0 means
"no better than chance" and 1 means perfect. More on evaluation in [11](11-evaluation.md).

## 6. Running it here

```powershell
.venv\Scripts\python.exe -m scripts.chat_sft --model-tag=d4-500 --chatcore-every=-1 --eval-every=-1 --num-iterations=20
```

Untested sketch, for whenever you want to try it. Caveats:
- It downloads SmolTalk and MMLU's auxiliary train split (a few hundred MB, my guess).
- It inherits the d4-500 run's `max_seq_len=512` and batch settings.
- ChatCORE's GSM8K part would hit the Windows calculator bug from [09](09-inference.md), hence
  `--chatcore-every=-1`.
- A 500-step d4 model is far too weak to become a useful chatbot. The point would be watching the mechanics.

## Exercises

1. In the GSM8K rendering, why are `<|python_start|>` and `<|python_end|>` inside `[[ ]]` while
   `<|output_start|>…<|output_end|>` is outside?
2. The MMLU example trains on 2% of its tokens. What does that mean for how many examples SFT needs, compared
   with pretraining?
3. `TaskMixture` oversamples GSM8K 4×. What could go wrong if you oversampled it 40×?
