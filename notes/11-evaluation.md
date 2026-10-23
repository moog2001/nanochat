# 11 — Evaluation: how nanochat measures a model

> Companion experiment: [experiments/core_scoring.py](experiments/core_scoring.py)
> (`.venv\Scripts\python.exe -m notes.experiments.core_scoring`). Code: `nanochat/loss_eval.py`,
> `nanochat/core_eval.py`, `scripts/base_eval.py`, `scripts/chat_eval.py`, `tasks/` at commit `92d63d4`.

Three kinds of measurement, from cheapest to most meaningful:

| Metric | Measures | Used by | Needs |
|---|---|---|---|
| **bpb** (bits per byte) | how well the model predicts held-out text | `base_train`, `base_eval`, `chat_sft` | just the val shard |
| **CORE** | few-shot ability on 22 tasks (the DCLM paper's CORE suite), by comparing losses | `base_train`, `base_eval` | the eval bundle |
| **ChatCORE** | chat-model accuracy on ARC, MMLU, GSM8K, HumanEval | `chat_sft`, `chat_eval` | task datasets from HuggingFace |

## 1. bpb (`evaluate_bpb`, `loss_eval.py:9-65`)

Sum the loss (in nats) over every target token, divide by `ln 2 × total bytes of those tokens`. Special tokens
count 0 bytes and are left out. The result doesn't depend on the tokenizer, so models with different
vocabularies can be compared. Worked examples are in [03](03-cpu-run-40-steps.md) (step 0: 15 bits/token ÷
4.8 bytes/token ≈ 3.14) and [04](04-cpu-run-500-steps.md).

Limitation: bpb says how *surprised* the model is by text, not whether it can *do* anything. Two models with
equal bpb can differ in usefulness. Hence CORE.

## 2. CORE: scoring without generating (`core_eval.py`)

CORE asks multiple-choice-style questions but **never samples text**. For each answer option it computes the
model's **average loss on that option's tokens** and picks the lowest (`evaluate_example`, L232–237). Three task
formats:

| `task_type` | Prompts differ in… | Scored by |
|---|---|---|
| `multiple_choice` | the continuation (same question, different answers) | lowest mean loss over the answer tokens |
| `schema` | the context (different setups, same ending) | lowest mean loss over the shared ending |
| `language_modeling` | n/a (one prompt) | does greedy prediction reproduce the continuation exactly? |

Few-shot examples are sampled with a seeded RNG (`random.Random(1234 + idx)`, L178), so scores are reproducible.
Long prompts keep the last `max_seq_len` tokens (L196–213).

### The mechanism with our models (experiment)

Three 0-shot questions, scored with `core_eval`'s own functions:

```
model d4 (step 40)
'The capital of France is'              'Paris' 11.61  'London' 11.54  'banana' 14.37  -> picks 'London' (wrong)
'Ice is made of frozen'                 'water'  6.92  'fire'    9.29  'music'   8.95  -> picks 'water' (correct)
'Question: What do bees make?\nAnswer:' 'honey'  9.80  'steel'   9.56  'glass'   8.83  -> picks 'glass' (wrong)

model d4-500 (step 500)
'The capital of France is'              'Paris' 10.70  'London' 10.40  'banana' 12.68  -> picks 'London' (wrong)
'Ice is made of frozen'                 'water'  4.77  'fire'    7.74  'music'   8.31  -> picks 'water' (correct)
'Question: What do bees make?\nAnswer:' 'honey' 12.12  'steel'  11.26  'glass'  10.45  -> picks 'glass' (wrong)
```

- **1/3 correct for both models, which is chance level.** Expected at this scale.
- But the losses show what training *did* learn. "banana" after "The capital of France is" is far less likely
  (12.68) than either city: the model knows a **place name** fits, just not which one. "frozen water" got much
  more likely (6.92 → 4.77), a common phrase it picked up.
- This is why CORE is centered on random chance (next section). A tiny model's raw accuracy is mostly luck.

### Centering and the CORE number (`base_eval.py:57-123`)

For each task: `centered = (accuracy − random_baseline) / (1 − random_baseline)`, with the random baseline for
that task taken from the bundle's `eval_meta_data.csv`. 0 means chance, 1 means perfect, and a negative score
means worse than chance. **CORE = the mean of the centered scores.** GPT-2 scores 0.256525: that's the
speedrun's finish line in the README, which calls it "GPT-2 (1.6B)".

Running it here, like `runs/runcpu.sh` does (downloads the eval bundle from S3 on first use):

```powershell
.venv\Scripts\python.exe -m scripts.base_eval --model-tag=d4-500 --device-batch-size=1 --split-tokens=16384 --max-per-task=16
```

With `--max-per-task=16`, every task score is ±12 percentage points of noise (one standard error). Fine for
checking that the pipeline works, useless for comparing models.

### The real CORE run on the 500-step model

Ran on 2026-10-23 (2.5 minutes; the bundle is a 26 MB download, 162 MB unpacked). Full output:
[logs/base-eval-d4-500.log](logs/base-eval-d4-500.log).

```
train bpb: 1.943995     val bpb: 1.806700        (val matches training's final 1.806699: same 16K tokens)

task (shots, type)                         accuracy  centered
hellaswag_zeroshot (0, multiple_choice)     0.5000    0.3333
jeopardy (10, language_modeling)            0.0000    0.0000
bigbench_qa_wikidata (10, language_modeling) 0.0000   0.0000
arc_easy (10, multiple_choice)              0.2500    0.0000
arc_challenge (10, multiple_choice)         0.3125    0.0833
copa (0, multiple_choice)                   0.6875    0.3750
commonsense_qa (10, multiple_choice)        0.1250   -0.0938
piqa (10, multiple_choice)                  0.6875    0.3750
openbook_qa (0, multiple_choice)            0.2500    0.0000
lambada_openai (0, language_modeling)       0.0000    0.0000
hellaswag (10, multiple_choice)             0.3750    0.1667
winograd (0, schema)                        0.5625    0.1250
winogrande (0, schema)                      0.5000    0.0000
bigbench_dyck_languages (10, language_mod.) 0.0000    0.0000
agi_eval_lsat_ar (3, multiple_choice)       0.0625   -0.1719
bigbench_cs_algorithms (10, language_mod.)  0.0000    0.0000
bigbench_operators (10, language_mod.)      0.0625    0.0625
bigbench_repeat_copy_logic (10, lang. mod.) 0.0000    0.0000
squad (10, language_modeling)               0.0000    0.0000
coqa (0, language_modeling)                 0.0000    0.0000
boolq (10, multiple_choice)                 0.4375   -0.4803
bigbench_language_identification (10, mc)   0.2500    0.1749
CORE metric: 0.0432                        (GPT-2: 0.2565)
```

How to read it:
- **Every `language_modeling` task scores ~0.** Those need the *exact* continuation by greedy decoding (an
  answer span from SQuAD, the last word of a LAMBADA passage). Hopeless at this size.
- **The high-looking ones are 2-way tasks plus luck.** COPA and PIQA have two options, so 50% is chance, and
  11/16 correct is well within noise of 8/16. Centering converts that to 0.375, which is why 16 examples per
  task can't rank models.
- **Centering uses each task's own baseline.** BoolQ's 43.8% centers to −0.48 because its baseline is **62%**,
  the share of "yes" answers (solving `(0.4375 − b)/(1 − b) = −0.4803` gives b ≈ 0.62), not 50%. Guessing the
  majority answer beats this model.
- **train bpb (1.94) vs val bpb (1.81)**: different 16K-token samples of different shards. Each sample is
  small, so this is just which text landed in each, not a sign of a problem.

Unconditioned samples (the model writing from just `<|bos|>`, temperature 1.0) show where a 500-step d4 is:
"If you want to use this condition inovered not have no proteins, during the dynamic science, better practical
protection for particular matrix." English-shaped, meaningless.

## 3. ChatCORE (`scripts/chat_eval.py`)

For chat (SFT/RL) models. Five tasks, two styles:

- **Categorical** (ARC-Easy, ARC-Challenge, MMLU; `run_categorical_eval`, L87–152). Render the question with
  `render_for_completion` ([05](05-tokenizer.md)), run one forward pass, and at the answer position compare the
  logits of **only the allowed letters** (A/B/C/D). No sampling, and batched across questions. The comment at
  L116–119 is honest that restricting to the letters makes the task easier than letting the model answer freely.
- **Generative** (GSM8K, HumanEval; `run_generative_eval`, L28–80). Actually generate an answer (greedy by
  default, up to 512 tokens), then grade it. GSM8K compares the number after `####` (`tasks/gsm8k.py:86-107`);
  HumanEval runs the generated code against unit tests in a sandbox (`nanochat/execution.py`). With
  `--num-samples > 1` a problem counts as passed if **any** sample passes (pass@k).
- Centering uses baseline 0.25 for 4-way multiple choice and 0 for generative tasks. ChatCORE is the mean.

On this machine GSM8K generation would hit the Windows calculator bug from [09](09-inference.md) as soon as a
model calls the tool.

## 4. Which number to trust when

- **During pretraining**: val bpb every N steps. It's cheap, smooth, and scores the same tokens every time
  ([06](06-data-pipeline.md), section 5). Use CORE as an occasional reality check.
- **Comparing architectures or tricks**: bpb at **equal compute** (the leaderboard's discipline), then CORE.
- **Chat models**: ChatCORE plus reading actual conversations. A model can score well and still be unpleasant
  to talk to.
- **Always**: look at samples. The fixed-prompt samples in `base_train` exist because numbers can look fine
  while the text is broken.

## Exercises

1. Add a question to `questions` in the experiment where *frequency* would mislead, e.g. "The largest planet is"
   with "Jupiter" vs "Earth". Which does d4-500 pick, and why?
2. Why does CORE use the *mean* loss over the answer tokens instead of the *sum*? (Think about answers with
   different token counts.)
3. MMLU in ChatCORE only compares 4 letter logits. What would change if it instead generated freely and parsed
   the answer?
