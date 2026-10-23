# 05 — The tokenizer: text ↔ token ids

> Companion experiment: [experiments/tokenizer_tour.py](experiments/tokenizer_tour.py)
> (`.venv\Scripts\python.exe -m notes.experiments.tokenizer_tour`). Code: `nanochat/tokenizer.py`,
> `scripts/tok_train.py`, `scripts/tok_eval.py` at commit `92d63d4`. All outputs are from runs on 2026-10-23.
> Caveat: the tokenizer used here was trained on 200M characters instead of the default 2B (see
> [03-cpu-run-40-steps.md](03-cpu-run-40-steps.md)), so exact numbers may differ slightly from a full run.

The model never sees text, only integers. The tokenizer turns text into a list of ids from a fixed vocabulary
(32,768 here) and back. It is trained **before** the model and never changes afterwards.

## 1. Why tokens at all?

- **Characters or bytes** would make sequences very long. Attention cost grows with length squared, and the
  model would spend its capacity learning to spell.
- **Whole words** would need a gigantic vocabulary and still fail on new words.
- **Subword tokens** (BPE) are the compromise: frequent words become one token, rare words are built from
  pieces, and every possible string can still be encoded, because the 256 raw bytes are in the vocabulary.

## 2. How BPE (byte-pair encoding) training works

1. Start with a vocabulary of the 256 single bytes. Text is just bytes (UTF-8).
2. **Pre-split** the text with a regex (`SPLIT_PATTERN`, `tokenizer.py:26`) into chunks like words, numbers,
   punctuation and whitespace, so merges never cross those boundaries.
3. Count every adjacent pair of tokens in the training text. **Merge the most frequent pair** into a new token.
4. Repeat until the vocabulary is full.

The arithmetic for nanochat: 32,768 total = 256 bytes + **32,503 merges** (the training log printed
`32503/32503 merges`) + **9 special tokens**.

`tok_train.py` streams training-split documents, crops each to 10,000 characters (`--doc-cap`), stops at
`--max-chars`, and trains with **rustbpe** (a Rust BPE trainer). For actually *using* the tokenizer it builds a
**tiktoken** encoding (`tokenizer.py:49-61`), OpenAI's fast BPE library. Finally it saves `token_bytes.pt`, the
byte length of every token id, with 0 for special tokens. That file is what the bpb metric uses
(see [02](02-base-train-walkthrough.md), Phase 6).

### What the merges learned (experiment section 3)

```
first merges (ids 256-275): ' t' ' a' 'in' 'he' 're' 'on' ' the' 'er' ' s' 'at' ' c' ' o' 'es' 'en' ' w' 'nd' 'ing' 'it' 'is' ' p'
last merges (ids 32749-32758): ' adenosine' ' Utilize' ' inductive' ' tilted' ' Fungi' ' Barton' 'Audio' 'Western' 'Appe' ' Sewing'
longest tokens: '-----...' (32 bytes) '.....' '_____' ... ' telecommunications' ' interdisciplinary' ' neurotransmitters'
```

The first merges are the most common English pairs. ' the' is merge #7. The last merges are rare whole words.
The average vocabulary entry is 6.57 bytes.

## 3. What tokenized text looks like (experiment sections 1–2)

```
'Hello world! The cat sat on the mat.'      10 tokens, 3.60 bytes/token
  'Hello' ' world' '!' ' The' ' cat' ' sat' ' on' ' the' ' mat' '.'

'tokenization tokenizer untokenizable'      9 tokens
  'to' 'ken' 'ization' ' token' 'izer' ' unt' 'o' 'ken' 'izable'

'In 2026 the price was 1234567 won.'        14 tokens
  'In' ' ' '20' '26' ' the' ' price' ' was' ' ' '12' '34' '56' '7' ' won' '.'

'def add(a, b):\n    return a + b'          11 tokens
  'def' ' add' '(a' ',' ' b' '):\n' '   ' ' return' ' a' ' +' ' b'
```

Things to notice:

- **Spaces belong to the next word**: ' world', ' cat'. And `'the'`, `' the'`, `'The'`, `' The'`, `' THE'` are
  five *different* ids (1234, 262, 483, 370, 6669). The model has to learn they are related.
- **Numbers are cut into 1–2 digit chunks** (`\p{N}{1,2}` in the regex). GPT-4 uses up to 3; the comment at
  `tokenizer.py:23-25` says 2 was the sweet spot for a 32K vocabulary. This is one reason arithmetic is hard for
  LLMs: "1234567" arrives as 12 | 34 | 56 | 7.
- **Code indentation**: four spaces became `'   '` + `' return'`. The last space glues onto the word.

### Korean, and why the training data matters

```
'안녕하세요, 반갑습니다.'    29 tokens, 33 bytes -> 1.14 bytes/token
  b'\xec' b'\x95' b'\x88' b'\xeb' b'\x85' b'\x95' b'\xed\x95' b'\x98' ...
```

Each Hangul syllable is 3 bytes in UTF-8, and almost none of them got merges: the model would read Korean
nearly **byte by byte**. ClimbMix is mostly English, so BPE spent its merges on English. `tok_eval` confirms
it across a whole Korean passage (next section): GPT-4's tokenizer needs about half as many tokens.

## 4. How good is the compression? (`scripts/tok_eval.py`)

Bytes per token, higher is better:

| Text | GPT-2 (50,257 vocab) | GPT-4 (100,277) | Ours (32,768) |
|---|---|---|---|
| news | 4.50 | 4.70 | 4.50 |
| korean | 1.20 | 2.45 | 1.27 |
| code | 2.19 | 4.07 | 3.13 |
| math | 1.96 | 2.20 | 2.01 |
| science | 4.28 | 4.47 | 4.54 |
| climbmix-train | 4.67 | 4.82 | 4.74 |
| climbmix-val | 4.63 | 4.79 | **4.69** |

- With a vocabulary 35% smaller than GPT-2's, ours ties or beats it everywhere, because it was trained on the
  same kind of text it will see.
- GPT-4's 3× larger vocabulary wins on code and Korean.
- **4.69 bytes/token on validation data** matches what the training run implied at step 0 (about 4.8; see
  [03](03-cpu-run-40-steps.md)). That's two independent measurements agreeing.

Why compression matters: at a fixed context of 2048 tokens, better compression means the model sees more text.
Per training token, it also learns more.

## 5. Special tokens (experiment section 4)

```
32759 <|bos|>            32763 <|assistant_end|>     32767 <|output_end|>
32760 <|user_start|>     32764 <|python_start|>
32761 <|user_end|>       32765 <|python_end|>
32762 <|assistant_start|> 32766 <|output_start|>
```

- **`<|bos|>`** (beginning of sequence) starts every document in pretraining. In the GPT-2/GPT-4 tokenizers the
  same role is called `<|endoftext|>` (`tokenizer.py:74-78`).
- The other 8 only appear in chat finetuning. They give conversations a structure the model can learn: who is
  speaking, and where a Python tool call and its output begin and end.
- **Typing a special token's text doesn't produce it**: `encode("<|bos|>")` gives `[60, 124, 32601, 124, 62]`,
  i.e. `<` `|` `bos` `|` `>` as ordinary text. Only `encode_special` produces the control id. So text from a
  document or a user can't impersonate the chat structure.

## 6. Conversations and the SFT loss mask (`render_conversation`, `tokenizer.py:140-224`)

Experiment section 5 renders a two-turn chat that includes a Python tool call. `[[...]]` marks tokens with
`mask=1`, the only ones the SFT loss trains on:

```
<|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>[[2+2 = 4.<|assistant_end|>]]
<|user_start|>And times 3?<|user_end|><|assistant_start|>[[Let me compute: <|python_start|>4*3<|python_end|>]]
<|output_start|>12<|output_end|>[[ The answer is 12.<|assistant_end|>]]

47 tokens, 25 of them trained on
```

The rules:

| Part | Trained on? | Why |
|---|---|---|
| `<|bos|>`, user turns | no | The model should *read* users, not imitate them. |
| `<|assistant_start|>` | no | At chat time the system inserts it to tell the model "your turn". |
| assistant text | **yes** | This is the behaviour we want. |
| `<|assistant_end|>` | **yes** | The model must learn *when to stop talking*. |
| `<|python_start|>` … `<|python_end|>` | **yes** | The model must learn to *call* the tool. |
| `<|output_start|>` … `<|output_end|>` | no | Tool outputs come from Python at run time, so the model shouldn't learn to make them up. |

A system message, if present, is merged into the first user message (`tokenizer.py:157-163`). Conversations
are truncated to 2048 tokens.

**`render_for_completion`** (`tokenizer.py:241-259`, experiment section 6) drops the final assistant reply and
appends `<|assistant_start|>`, leaving the model primed to answer. RL and chat eval use it to ask the model
questions.

## Exercises

1. Run `show()` from the experiment on your own sentences: English vs Korean vs a URL vs an email address.
2. Why does `encode` treat `"<|bos|>"` as ordinary text? What could go wrong if it didn't?
3. In section 5, which token ids have mask=1 but would *never* be typed by a user? (Answer: `<|assistant_end|>`,
   `<|python_start|>`, `<|python_end|>`, which the model itself has to produce.)
