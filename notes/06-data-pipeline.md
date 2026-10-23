# 06 — The data pipeline: parquet shards → documents → (x, y) batches

> Companion experiment: [experiments/inspect_data.py](experiments/inspect_data.py)
> (`.venv\Scripts\python.exe -m notes.experiments.inspect_data`, about 30 s). Code: `nanochat/dataset.py` and
> `nanochat/dataloader.py` at commit `92d63d4`. Outputs are from runs on 2026-10-23 with the 4+1 shards on this machine.

```
HuggingFace parquet shards ──download──▶ ~/.cache/nanochat/base_data_climbmix/
        │  _document_batches: shards → row groups → chunks of 128 documents (text)
        ▼
  tokenizer.encode(docs, prepend=<|bos|>)        (4 threads)
        │  buffer of ≥1000 tokenized documents
        ▼
  best-fit packing into rows of T+1 tokens
        │  x = row[:-1], y = row[1:]
        ▼
  (x, y) of shape (B, T), plus a resume state {pq_idx, rg_idx, epoch}
```

## 1. Where the data comes from (`nanochat/dataset.py`)

- **ClimbMix-400B**, pre-shuffled and re-packaged as `karpathy/climbmix-400b-shuffle` on HuggingFace
  (`dataset.py:23`). 6,543 shards, `shard_00000` to `shard_06542`. The last one is **always** the validation
  shard.
- `python -m nanochat.dataset -n N` downloads the first N training shards **plus** the validation shard, with
  5 retries and exponential backoff, writing to a `.tmp` file first so a half-finished download never looks
  complete (`dataset.py:84-133`).
- The repo switched from FineWeb-Edu-100B to ClimbMix in March 2026; the legacy fallback is at
  `dataset.py:36-58`. ClimbMix was the "change dataset" row on the README leaderboard, which cut time-to-GPT-2
  from 2.76 to 2.02 hours.

What's on this machine (experiment section 1):

```
shard_00000.parquet    92.3 MB   86,016 docs   84 row groups  -> train
shard_00001.parquet    92.1 MB   86,016 docs   84 row groups  -> train
shard_00002.parquet    91.8 MB   84,992 docs   83 row groups  -> train
shard_00003.parquet    91.8 MB   83,968 docs   82 row groups  -> train
shard_06542.parquet    91.7 MB   84,992 docs   83 row groups  -> val
```

A row group is a chunk of 1,024 documents, the unit parquet reads at once. Rough size: ~85K docs × ~631
tokens ≈ 54M tokens per shard with our tokenizer, so the README's suggested `-n 170` for a GPT-2 run is about
9B tokens (my estimate from this sample).

The documents are ordinary web pages, e.g.
`"Protect your business and employee's during flu seaon\n\nStarting in 2005, the Center for Disease Control..."`.
Typos and scraping debris included.

## 2. Streaming documents (`_document_batches`, `dataloader.py:25-71`)

- An **infinite** loop: shards → row groups → slices of 128 documents. When the shards run out, `epoch` goes up
  and it starts again.
- **Train = every shard but the last; val = the last** (`dataloader.py:38`).
- **Multi-GPU**: rank *r* of *N* reads row groups r, r+N, r+2N, … (`dataloader.py:62-68`), so the GPUs never
  see the same documents.
- **Resume**: each yield carries `(pq_idx, rg_idx, epoch)`. On resume the loader skips to the *next* row group
  after the saved one (`dataloader.py:53-60`). Approximate, but it never repeats data.

## 3. Packing rows (`tokenizing_distributed_data_loader_with_state_bos_bestfit`, `dataloader.py:74-161`)

Each row has `T+1` slots. For each row:

1. Top the buffer up to at least 1,000 tokenized documents, each starting with `<|bos|>`.
2. Put in the **largest** document that fits entirely. Repeat.
3. When nothing fits, take the **shortest** buffered document, **crop** it to fill the row exactly, and throw the
   rest of it away.

Then `x = row[:-1]` and `y = row[1:]`. There's no padding: every slot is a real token that gets trained on. On
CUDA, rows are staged in pinned memory and copied to the GPU in one non-blocking transfer.

One real batch (experiment section 2, B=2, T=48):

```
x shape (2, 48)  y shape (2, 48)  loader state {'pq_idx': 0, 'rg_idx': 0, 'epoch': 1}
row 0: "<|bos|>Menu\n\nMonth: June 2019June that are leaking and falling down, there's mold, ..."
row 1: "<|bos|>The behind-the-scenes docu-series capturing the inaugural season of the pioneering Extreme E ..."
y is x shifted left by one: True

input token -> target token
 '<|bos|>' -> 'Menu'
 'Menu'    -> '\n\n'
 '\n\n'    -> 'Month'
 'Month'   -> ':'
 ':'       -> ' June'
 ' June'   -> ' '
 ' '       -> '20'
 '20'      -> '19'
```

Every position is a training example: "given everything up to here, predict the next token". One row of
T tokens gives T examples at once. Causal attention is what makes that possible (see [07](07-gpt-forward-pass.md)).

Rows can hold several documents, and a token can attend to an *earlier document in the same row*. nanochat
doesn't mask that. `dev/LOG.md`, entry "2026-01-13: Varlen Attention (Negative Result)": blocking
cross-document attention in a d16 run moved val bpb from 0.85427 to 0.85407, which is noise. Conclusion there:
"the model handles it fine". A nice example of a reasonable idea that measurement showed wasn't worth the code.

## 4. How much text does packing throw away? (experiment section 3)

The loader's docstring says "~35% tokens cropped at T=2048". To measure it, the experiment re-implements the
packing rule on document lengths. It first checks the copy against the real loader:

```
re-implementation matches the real loader on 12 rows: True
document start positions in the first 3 rows: [[0, 512], [0, 511], [0, 511]]
fast sorted-buffer version agrees with the exact one over 2000 rows: True
```

(`[0, 512]` means one document of exactly 512 tokens filled the row, then the shortest buffered document was
cropped to just its `<|bos|>` to fill the last slot. With 1,000 documents to choose from, best-fit usually
finds a near-perfect fit.)

Document lengths:

```
median 528, mean 631, max 31,962 tokens (10,240 docs)
longer than a row of 513 tokens:  51.1% of docs, holding 80.6% of all tokens
longer than a row of 2049 tokens:  3.2% of docs, holding 21.8% of all tokens
```

Crop rate over time, in windows of 5,000 rows:

```
T=512:  18.8%  26.2%  30.0%  31.4%  32.9%  33.9%  34.8%  34.5%  35.3%  36.0%
T=2048:  3.3%   5.2%   7.7%   9.3%  10.1%  10.7%  11.7%  11.6%  11.3%  12.0%
after 50,000 rows, every buffered document is longer than a row and still waiting
```

**Why it climbs.** A document longer than a row can never "fit", so it only leaves the buffer when it is the
shortest one left, and then it gets cropped. Long documents pile up until they fill all ~1,000 buffer slots.
From then on the dataloader regularly has to crop one. The first few thousand rows look cheap, but the steady
state is what matters for a real run, which makes hundreds of thousands of rows.

**What it means.**
- At **T=2048** (real runs), this data levels off around **12%** discarded, not the docstring's ~35%. The
  docstring might come from the FineWeb-Edu era, when documents had a different length mix. That's a guess I
  haven't checked.
- At **T=512** (the CPU runs in these notes), about **35%** is discarded, because half of all documents are
  longer than a row.
- **Why accept any waste?** Every row starts at a document boundary with `<|bos|>`, so no row begins
  mid-document, where the first tokens would have no context ("confusing tokens", per the docstring). The
  older loader (linked in the docstring) wasted nothing but cut documents anywhere. With 400B tokens available,
  data is cheap and clean context is worth it.

## 5. The validation loader

`base_train.py:332` builds the val loader with a lambda, and every evaluation calls it again (`L423`). So each
eval starts from the beginning of the val shard, and **every evaluation scores the same tokens**. That's why
val bpb is comparable from step to step, and why it is smoother than the training loss.

## Exercises

1. In the experiment, change `B, T` in section 2 to `4, 1024`. How many documents start in each row now?
2. Why does `simulate_packing` count `row_capacity = T + 1` slots instead of T? (Hint: section 2's shift.)
3. The crop rate at T=512 is 3× the rate at T=2048. Using the document-length numbers, explain why.
