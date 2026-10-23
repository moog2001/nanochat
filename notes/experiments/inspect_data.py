"""
Look inside the pretraining data pipeline: parquet shards -> documents -> packed (x, y) batches.
Companion to notes/06-data-pipeline.md.

Run from the repo root (needs the data shards and the tokenizer):
    .venv\Scripts\python.exe -m notes.experiments.inspect_data
"""
import os
import bisect
import statistics
import pyarrow.parquet as pq

from nanochat.dataset import list_parquet_files, parquets_iter_batched
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

tok = get_tokenizer()
bos = tok.get_bos_token_id()

def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

# -----------------------------------------------------------------------------
header("1) The shards on disk")
paths = list_parquet_files()
for i, p in enumerate(paths):
    pf = pq.ParquetFile(p)
    role = "val" if i == len(paths) - 1 else "train"
    print(f"{os.path.basename(p)}  {os.path.getsize(p) / 1e6:6.1f} MB  {pf.metadata.num_rows:7,d} docs  "
          f"{pf.num_row_groups:3d} row groups  -> {role}")
first_docs = next(parquets_iter_batched(split="train"))
print(f"\nfirst row group of the first train shard: {len(first_docs)} documents. The first two, cut to 160 chars:")
for doc in first_docs[:2]:
    print("  ", repr(doc[:160]))

# -----------------------------------------------------------------------------
header("2) One real batch from the training loader (B=2, T=48)")
loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tok, 2, 48, "train", device="cpu")
x, y, state = next(loader)
print("x shape", tuple(x.shape), " y shape", tuple(y.shape), " loader state", state)
for r in range(x.size(0)):
    row = x[r].tolist()
    print(f"\nrow {r}: {row.count(bos)} <|bos|> tokens = documents starting in this row")
    print("  ", repr(tok.decode(row)))
print("\ny is x shifted left by one:", bool((x[:, 1:] == y[:, :-1]).all()))
print("first 8 training pairs of row 0 (input token -> target token):")
for t in range(8):
    print(f"   {tok.decode([x[0, t].item()])!r:14} -> {tok.decode([y[0, t].item()])!r}")

# -----------------------------------------------------------------------------
header("3) How much text gets cropped by best-fit packing?")

def _tokenize_stream():
    """Same document stream as the loader on a single rank: row groups in order, 128 docs per tokenizer call."""
    for texts in parquets_iter_batched(split="train"):
        for i in range(0, len(texts), 128):
            yield [len(ids) for ids in tok.encode(texts[i:i + 128], prepend=bos, num_threads=4)]

_cache, _source = [], _tokenize_stream()
def tokenized_batches():
    """Replayable version of the stream above (document lengths are cached after the first pass)."""
    i = 0
    while True:
        if i == len(_cache):
            _cache.append(next(_source))
        yield _cache[i]
        i += 1

def simulate_packing(T, n_rows, buffer_size=1000):
    """The packing rule of nanochat/dataloader.py L122-151, run on document lengths only."""
    batches = tokenized_batches()
    buffer, rows, emitted, cropped = [], [], 0, 0
    for _ in range(n_rows):
        pos, starts = 0, []
        while pos < T + 1:
            while len(buffer) < buffer_size:
                buffer.extend(next(batches))
            remaining = T + 1 - pos
            best_idx, best_len = -1, 0
            for i, length in enumerate(buffer):  # largest document that fits entirely
                if length <= remaining and length > best_len:
                    best_idx, best_len = i, length
            starts.append(pos)
            if best_idx >= 0:
                pos += buffer.pop(best_idx)
            else:  # nothing fits: crop the shortest document to fill the row exactly
                shortest = min(range(len(buffer)), key=lambda i: buffer[i])
                cropped += buffer.pop(shortest) - remaining
                pos += remaining
        rows.append(starts)
        emitted += T + 1
    return rows, cropped

def packing_over_time(T, n_rows, window, buffer_size=1000):
    """Same rule, but with the buffer kept sorted so long runs are fast. Only lengths matter here,
    so which of several equal-length documents gets picked doesn't change any of the numbers."""
    batches = tokenized_batches()
    buf, windows, total_cropped = [], [], 0
    win_emitted = win_cropped = 0
    for r in range(n_rows):
        pos = 0
        while pos < T + 1:
            while len(buf) < buffer_size:
                for length in next(batches):
                    bisect.insort(buf, length)
            remaining = T + 1 - pos
            i = bisect.bisect_right(buf, remaining) - 1  # largest document that fits entirely
            if i >= 0:
                pos += buf.pop(i)
            else:  # nothing fits: crop the shortest
                c = buf.pop(0) - remaining
                win_cropped += c
                total_cropped += c
                pos += remaining
        win_emitted += T + 1
        if (r + 1) % window == 0:
            windows.append(win_cropped / (win_emitted + win_cropped))
            win_emitted = win_cropped = 0
    waiting = sum(1 for length in buf if length > T + 1)
    return windows, total_cropped, waiting, len(buf)

# check the re-implementation against the real loader: document start positions must match exactly
B, T = 4, 512
real_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tok, B, T, "train", device="cpu")
real_rows = []
for _ in range(3):
    xb, yb, _ = next(real_loader)
    for r in range(B):
        full_row = xb[r].tolist() + [yb[r, -1].item()]  # the T+1 tokens the row was built from
        real_rows.append([i for i, t in enumerate(full_row) if t == bos])
sim_rows, _ = simulate_packing(T, len(real_rows))
print(f"re-implementation matches the real loader on {len(real_rows)} rows: {sim_rows == real_rows}")
print("document start positions in the first 3 rows:", real_rows[:3])
_, exact_cropped = simulate_packing(T, 2000)
_, fast_cropped, _, _ = packing_over_time(T, 2000, window=2000)
print(f"fast sorted-buffer version agrees with the exact one over 2000 rows: {exact_cropped == fast_cropped}")

lengths = [n for batch, _ in zip(tokenized_batches(), range(80)) for n in batch]  # 80 x 128 = 10,240 docs
print(f"\ndocument lengths in tokens over {len(lengths):,} docs: median {statistics.median(lengths):.0f}, "
      f"mean {statistics.mean(lengths):.0f}, max {max(lengths):,}")
for limit in (513, 2049):
    share_docs = sum(n > limit for n in lengths) / len(lengths)
    share_tokens = sum(n for n in lengths if n > limit) / sum(lengths)
    print(f"  longer than a row of {limit} tokens: {share_docs:.1%} of docs, holding {share_tokens:.1%} of all tokens")

n_rows, window = 50000, 5000  # takes a minute or two: mostly tokenizing ~200K documents
for T in (512, 2048):
    windows, _, waiting, buffered = packing_over_time(T, n_rows, window)
    print(f"\nT={T}: share of document tokens cropped away, per window of {window} rows:")
    print("   " + "  ".join(f"{w:.1%}" for w in windows))
    print(f"   after {n_rows:,} rows, {waiting} of the {buffered} buffered docs are longer than a row and still waiting")
