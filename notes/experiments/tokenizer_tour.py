"""
A tour of nanochat's tokenizer: token pieces, numbers, the vocabulary, special tokens, the chat loss mask.
Companion to notes/05-tokenizer.md.

Run from the repo root (needs the tokenizer trained by scripts.tok_train):
    .venv\Scripts\python.exe -m notes.experiments.tokenizer_tour
"""
from nanochat.tokenizer import get_tokenizer, get_token_bytes, SPECIAL_TOKENS

tok = get_tokenizer()

def header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

def piece(token_id):
    """Readable form of one token: its text if it is valid UTF-8, else its raw bytes."""
    raw = tok.decode_single_token_bytes(token_id)
    try:
        return repr(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return repr(raw)  # a fragment of a multi-byte character

def show(text):
    ids = tok.encode(text)
    n_bytes = len(text.encode("utf-8"))
    print(f"{text!r}")
    print(f"  {len(ids)} tokens, {n_bytes} bytes -> {n_bytes / len(ids):.2f} bytes/token")
    print("  pieces:", " ".join(piece(i) for i in ids))

# -----------------------------------------------------------------------------
header("1) Text -> token pieces")
show("Hello world! The cat sat on the mat.")
show("tokenization tokenizer untokenizable")
show("In 2026 the price was 1234567 won.")
show("def add(a, b):\n    return a + b")
show("안녕하세요, 반갑습니다.")  # Korean: "Hello, nice to meet you."

header("2) Case and leading spaces make different tokens")
for word in ["the", " the", "The", " The", " THE"]:
    print(f"{word!r:8} -> ids {tok.encode(word)}")

# -----------------------------------------------------------------------------
header("3) The vocabulary")
vocab = tok.get_vocab_size()
token_bytes = get_token_bytes()
print("vocab size:", vocab)
print("ids 0-255 are single bytes:", all(len(tok.decode_single_token_bytes(i)) == 1 for i in range(256)))
print("first merges (ids 256-275), the most frequent pairs in the training text:")
print("  ", " ".join(piece(i) for i in range(256, 276)))
special_ids = {tok.encode_special(s) for s in SPECIAL_TOKENS}
last_normal = min(special_ids) - 1
print(f"last merges (ids {last_normal - 9}-{last_normal}), the rarest:")
print("  ", " ".join(piece(i) for i in range(last_normal - 9, last_normal + 1)))
normal = token_bytes[token_bytes > 0].float()
print(f"bytes per vocabulary entry: mean {normal.mean().item():.2f}, max {int(normal.max().item())}")
longest = sorted(range(vocab), key=lambda i: int(token_bytes[i]), reverse=True)[:8]
print("longest tokens:", " ".join(piece(i) for i in longest))

# -----------------------------------------------------------------------------
header("4) Special tokens")
for s in SPECIAL_TOKENS:
    print(f"  {tok.encode_special(s):6d}  {s}")
print("encode('<|bos|>') treats it as ordinary text:", tok.encode("<|bos|>"))
print("encode_special('<|bos|>') gives the one control token:", tok.encode_special("<|bos|>"))
print("encode('hi', prepend=bos):", tok.encode("hi", prepend=tok.get_bos_token_id()))

# -----------------------------------------------------------------------------
header("5) A conversation rendered for SFT, with its loss mask")
conversation = {"messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 = 4."},
    {"role": "user", "content": "And times 3?"},
    {"role": "assistant", "content": [
        {"type": "text", "text": "Let me compute: "},
        {"type": "python", "text": "4*3"},
        {"type": "python_output", "text": "12"},
        {"type": "text", "text": " The answer is 12."},
    ]},
]}
ids, mask = tok.render_conversation(conversation)
print(f"{len(ids)} tokens, {sum(mask)} of them trained on (mask=1)")
print("legend: [[ trained on ]]   everything else is context only")
groups, current, current_mask = [], [], mask[0]
for token_id, m in zip(ids, mask):
    if m != current_mask:
        groups.append((current_mask, current))
        current, current_mask = [], m
    current.append(token_id)
groups.append((current_mask, current))
print("".join(f"[[{tok.decode(g)}]]" if m else tok.decode(g) for m, g in groups))

header("6) The same conversation primed for a completion (used by RL and chat eval)")
prompt_ids = tok.render_for_completion(conversation)
print("ends with:", repr(tok.decode(prompt_ids[-12:])))
