"""
A/B experiment: does the Muon momentum jump at step 400 cause the loss bump seen in the 500-step CPU run?
Companion to notes/04-cpu-run-500-steps.md.

Run A is the unmodified scripts/base_train.py: in runs shorter than ~1,140 steps its momentum schedule ramps up
until step 400 and then jumps down to the warmdown line (0.970 -> 0.922 in a 500-step run).
Run B (this script) runs the same file with one change: from step 400 on, momentum falls smoothly from 0.97 to
0.90 instead of jumping. Steps 0-399 are identical in both runs (same seed, same data order), so any
difference afterwards comes from this change alone. The repo file is not modified: its source is read,
patched in memory, and executed.

Run from the repo root, with the same flags as run A:
    $env:TORCHDYNAMO_DISABLE = "1"
    .venv\Scripts\python.exe -m notes.experiments.momentum_ab --depth=4 --head-dim=64 --window-pattern=L --max-seq-len=512 --device-batch-size=4 --total-batch-size=4096 --eval-every=100 --eval-tokens=16384 --core-metric-every=-1 --sample-every=100 --num-iterations=500 --model-tag=d4-500-momfix
"""
import sys

ORIGINAL = """    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:"""

PATCHED = """    if it >= 400 and warmdown_start < 400:  # momentum_ab patch: no jump, glide from 0.97 to 0.90
        progress = (it - 400) / max(num_iterations - 400, 1)
        return 0.97 * (1 - progress) + 0.90 * progress
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:"""

path = "scripts/base_train.py"
with open(path, encoding="utf-8") as f:
    source = f.read()
assert source.count(ORIGINAL) == 1, "base_train.py changed; update ORIGINAL in this script"
sys.argv[0] = path
print("momentum_ab: running scripts/base_train.py with the smooth step-400+ momentum patch")
exec(compile(source.replace(ORIGINAL, PATCHED), path + " (momentum_ab patch)", "exec"), {"__name__": "__main__"})
