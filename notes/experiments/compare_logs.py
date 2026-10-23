"""
Compare two base_train logs side by side: where they diverge, smoothed loss over a step range, val bpb, samples.
Used for the momentum A/B in notes/04-cpu-run-500-steps.md; works for any pair of base_train logs.

Run from the repo root:
    .venv\Scripts\python.exe -m notes.experiments.compare_logs notes/logs/cpu-run-d4-500steps.log notes/logs/cpu-run-d4-500steps-momfix.log --from-step 395 --to-step 430
"""
import re
import argparse

STEP_RE = re.compile(r"^step (\d+)/\d+ .*?\| loss: ([\d.]+)")
VAL_RE = re.compile(r"^Step (\d+) \| Validation bpb: ([\d.]+)")

def parse(path):
    losses, vals, samples = {}, {}, {}
    current_val_step = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if m := STEP_RE.match(line):
                losses[int(m.group(1))] = float(m.group(2))
            elif m := VAL_RE.match(line):
                current_val_step = int(m.group(1))
                vals[current_val_step] = float(m.group(2))
            elif line.startswith("<|bos|>") and current_val_step is not None:
                samples.setdefault(current_val_step, []).append(line)
    return losses, vals, samples

parser = argparse.ArgumentParser()
parser.add_argument("log_a")
parser.add_argument("log_b")
parser.add_argument("--from-step", type=int, default=0)
parser.add_argument("--to-step", type=int, default=10**9)
args = parser.parse_args()

loss_a, val_a, samples_a = parse(args.log_a)
loss_b, val_b, samples_b = parse(args.log_b)
common = sorted(set(loss_a) & set(loss_b))
first_diff = next((s for s in common if loss_a[s] != loss_b[s]), None)
print(f"A: {args.log_a}\nB: {args.log_b}")
print(f"steps in both logs: {len(common)}; identical loss up to step "
      f"{(first_diff - 1) if first_diff is not None else common[-1]}; first difference at step {first_diff}")

print(f"\nsmoothed train loss, steps {args.from_step}-{min(args.to_step, common[-1])}:")
print(f"{'step':>5} {'A':>9} {'B':>9} {'B - A':>8}")
for s in common:
    if args.from_step <= s <= args.to_step:
        print(f"{s:5d} {loss_a[s]:9.4f} {loss_b[s]:9.4f} {loss_b[s] - loss_a[s]:+8.4f}")
tail = [s for s in common if s >= (first_diff or 0)]
if tail:
    mean_diff = sum(loss_b[s] - loss_a[s] for s in tail) / len(tail)
    print(f"mean (B - A) over steps {tail[0]}-{tail[-1]}: {mean_diff:+.4f}")

print("\nvalidation bpb:")
for s in sorted(set(val_a) | set(val_b)):
    a, b = val_a.get(s), val_b.get(s)
    diff = f"{b - a:+.6f}" if a is not None and b is not None else ""
    print(f"  step {s:5d}: A {a}  B {b}  {diff}")

last = max(set(samples_a) & set(samples_b), default=None)
if last is not None:
    print(f"\nsamples at step {last}:")
    for sa, sb in zip(samples_a[last], samples_b[last]):
        print(f"  A: {sa}\n  B: {sb}")
