"""Compare long-run loss curves (results/long_curve_*.jsonl): windowed means, deviations from fp32 and from oc_amp."""
import json, sys
W = 100
def load(p):
    L = [json.loads(l) for l in open(p)]
    return [r["loss"] for r in L if "loss" in r], next((r for r in L if r.get("kind")), {})
d = sys.argv[1] if len(sys.argv) > 1 else "results"
import os
C = {k: load(f"{d}/long_curve_{k}.jsonl") for k in ("oc_fp32", "oc_amp", "oc_amp_compile", "oc_amp_norr", "cf", "cf_sdpa",
                                                      "cf_eager_sdpa")
     if os.path.exists(f"{d}/long_curve_{k}.jsonl") and os.path.getsize(f"{d}/long_curve_{k}.jsonl")}
win = {k: [sum(v[0][i:i + W]) / W for i in range(0, len(v[0]) - W + 1, W)] for k, v in C.items()}
for k in [k for k in C if k != "oc_fp32"]:
    for refk in ("oc_fp32", "oc_amp"):
        if k == refk: continue
        dd = [abs(a - b) for a, b in zip(win[k], win[refk])]
        print(f"{k:7s} vs {refk:7s}: max |window-mean diff| {max(dd):.4f}  final window {win[k][-1]:.4f} vs {win[refk][-1]:.4f}")
for k, (_, s) in C.items():
    print(k, s.get("samples_per_s"), "samples/s", s.get("peak_gb"), "GB", "non-finite:", sum(1 for x in C[k][0] if x != x))
print("windows:", {k: [round(x, 3) for x in v[::5]] for k, v in win.items()})
