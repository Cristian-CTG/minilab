"""Aggregate results/runs.jsonl -> results/REPORT.md + results/curves.png"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent / "results"
runs = [json.loads(l) for l in open(R / "runs.jsonl")]
g = defaultdict(list)
for r in runs: g[r["name"]].append(r)
base = st.mean(r["bpb"] for r in g["baseline"]) if "baseline" in g else None
rows = []
for n, rs in g.items():
    b = [r["bpb"] for r in rs if not r["diverged"]]
    m = st.mean(b) if b else float("nan"); sd = st.stdev(b) if len(b) > 1 else 0.0
    rows.append((m, n, sd, len(rs), rs[0]["params"], st.mean(r["tok_per_s"] for r in rs), max(r["peak_gb"] for r in rs)))
rows.sort()
out = ["| # | variante | bpb media | ± desv | Δ vs base | semillas | params | tok/s | VRAM GB |", "|---|---|---|---|---|---|---|---|---|"]
for i, (m, n, sd, k, p, tps, gb) in enumerate(rows, 1):
    d = f"{(m - base) / base * 100:+.2f}%" if base else "-"
    out.append(f"| {i} | {n} | {m:.4f} | {sd:.4f} | {d} | {k} | {p/1e6:.2f}M | {tps:.0f} | {gb} |")
(R / "REPORT.md").write_text("# MiniLab — Batch 1\n\nMenor bpb (bits por byte de validación) = mejor.\n\n" + "\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
plt.figure(figsize=(9, 5))
for n, rs in g.items():
    c = rs[0]["curve"]
    if c: plt.plot([s for s, _ in c], [v for _, v in c], label=n, lw=2.5 if n == "baseline" else 1.2)
plt.xlabel("step"); plt.ylabel("val loss (seed 0)"); plt.ylim(top=5.5); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.savefig(R / "curves.png", dpi=120, bbox_inches="tight")
