"""Run one experiment: python lab/train.py name key=val ...  -> results/runs.jsonl"""
import json, math, sys, time, numpy as np, torch, torch.nn.functional as F
from dataclasses import asdict, fields
from pathlib import Path
from model import Cfg, LM

ROOT = Path(__file__).resolve().parent.parent
name, kv = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
run = dict(seed=0, steps=1500, bs=16, accum=2, lr=3e-3, warmup=100, opt="adamw", wd=0.1, eval_every=250, seq=512)
cfg = Cfg()
for k, v in kv.items():
    tgt = run if k in run else cfg.__dict__
    cur = tgt[k]
    tgt[k] = (v.lower() == "true") if isinstance(cur, bool) else type(cur)(v)
cfg.seq = run["seq"]
torch.manual_seed(run["seed"]); np.random.seed(run["seed"])
dev = "cuda"
tr = np.memmap(ROOT / "data/train.bin", dtype=np.uint16, mode="r")
va = np.memmap(ROOT / "data/val.bin", dtype=np.uint16, mode="r")
vmeta = json.load(open(ROOT / "data/val.json")); BPT = vmeta["bytes"] / vmeta["tokens"]


def batch(d, bs, rng):
    ix = rng.integers(0, len(d) - cfg.seq - 1, bs)
    x = torch.from_numpy(np.stack([d[i:i + cfg.seq + 1].astype(np.int64) for i in ix])).to(dev)
    return x[:, :-1], x[:, 1:]


model = LM(cfg).to(dev)
decay = [p for p in model.parameters() if p.dim() >= 2]
other = [p for p in model.parameters() if p.dim() < 2]
groups = [{"params": decay, "weight_decay": run["wd"]}, {"params": other, "weight_decay": 0.0}]
if run["opt"] == "adamw":
    opt = torch.optim.AdamW(groups, lr=run["lr"], betas=(0.9, 0.95), fused=True)
elif run["opt"] == "lion":
    from opts import Lion; opt = Lion(groups, lr=run["lr"] / 10)
elif run["opt"] == "muon":
    from opts import muon_groups; opt = muon_groups(model, run["lr"], run["wd"])
else:
    raise SystemExit(f"unknown opt {run['opt']}")
for g in opt.param_groups: g["base_lr"] = g["lr"]


def lr_mult(s):
    if s < run["warmup"]: return (s + 1) / run["warmup"]
    p = (s - run["warmup"]) / max(1, run["steps"] - run["warmup"])
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(n=40):
    model.eval(); rng = np.random.default_rng(1234); tot = 0.0
    for _ in range(n):
        x, y = batch(va, 16, rng)
        with torch.autocast("cuda", torch.bfloat16):
            tot += F.cross_entropy(model(x).float().view(-1, cfg.vocab), y.reshape(-1)).item()
    model.train(); l = tot / n
    return l, l / math.log(2) / BPT  # loss (nats/token), bits per byte


rng = np.random.default_rng(run["seed"]); curve = []; t0 = time.time(); diverged = False
torch.cuda.reset_peak_memory_stats()
for s in range(run["steps"]):
    for g in opt.param_groups: g["lr"] = g["base_lr"] * lr_mult(s)
    for _ in range(run["accum"]):
        x, y = batch(tr, run["bs"], rng)
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(model(x).float().view(-1, cfg.vocab), y.reshape(-1)) / run["accum"]
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if not math.isfinite(loss.item()): diverged = True; break
    if (s + 1) % run["eval_every"] == 0 or s + 1 == run["steps"]:
        vl, bpb = evaluate(); curve.append((s + 1, round(vl, 4)))
        print(f"{name} step {s+1} val {vl:.4f} bpb {bpb:.4f} {time.time()-t0:.0f}s", flush=True)

vl, bpb = (float("nan"),) * 2 if diverged else evaluate(100)
res = dict(name=name, overrides=kv, run=run, cfg=asdict(cfg), params=model.n_params(), val_loss=vl, bpb=bpb,
           diverged=diverged, secs=round(time.time() - t0), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
           tok_per_s=round(run["steps"] * run["accum"] * run["bs"] * cfg.seq / (time.time() - t0)), curve=curve)
(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results/runs.jsonl", "a") as f: f.write(json.dumps(res) + "\n")
print("RESULT", json.dumps({k: res[k] for k in ("name", "params", "val_loss", "bpb", "secs", "peak_gb", "tok_per_s")}))
