"""Phase 3: does a model reason better in text or in vectors?  Exact-answer arithmetic.
python lab/latent.py name mode=direct|cot|pause|latent terms=6 K=5 steps=3000 seed=0 dev=cuda
  direct: "12+45+...=" -> answer
  cot:    model writes partial sums in text, then answer   (human-language reasoning)
  pause:  K dummy tokens before the answer                 (control: extra compute, no feedback)
  latent: K steps feeding the last hidden state back as the next input (Coconut-style)
"""
import json, sys, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from model import Cfg, LM

ROOT = Path(__file__).resolve().parent.parent
name, kv = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
a = dict(mode="direct", terms=6, K=5, steps=3000, bs=256, lr=1e-3, seed=0, dev="cuda", d=128, layers=4)
for k, v in kv.items(): a[k] = type(a[k])(v)
torch.manual_seed(a["seed"]); rng = np.random.default_rng(a["seed"])
CH = "0123456789+=;|"; PAD, BOT, EOS = len(CH), len(CH) + 1, len(CH) + 2
enc = lambda s: [CH.index(c) for c in s]


def sample(n):
    nums = rng.integers(0, 100, (n, a["terms"])); ps = nums.cumsum(1)
    P, T = [], []
    for row, p in zip(nums, ps):
        P.append(enc("+".join(f"{x:02d}" for x in row) + "="))
        ans = enc(f"{p[-1]:04d}")
        T.append((enc(";".join(f"{x:04d}" for x in p[:-1]) + "|") if a["mode"] == "cot" else []) + ans + [EOS])
    return torch.tensor(P), torch.tensor(T), ps[:, -1]


nlat = a["K"] if a["mode"] in ("pause", "latent") else 0
P0, T0, _ = sample(1)
cfg = Cfg(vocab=PAD + 3, d=a["d"], layers=a["layers"], heads=4, kv_heads=2, seq=P0.shape[1] + nlat + T0.shape[1] + 1)
dev = a["dev"]; m = LM(cfg).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=a["lr"], weight_decay=0.1)


def hidden(x):  # x: embeddings (B,T,d) -> final normed states
    T = x.shape[1]; cos, sin = m.cos[:T], m.sin[:T]
    for b in m.blocks: x = b(x, cos, sin)
    return m.norm(x)


def prefix(P):  # prompt (+ K pause tokens or K latent thoughts) as embeddings
    x = m.emb(P)
    if a["mode"] == "pause":
        x = torch.cat([x, m.emb(torch.full((P.shape[0], nlat), BOT, device=dev))], 1)
    elif a["mode"] == "latent":
        scale = m.emb.weight.std()
        for _ in range(nlat): x = torch.cat([x, hidden(x)[:, -1:] * scale], 1)
    return x


def loss_fn(P, T):
    x = torch.cat([prefix(P), m.emb(T[:, :-1])], 1)
    logits = m.head(hidden(x))[:, -T.shape[1]:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), T.reshape(-1))


@torch.no_grad()
def accuracy(n=2000):
    P, T, ans = sample(n); P = P.to(dev); x = prefix(P); out = []
    for _ in range(T.shape[1]):
        nxt = m.head(hidden(x)[:, -1]).argmax(-1); out.append(nxt); x = torch.cat([x, m.emb(nxt)[:, None]], 1)
    out = torch.stack(out, 1).cpu()[:, -5:-1]  # 4 answer digits before EOS
    pred = [int("".join(CH[i] for i in r)) if all(i < 10 for i in r) else -1 for r in out.tolist()]
    return float(np.mean(np.array(pred) == ans))


t0 = time.time(); curve = []
for s in range(a["steps"]):
    lr = a["lr"] * min(1, (s + 1) / 100) * (0.1 + 0.9 * (1 - s / a["steps"]))
    for g in opt.param_groups: g["lr"] = lr
    P, T, _ = sample(a["bs"])
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        loss = loss_fn(P.to(dev), T.to(dev))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if (s + 1) % max(1, a["steps"] // 6) == 0:
        acc = accuracy(500); curve.append((s + 1, acc))
        print(f"{name} step {s+1} loss {loss.item():.4f} acc {acc:.3f} {time.time()-t0:.0f}s", flush=True)
res = dict(name=name, args=a, params=m.n_params(), acc=accuracy(), secs=round(time.time() - t0), curve=curve,
           fwd_passes_per_answer=1 + nlat if a["mode"] == "latent" else 1)
with open(ROOT / "results/latent.jsonl", "a") as f: f.write(json.dumps(res) + "\n")
print("RESULT", json.dumps({k: res[k] for k in ("name", "params", "acc", "secs")}))
