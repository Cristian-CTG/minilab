"""Tokenize the first N docs of pretrain_t2t_mini.jsonl into uint16 train/val bins + byte counts (for bits-per-byte).
Val = first 2000 docs (fixed). Docs 200000-201999 are skipped: they are the held-out test set (prep_test.py).
Streams in chunks, so it also works for the whole file: python lab/prep.py 1000000"""
import json, sys, numpy as np
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
tok = Tokenizer.from_file(str(ROOT / "minimind/model/tokenizer.json"))
EOS = tok.token_to_id("<|im_end|>")
N_VAL = min(2000, max(1000, N_DOCS // 100))
out = {s: open(ROOT / f"data/{s}.bin", "wb") for s in ("val", "train")}
meta = {s: {"tokens": 0, "bytes": 0, "docs": 0} for s in out}


def flush(split, chunk):
    ids = []
    for enc, d in zip(tok.encode_batch(chunk), chunk):
        ids.extend(enc.ids); ids.append(EOS); meta[split]["bytes"] += len(d.encode("utf-8"))
    np.array(ids, dtype=np.uint16).tofile(out[split]); meta[split]["tokens"] += len(ids); meta[split]["docs"] += len(chunk)


buf = {"val": [], "train": []}
with open(ROOT / "data/pretrain_t2t_mini.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= N_DOCS: break
        if 200_000 <= i < 202_000: continue
        s = "val" if i < N_VAL else "train"
        buf[s].append(json.loads(line)["text"])
        if len(buf[s]) >= 20_000: flush(s, buf[s]); buf[s] = []
for s in buf:
    if buf[s]: flush(s, buf[s])
    out[s].close(); m = meta[s]
    json.dump({"tokens": m["tokens"], "bytes": m["bytes"]}, open(ROOT / f"data/{s}.json", "w"))
    print(s, m["docs"], "docs", m["tokens"], "tokens", m["bytes"], "bytes", f"{m['bytes']/m['tokens']:.2f} B/tok")
