"""Tokenize a slice of pretrain_t2t_mini.jsonl into uint16 train/val bins + byte counts (for bits-per-byte)."""
import json, sys, numpy as np
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
tok = Tokenizer.from_file(str(ROOT / "minimind/model/tokenizer.json"))
EOS = tok.token_to_id("<|im_end|>")

docs = []
with open(ROOT / "data/pretrain_t2t_mini.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= N_DOCS: break
        docs.append(json.loads(line)["text"])

n_val = max(1000, len(docs) // 100)
for split, chunk in (("val", docs[:n_val]), ("train", docs[n_val:])):
    ids, nbytes = [], 0
    for enc, d in zip(tok.encode_batch(chunk), chunk):
        ids.extend(enc.ids); ids.append(EOS); nbytes += len(d.encode("utf-8"))
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(ROOT / f"data/{split}.bin")
    json.dump({"tokens": len(arr), "bytes": nbytes}, open(ROOT / f"data/{split}.json", "w"))
    print(split, len(chunk), "docs", len(arr), "tokens", nbytes, "bytes", f"{nbytes/len(arr):.2f} B/tok")
