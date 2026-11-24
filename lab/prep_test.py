"""Held-out test set: docs 200000-201999 of pretrain_t2t_mini.jsonl (never seen by train/val)."""
import json, itertools, numpy as np
from pathlib import Path
from tokenizers import Tokenizer
ROOT = Path(__file__).resolve().parent.parent
tok = Tokenizer.from_file(str(ROOT / "minimind/model/tokenizer.json")); EOS = tok.token_to_id("<|im_end|>")
with open(ROOT / "data/pretrain_t2t_mini.jsonl", encoding="utf-8") as f:
    docs = [json.loads(l)["text"] for l in itertools.islice(f, 200_000, 202_000)]
ids = []
for e in tok.encode_batch(docs): ids += e.ids + [EOS]
np.array(ids, dtype=np.uint16).tofile(ROOT / "data/test.bin")
meta = {"tokens": len(ids), "bytes": sum(len(d.encode()) for d in docs)}
json.dump(meta, open(ROOT / "data/test.json", "w")); print(len(docs), meta)
