"""Phase 2 data: same parallel content in several 'internal languages'.
Each variant gets its own BPE tokenizer (same vocab size) -> data/p2_<variant>_{train,val}.bin
Fair metric across variants = bits per sentence (same content), stored as 'sents' in meta."""
import json, re, sys, unicodedata, numpy as np, pyarrow.parquet as pq
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

D = Path(__file__).resolve().parent.parent / "data"
VOCAB = int(sys.argv[1]) if len(sys.argv) > 1 else 6400
N = 600_000
pairs = [r["translation"] for r in pq.read_table(D / "opus_enes.parquet").to_pylist()[:N]]


def strip_acc(s): return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


VARIANTS = {
    "es": lambda p: p["es"],
    "en": lambda p: p["en"],
    # normalized Spanish: lowercase, no accents, punctuation removed (lossy but compact)
    "es_norm": lambda p: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", strip_acc(p["es"].lower()))).strip(),
}
only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)
VARIANTS["bytes"] = VARIANTS["es"]  # machine level: raw UTF-8 bytes, no tokenizer (vocab 257)
for name in only:
    texts = [VARIANTS[name](p) for p in pairs]
    if name == "bytes":
        nv = N // 100
        for split, chunk in (("val", texts[:nv]), ("train", texts[nv:])):
            ids = []
            for t in chunk: ids += list(t.encode()) + [256]
            np.array(ids, dtype=np.uint16).tofile(D / f"p2_bytes_{split}.bin")
            meta = {"tokens": len(ids), "bytes": len(ids) - len(chunk), "sents": len(chunk), "vocab": 257}
            json.dump(meta, open(D / f"p2_bytes_{split}.json", "w")); print("bytes", split, meta, flush=True)
        continue
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["<eos>"],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(D / f"p2_{name}_tok.json"))
    eos = tok.token_to_id("<eos>"); nv = N // 100
    for split, chunk in (("val", texts[:nv]), ("train", texts[nv:])):
        ids = []
        for e in tok.encode_batch(chunk): ids += e.ids + [eos]
        np.array(ids, dtype=np.uint16).tofile(D / f"p2_{name}_{split}.bin")
        meta = {"tokens": len(ids), "bytes": sum(len(t.encode()) for t in chunk), "sents": len(chunk), "vocab": VOCAB}
        json.dump(meta, open(D / f"p2_{name}_{split}.json", "w"))
        print(name, split, meta, f"{len(ids)/len(chunk):.1f} tok/frase", flush=True)
