"""
Fine-tune DistilBERT for real/fake classification (CPU-sized).

    python train_distilbert.py --isot ../Prototype/data/raw --ext external/welfake_clean.csv \
        --train combined --n-train 3000 --out bert_combined

--train isot | welfake | combined  chooses the training source(s).
Test sets are the SAME held-out 20% splits used by cross_dataset.py (random_state=42),
subsampled to --n-test per source so that CPU evaluation stays quick.
Writes <out>/ (model + tokenizer) and results/<out>.json.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import fnd

MODEL_NAME = "distilbert-base-uncased"


def raw_text(d):
    # Same cleaner as the TF-IDF models (lowercase, no punctuation, source giveaways
    # removed) so that all models are compared on identical input.
    return fnd.build_field(d, "title+text", 0, True).tolist()


def batches(texts, labels, tok, bs, max_len, shuffle, rng=None):
    idx = np.arange(len(texts))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        enc = tok([texts[k] for k in j], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        yield enc, torch.tensor([labels[k] for k in j])


def evaluate(model, tok, texts, labels, max_len):
    model.eval()
    preds = []
    with torch.no_grad():
        for enc, _ in batches(texts, labels, tok, 32, max_len, False):
            preds += model(**enc).logits.argmax(-1).tolist()
    return {"acc": round(accuracy_score(labels, preds), 4),
            "f1": round(f1_score(labels, preds), 4), "n": len(labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isot", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--train", choices=["isot", "welfake", "combined"], default="combined")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--out", default="bert_combined")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    isot = fnd.load_raw(a.isot)[["title", "text", "label"]]
    ext = pd.read_csv(a.ext)
    ext["title"], ext["text"] = ext["title"].fillna("").astype(str), ext["text"].fillna("").astype(str)
    splits = {}
    for name, d in [("ISOT", isot), ("WELFake", ext)]:
        tr, te = train_test_split(d, test_size=0.2, stratify=d.label, random_state=42)
        splits[name] = (tr.reset_index(drop=True), te.reset_index(drop=True))

    if a.train == "isot":
        pool = splits["ISOT"][0]
    elif a.train == "welfake":
        pool = splits["WELFake"][0]
    else:
        pool = pd.concat([splits["ISOT"][0], splits["WELFake"][0]], ignore_index=True)
    train = pool.sample(min(a.n_train, len(pool)), random_state=42)
    tx, ty = raw_text(train), train.label.tolist()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(tx) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps))

    t0, step = time.time(), 0
    for ep in range(a.epochs):
        model.train()
        for enc, y in batches(tx, ty, tok, a.bs, a.max_len, True, rng):
            loss = model(**enc, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f} "
                      f"elapsed {time.time()-t0:.0f}s", flush=True)

    res = {"trained_on": a.train, "n_train": len(tx), "epochs": a.epochs,
           "max_len": a.max_len, "train_seconds": round(time.time() - t0)}
    for name in ("ISOT", "WELFake"):
        te = splits[name][1].sample(min(a.n_test, len(splits[name][1])), random_state=1)
        res[f"test_{name}"] = evaluate(model, tok, raw_text(te), te.label.tolist(), a.max_len)
        print(name, res[f"test_{name}"], flush=True)

    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/{a.out}.json", "w"), indent=2)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved", a.out, res)


if __name__ == "__main__":
    main()
