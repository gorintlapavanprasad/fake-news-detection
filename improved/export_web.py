"""
Train the browser model and export it to JSON so the website can run it entirely
in the browser (no server, so it works on GitHub Pages).

    python export_web.py --isot ../Prototype/data/raw --ext external/welfake_clean.csv

Steps
 1. pick the regularisation strength C by the WORSE of the two held-out accuracies
    (so a model cannot win by being strong on one source and weak on the other)
 2. write docs/model.json   (final model, trained on ALL ISOT + WELFake articles)
 3. write docs/parity.json  (texts + Python probabilities for tests/test_parity.mjs)
 4. write build/model_split.json + build/eval.json: a model trained on the train split
    only, plus held-out articles and an unseen third source (GossipCop), used by
    tests/eval_js_engine.mjs for an honest accuracy check of the JavaScript engine.
"""

import argparse
import glob
import json
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import fnd

MAX_FEATURES = 30000
EVAL_N = 1500          # held-out articles per source in build/eval.json
TEXT_CAP = 3000        # characters kept per article in eval.json


def build(C):
    return make_pipeline(
        TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        LogisticRegression(max_iter=3000, C=C))


def dump_model(pipe, path, metrics):
    tf, lr = pipe.steps[0][1], pipe.steps[1][1]
    json.dump({
        "vocab": {w: [round(float(i), 5), round(float(c), 5)]
                  for w, i, c in zip(tf.get_feature_names_out(), tf.idf_, lr.coef_[0])},
        "intercept": round(float(lr.intercept_[0]), 5),
        "metrics": metrics,
    }, open(path, "w"), separators=(",", ":"))


def load_gossipcop(root):
    """Human-written GossipCop articles (HF = fake, HR = real) and machine-written
    variants (MF/MR). Entirely unseen by every model here."""
    out = {}
    for kind, files in (("human", ("HF", "HR")), ("machine", ("MF", "MR"))):
        parts = []
        for code, label in ((files[0], 0), (files[1], 1)):
            for p in glob.glob(os.path.join(root, "data", f"{code}-*.parquet")):
                d = pd.read_parquet(p)[["title", "text"]].copy()
                d["label"] = label
                parts.append(d)
        if parts:
            d = pd.concat(parts, ignore_index=True)
            d["title"], d["text"] = d["title"].fillna("").astype(str), d["text"].fillna("").astype(str)
            out[kind] = d.drop_duplicates(["title", "text"]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isot", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--gossip", default="external/GossipCop")
    ap.add_argument("--C", type=float, default=0, help="skip the search and use this C")
    ap.add_argument("--out", default="../docs")
    ap.add_argument("--build", default="../build")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.build, exist_ok=True)

    isot = fnd.load_raw(a.isot)[["title", "text", "label"]]
    ext = pd.read_csv(a.ext)
    ext["title"], ext["text"] = ext["title"].fillna("").astype(str), ext["text"].fillna("").astype(str)
    feat = lambda d: fnd.build_field(d, "title+text", 0, True)

    split = {n: train_test_split(d, test_size=0.2, stratify=d.label, random_state=42)
             for n, d in [("ISOT", isot), ("WELFake", ext)]}
    tr = pd.concat([split["ISOT"][0], split["WELFake"][0]])
    Xtr = feat(tr)
    Xte = {n: feat(te) for n, (_, te) in split.items()}

    best = None
    for C in ((a.C,) if a.C else (3, 10, 30)):
        m = build(C).fit(Xtr, tr.label)
        acc = {n: round(accuracy_score(te.label, m.predict(Xte[n])), 4) for n, (_, te) in split.items()}
        worst = min(acc.values())
        print(f"C={C}: {acc} worst={worst}", flush=True)
        if best is None or worst > best[0]:
            best = (worst, C, m, acc)
    _, C, m_split, metrics_acc = best
    print("chosen C =", C)

    gossip = load_gossipcop(a.gossip)
    meta = {"heldout_acc": metrics_acc, "C": C, "n_train": 0,
            "sources": ["ISOT (Reuters vs fake sites)", "WELFake (multi-source)"]}
    dump_model(m_split, os.path.join(a.build, "model_split.json"), meta)

    # eval set for the JS engine
    ev = {}
    for n, (_, te) in split.items():
        s = te.sample(min(EVAL_N, len(te)), random_state=7)
        ev[n] = [{"title": r.title, "text": r.text[:TEXT_CAP], "label": int(r.label)} for r in s.itertuples()]
    for kind, d in gossip.items():
        s = d.sample(min(EVAL_N, len(d)), random_state=7)
        ev["GossipCop_" + kind] = [{"title": r.title, "text": r.text[:TEXT_CAP], "label": int(r.label)}
                                   for r in s.itertuples()]
    json.dump(ev, open(os.path.join(a.build, "eval.json"), "w"))
    # Python's own numbers on exactly the same truncated texts, for comparison
    py = {}
    for k, rows in ev.items():
        d = pd.DataFrame(rows)
        py[k] = round(accuracy_score(d.label, m_split.predict(feat(d))), 4)
    json.dump(py, open(os.path.join(a.build, "eval_python.json"), "w"))
    print("python accuracy on eval sets:", py)
    meta["unseen_acc"] = {k: v for k, v in py.items() if k.startswith("GossipCop")}

    # final shipped model: all data
    allx = pd.concat([isot, ext], ignore_index=True)
    final = build(C).fit(feat(allx), allx.label)
    meta["n_train"] = int(len(allx))
    dump_model(final, os.path.join(a.out, "model.json"), meta)

    sample = pd.concat([split["ISOT"][1].sample(10, random_state=1),
                        split["WELFake"][1].sample(10, random_state=1)])
    texts = [(r.title, r.text) for r in sample.itertuples()]
    texts += [("", "WASHINGTON (Reuters) - The Senate voted on Tuesday."),
              ("Big news - The New York Times", "Visit http://x.com now, it's 100% true!!!"),
              ("", "hi"), ("", ""), ("A  B", "c d e f g h")]
    df = pd.DataFrame(texts, columns=["title", "text"])
    p = final.predict_proba(feat(df))[:, 1]
    json.dump([{"title": t, "text": x, "p_real": float(q)} for (t, x), q in zip(texts, p)],
              open(os.path.join(a.out, "parity.json"), "w"))
    print("wrote", a.out, "features:", len(final.steps[0][1].vocabulary_))


if __name__ == "__main__":
    main()
