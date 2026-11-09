"""
Train the browser model on all five sources and export it to JSON so the website can
run it entirely in the browser (no server, so it works on GitHub Pages).

    python export_web.py --isot ../Prototype/data/raw --ext external

Steps
 1. split every source 80/20; train on the 80% parts (large sources capped so they do
    not dominate), measure accuracy on each source's untouched 20%
 2. write build/model_split.json + build/eval.json: that same model plus held-out
    articles, used by tests/eval_js_engine.mjs for an honest check of the JS engine
 3. write docs/model.json: the final model, trained on ALL articles, together with the
    metrics shown on the site (per-source held-out accuracy, plus the leave-one-source-out
    accuracy from results/loso.csv, i.e. news from a source never seen in training)
 4. write docs/parity.json for tests/test_parity.mjs
"""

import argparse
import json
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import fnd
import multisource

MAX_FEATURES = 30000
CAP_TRAIN = 15000      # max training articles per source
EVAL_N = 1000          # held-out articles per source in build/eval.json
TEXT_CAP = 3000        # characters kept per article in eval.json


def build(C):
    return make_pipeline(
        TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        LogisticRegression(max_iter=3000, C=C, class_weight="balanced"))


def dump_model(pipe, path, metrics):
    tf, lr = pipe.steps[0][1], pipe.steps[1][1]
    json.dump({
        "vocab": {w: [round(float(i), 5), round(float(c), 5)]
                  for w, i, c in zip(tf.get_feature_names_out(), tf.idf_, lr.coef_[0])},
        "intercept": round(float(lr.intercept_[0]), 5),
        "metrics": metrics,
    }, open(path, "w"), separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isot", required=True)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--C", type=float, default=30)
    ap.add_argument("--loso", default="results/loso.csv")
    ap.add_argument("--out", default="../docs")
    ap.add_argument("--build", default="../build")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.build, exist_ok=True)

    src = multisource.load_sources(a.isot, a.ext)
    feat = lambda d: fnd.build_field(d, "title+text", 0, True)
    split = {n: train_test_split(d, test_size=0.2, stratify=d.label, random_state=42)
             for n, d in src.items()}
    tr = pd.concat([s[0].sample(min(CAP_TRAIN, len(s[0])), random_state=1) for s in split.values()])
    m = build(a.C).fit(feat(tr), tr.label)
    heldout = {n: round(accuracy_score(te.label, m.predict(feat(te))), 4) for n, (_, te) in split.items()}
    print("held-out accuracy per source (sources seen in training):", heldout)

    # leave-one-source-out results (recipe B) from loso_eval.py
    unseen = {}
    if os.path.exists(a.loso):
        l = pd.read_csv(a.loso)
        l = l[l.recipe.str.startswith("B")]
        unseen = {r.held_out: float(r.accuracy) for r in l.itertuples()}
    meta = {"heldout_acc": heldout, "unseen_acc": unseen, "C": a.C,
            "n_train": int(sum(len(d) for d in src.values())),
            "sources": list(src)}
    dump_model(m, os.path.join(a.build, "model_split.json"), meta)

    ev = {}
    for n, (_, te) in split.items():
        s = te.sample(min(EVAL_N, len(te)), random_state=7)
        ev[n] = [{"title": r.title, "text": r.text[:TEXT_CAP], "label": int(r.label)} for r in s.itertuples()]
    json.dump(ev, open(os.path.join(a.build, "eval.json"), "w"))
    py = {k: round(accuracy_score([r["label"] for r in rows],
                                  m.predict(feat(pd.DataFrame(rows)))), 4) for k, rows in ev.items()}
    json.dump(py, open(os.path.join(a.build, "eval_python.json"), "w"))
    print("python accuracy on eval samples:", py)

    allx = pd.concat([d.sample(min(CAP_TRAIN * 2, len(d)), random_state=1) for d in src.values()],
                     ignore_index=True)
    final = build(a.C).fit(feat(allx), allx.label)
    dump_model(final, os.path.join(a.out, "model.json"), meta)

    sample = pd.concat([split[n][1].sample(min(5, len(split[n][1])), random_state=1) for n in src])
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
