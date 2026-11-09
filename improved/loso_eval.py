"""
Leave-one-source-out (LOSO) evaluation: train on every source except one, test on the
source that was left out. This is the honest answer to "does it work on news from
somewhere it has never seen?"  It also records, per article, how much of the text the
model knew, so we can test whether a low-coverage warning predicts mistakes.

    python loso_eval.py --isot ../Prototype/data/raw --ext external

Outputs: results/loso.csv (accuracy per held-out source for two training recipes)
         results/loso_docs.csv (per-article confidence/coverage/correctness, recipe B only)
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import fnd
import multisource

CAP_TRAIN = 12000   # max training articles per source, so big sources do not dominate
CAP_TEST = 3000


def fit(train_df, C=30):
    X = fnd.build_field(train_df, "title+text", 0, True)
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced")
    clf.fit(vec.fit_transform(X), train_df.label)
    return vec, clf


def coverage(vec, texts):
    """Share of word tokens (len >= 2) that are known single-word features."""
    voc = vec.vocabulary_
    out = []
    for t in texts:
        toks = [w for w in t.split() if len(w) >= 2]
        out.append(sum(w in voc for w in toks) / len(toks) if toks else 0.0)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isot", required=True)
    ap.add_argument("--ext", required=True)
    a = ap.parse_args()

    src = multisource.load_sources(a.isot, a.ext)
    for k, v in src.items():
        print(f"{k:11s} n={len(v):6d} real={int((v.label == 1).sum()):6d} fake={int((v.label == 0).sum()):6d}")
    names = list(src)
    rows, docs = [], []
    rng = 1
    for held in names:
        te = src[held].sample(min(CAP_TEST, len(src[held])), random_state=rng)
        for recipe, pool in (("A: ISOT+WELFake only", ["ISOT", "WELFake"]),
                             ("B: all other sources", [n for n in names if n != held])):
            pool = [n for n in pool if n != held]
            if not pool:
                continue
            tr = pd.concat([src[n].sample(min(CAP_TRAIN, len(src[n])), random_state=rng) for n in pool])
            vec, clf = fit(tr)
            Xte = fnd.build_field(te, "title+text", 0, True)
            proba = clf.predict_proba(vec.transform(Xte))[:, 1]
            pred = (proba >= 0.5).astype(int)
            acc = accuracy_score(te.label, pred)
            base = max(te.label.mean(), 1 - te.label.mean())
            rows.append({"held_out": held, "recipe": recipe, "n_test": len(te),
                         "accuracy": round(acc, 4), "majority_baseline": round(base, 4)})
            print(rows[-1], flush=True)
            if recipe.startswith("B"):
                cov = coverage(vec, Xte.tolist())
                docs.append(pd.DataFrame({"held_out": held, "correct": (pred == te.label.to_numpy()).astype(int),
                                          "conf": np.maximum(proba, 1 - proba), "coverage": cov,
                                          "words": [len(t.split()) for t in Xte]}))
    pd.DataFrame(rows).to_csv("results/loso.csv", index=False)
    d = pd.concat(docs, ignore_index=True)
    d.to_csv("results/loso_docs.csv", index=False)
    print("\nDoes a signal predict mistakes? AUROC of (low value => error), per held-out source")
    for held, g in d.groupby("held_out"):
        if g.correct.nunique() < 2:
            continue
        print(f"{held:11s} coverage {roc_auc_score(g.correct, g.coverage):.3f}  "
              f"confidence {roc_auc_score(g.correct, g.conf):.3f}  words {roc_auc_score(g.correct, g.words):.3f}")


if __name__ == "__main__":
    main()
