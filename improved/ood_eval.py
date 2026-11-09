"""
Can we tell, from the text alone, that an article comes from a kind of news the model
was never trained on?  Signal: cosine similarity between the article's TF-IDF vector and
the centroid of each training source (take the maximum).  For every source we leave out,
measure how well the signal separates (a) held-out articles of the training sources from
(b) articles of the left-out source.  AUROC 0.5 = useless, 1.0 = perfect.

    python ood_eval.py --isot ../Prototype/data/raw --ext external
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import fnd
import multisource

CAP = 8000


def centroid(M):
    c = np.asarray(M.mean(axis=0)).ravel()
    n = np.linalg.norm(c)
    return c / n if n else c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isot", required=True)
    ap.add_argument("--ext", required=True)
    a = ap.parse_args()
    src = multisource.load_sources(a.isot, a.ext)
    names = list(src)
    rows = []
    for held in names:
        train_names = [n for n in names if n != held]
        tr_docs, in_docs = {}, []
        for n in train_names:
            d = src[n].sample(min(CAP, len(src[n])), random_state=1)
            a_, b_ = train_test_split(d, test_size=0.25, random_state=1)
            tr_docs[n] = a_
            in_docs.append(b_.sample(min(600, len(b_)), random_state=1))
        in_df = pd.concat(in_docs)
        ood_df = src[held].sample(min(2000, len(src[held])), random_state=1)

        texts = {n: fnd.build_field(d, "title+text", 0, True) for n, d in tr_docs.items()}
        vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        vec.fit(pd.concat(texts.values()))
        cents = np.vstack([centroid(vec.transform(t)) for t in texts.values()])

        def score(df):
            X = normalize(vec.transform(fnd.build_field(df, "title+text", 0, True)))
            return np.asarray((X @ cents.T).max(axis=1)).ravel()

        s_in, s_out = score(in_df), score(ood_df)
        y = np.r_[np.zeros(len(s_in)), np.ones(len(s_out))]
        auc = roc_auc_score(y, -np.r_[s_in, s_out])          # low similarity => unseen
        thr = np.percentile(s_in, 5)                         # flag the lowest 5% of known news
        rows.append({"held_out": held, "auroc_unseen_vs_seen": round(auc, 3),
                     "flagged_unseen_at_5pct_false_alarm": round(float((s_out < thr).mean()), 3),
                     "threshold": round(float(thr), 4)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/ood.csv", index=False)


if __name__ == "__main__":
    main()
