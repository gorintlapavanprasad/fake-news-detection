"""
Grid-search the settings of the browser model (must stay linear so it can run in JS).
Selection metric: the WORSE of the two held-out accuracies (ISOT, WELFake), so a model
cannot win by being great on one source and weak on the other.

    python tune_web_model.py --isot ../Prototype/data/raw --ext external/welfake_clean.csv
"""

import argparse
import itertools

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import fnd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isot", required=True)
    ap.add_argument("--ext", required=True)
    a = ap.parse_args()

    isot = fnd.load_raw(a.isot)[["title", "text", "label"]]
    ext = pd.read_csv(a.ext)
    ext["title"], ext["text"] = ext["title"].fillna("").astype(str), ext["text"].fillna("").astype(str)
    feat = lambda d: fnd.build_field(d, "title+text", 0, True)
    sp = {n: train_test_split(d, test_size=0.2, stratify=d.label, random_state=42)
          for n, d in [("ISOT", isot), ("WELFake", ext)]}
    tr = pd.concat([sp["ISOT"][0], sp["WELFake"][0]])
    Xtr, ytr = feat(tr), tr.label
    Xte = {n: feat(te) for n, (_, te) in sp.items()}

    rows = []
    for mf, C, ng in itertools.product([30000, 60000], [3, 10, 30], [(1, 2), (1, 3)]):
        m = make_pipeline(TfidfVectorizer(max_features=mf, ngram_range=ng, sublinear_tf=True, min_df=2),
                          LogisticRegression(max_iter=3000, C=C)).fit(Xtr, ytr)
        r = {"max_features": mf, "C": C, "ngram": ng}
        for n, (_, te) in sp.items():
            r[n] = round(accuracy_score(te.label, m.predict(Xte[n])), 4)
        r["worst"] = min(r["ISOT"], r["WELFake"])
        rows.append(r)
        print(r, flush=True)
    res = pd.DataFrame(rows).sort_values("worst", ascending=False)
    res.to_csv("results/tuning.csv", index=False)
    print("\nBEST:\n", res.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
