"""
Cross-dataset evaluation with TF-IDF models: does a model trained on one source
work on another?  Also trains the final DEPLOYED model on ISOT + WELFake combined.

    python cross_dataset.py --isot ../Prototype/data/raw --ext external/welfake_clean.csv
Writes results/cross_dataset.csv and model.joblib (combined, linear SVM -> LogReg).
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import fnd


def model(kind):
    clf = LogisticRegression(max_iter=1000, C=3.0) if kind == "logreg" else LinearSVC()
    return make_pipeline(TfidfVectorizer(max_features=100000, ngram_range=(1, 2),
                                         sublinear_tf=True, min_df=2), clf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isot", required=True)
    ap.add_argument("--ext", required=True)
    a = ap.parse_args()

    isot = fnd.load_raw(a.isot)[["title", "text", "label"]]
    ext = pd.read_csv(a.ext)
    ext["title"] = ext["title"].fillna("").astype(str)
    ext["text"] = ext["text"].fillna("").astype(str)
    sets = {}
    for name, d in [("ISOT", isot), ("WELFake", ext)]:
        tr, te = train_test_split(d, test_size=0.2, stratify=d.label, random_state=42)
        sets[name] = (tr.reset_index(drop=True), te.reset_index(drop=True))
    sets["Combined"] = (pd.concat([sets["ISOT"][0], sets["WELFake"][0]], ignore_index=True), None)

    feat = lambda d: fnd.build_field(d, "title+text", 0, True)
    rows = []
    for kind in ("logreg", "linsvc"):
        for train_name in ("ISOT", "WELFake", "Combined"):
            tr = sets[train_name][0]
            m = model(kind).fit(feat(tr), tr.label)
            row = {"model": kind, "trained_on": train_name}
            for test_name in ("ISOT", "WELFake"):
                te = sets[test_name][1]
                p = m.predict(feat(te))
                row[f"test_{test_name}_acc"] = round(accuracy_score(te.label, p), 4)
                row[f"test_{test_name}_f1"] = round(f1_score(te.label, p), 4)
            rows.append(row)
            print(row)
            if kind == "logreg" and train_name == "Combined":
                # Deployed model: retrain on ALL data from both sources.
                allx = pd.concat([sets["ISOT"][0], sets["ISOT"][1], sets["WELFake"][0],
                                  sets["WELFake"][1]], ignore_index=True)
                final = model("logreg").fit(feat(allx), allx.label)
                joblib.dump({"model": final, "field": "title+text", "max_words": 0,
                             "id": "logreg_tfidf_ISOT+WELFake"}, "model.joblib")
    pd.DataFrame(rows).to_csv("results/cross_dataset.csv", index=False)


if __name__ == "__main__":
    main()
