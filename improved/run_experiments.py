"""
Runs many model/feature/data variations and writes results/experiments.csv.

Usage (from this folder):
    python run_experiments.py --data ../Prototype/data/raw [--new-data other.csv]

Each experiment is evaluated on:
  - random : stratified random 80/20 split (what the original project reported)
  - shift  : train on some news sections, test on held-out sections (unseen topics)
             (a much more honest proxy for "new data")
  - new    : an optional external CSV with `text` (and optional `title`) + `label`
             columns (1 = real, 0 = fake), if --new-data is given
"""

import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import fnd


def make_model(name, max_features, ngram):
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram,
                            sublinear_tf=True, min_df=2)
    clf = {
        "logreg": LogisticRegression(max_iter=1000, C=3.0),
        "linsvc": LinearSVC(C=1.0),
        "cnb": ComplementNB(),
        "sgd": SGDClassifier(loss="modified_huber", random_state=42),
        "dtree": DecisionTreeClassifier(max_depth=25, min_samples_leaf=2, random_state=42),
        "rforest": RandomForestClassifier(n_estimators=100, max_depth=40, n_jobs=-1,
                                          random_state=42),
    }[name]
    return make_pipeline(tfidf, clf)


# (id, model, field, max_words, strip_leaks, max_features, ngram)
EXPERIMENTS = [
    ("01_logreg_baseline_stripped",     "logreg",  "title+text", 0,   True,  50000, (1, 1)),
    ("02_logreg_RAW_with_leaks",        "logreg",  "title+text", 0,   False, 50000, (1, 1)),
    ("03_logreg_bigrams",               "logreg",  "title+text", 0,   True,  100000, (1, 2)),
    ("04_logreg_small_vocab_5k",        "logreg",  "title+text", 0,   True,  5000,  (1, 1)),
    ("05_linsvc_stripped",              "linsvc",  "title+text", 0,   True,  50000, (1, 1)),
    ("06_linsvc_RAW_with_leaks",        "linsvc",  "title+text", 0,   False, 50000, (1, 1)),
    ("07_cnb_stripped",                 "cnb",     "title+text", 0,   True,  50000, (1, 1)),
    ("08_sgd_stripped",                 "sgd",     "title+text", 0,   True,  50000, (1, 1)),
    ("09_dtree_stripped",               "dtree",   "title+text", 0,   True,  20000, (1, 1)),
    ("10_rforest_stripped",             "rforest", "title+text", 0,   True,  20000, (1, 1)),
    ("11_logreg_title_only",            "logreg",  "title",      0,   True,  50000, (1, 2)),
    ("12_logreg_text_only",             "logreg",  "text",       0,   True,  50000, (1, 1)),
    ("13_logreg_first_50_words",        "logreg",  "title+text", 50,  True,  50000, (1, 1)),
    ("14_logreg_first_200_words",       "logreg",  "title+text", 200, True,  50000, (1, 1)),
    ("15_linsvc_bigrams_title_only",    "linsvc",  "title",      0,   True,  100000, (1, 2)),
    ("16_cnb_RAW_with_leaks",           "cnb",     "title+text", 0,   False, 50000, (1, 1)),
]


def score(model, X, y):
    p = model.predict(X)
    return accuracy_score(y, p), f1_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing Fake.csv and True.csv")
    ap.add_argument("--new-data", default=None, help="optional external CSV to test on")
    ap.add_argument("--out", default="results")
    ap.add_argument("--save-best", default="model.joblib")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = fnd.load_raw(args.data)
    print(f"Articles after de-duplication: {len(df)} "
          f"(real={int((df.label == 1).sum())}, fake={int((df.label == 0).sum())})")

    idx = np.arange(len(df))
    tr_r, te_r = train_test_split(idx, test_size=0.2, stratify=df.label, random_state=42)
    # Date-based splits do not work here: ~70% of dates are unparseable (almost all
    # in Fake.csv) and fake/real cover different date ranges. Instead we hold out
    # whole news SECTIONS (the `subject` column) to simulate unseen topics/sources.
    held_out = {"left-news", "US_News", "Middle-east", "worldnews"}
    is_test = df["subject"].isin(held_out).to_numpy()
    tr_f, te_f = np.where(~is_test)[0], np.where(is_test)[0]
    print(f"Unseen-section split: train {len(tr_f)}, test {len(te_f)} "
          f"(held-out sections: {sorted(held_out)})")

    new_df = None
    if args.new_data:
        new_df = pd.read_csv(args.new_data)
        if "title" not in new_df:
            new_df["title"] = ""
        new_df["title"] = new_df["title"].fillna("").astype(str)
        new_df["text"] = new_df["text"].fillna("").astype(str)

    rows, best = [], (-1, None)
    for eid, mname, field, mw, strip, mf, ng in EXPERIMENTS:
        t0 = time.time()
        X = fnd.build_field(df, field, mw, strip)
        y = df.label.to_numpy()
        row = {"experiment": eid, "model": mname, "field": field, "max_words": mw,
               "strip_leaks": strip}

        m = make_model(mname, mf, ng).fit(X.iloc[tr_r], y[tr_r])
        row["random_acc"], row["random_f1"] = score(m, X.iloc[te_r], y[te_r])

        mf_ = make_model(mname, mf, ng).fit(X.iloc[tr_f], y[tr_f])
        row["shift_acc"], row["shift_f1"] = score(mf_, X.iloc[te_f], y[te_f])

        if new_df is not None:
            Xn = fnd.build_field(new_df, field, mw, strip)
            row["new_acc"], row["new_f1"] = score(m, Xn, new_df.label.to_numpy())

        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

        # Keep the best *stripped* model by unseen-section accuracy as the deployable one.
        if strip and row["shift_acc"] > best[0]:
            best = (row["shift_acc"], (eid, mname, field, mw, mf, ng))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "experiments.csv"), index=False)

    eid, mname, field, mw, mf, ng = best[1]
    print(f"\nBest honest model: {eid} (unseen-section acc {best[0]:.4f}). Retraining on all data...")
    final = make_model(mname, mf, ng).fit(fnd.build_field(df, field, mw, True), df.label)
    joblib.dump({"model": final, "field": field, "max_words": mw, "id": eid},
                args.save_best)
    print("Saved", args.save_best)


if __name__ == "__main__":
    main()
