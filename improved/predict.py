"""
Live prediction with the saved model.

    python predict.py "Some headline or article text"        # one-off
    python predict.py --file articles.csv                    # batch (text[,title] columns)
    python predict.py                                        # interactive prompt
"""

import argparse
import sys

import joblib
import pandas as pd

import fnd


def load(path="model.joblib"):
    return joblib.load(path)


def predict_df(bundle, df):
    X = fnd.build_field(df, bundle["field"], bundle["max_words"], True)
    model = bundle["model"]
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        conf = model.predict_proba(X).max(axis=1)
    else:
        d = model.decision_function(X)
        conf = 1 / (1 + pow(2.718281828, -abs(d)))
    return [(fnd.LABEL_NAMES[int(p)], float(c)) for p, c in zip(pred, conf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--model", default="model.joblib")
    a = ap.parse_args()
    bundle = load(a.model)

    if a.file:
        df = pd.read_csv(a.file)
        df["title"] = df["title"].fillna("") if "title" in df else ""
        df["text"] = df["text"].fillna("")
        df["prediction"], df["confidence"] = zip(*predict_df(bundle, df))
        print(df[["prediction", "confidence"]].to_string())
        return

    def one(t):
        label, conf = predict_df(bundle, pd.DataFrame({"title": [""], "text": [t]}))[0]
        print(f"{label}  (confidence {conf:.2f})")

    if a.text:
        one(" ".join(a.text))
    else:
        print("Type an article/headline (blank line to quit):")
        for line in sys.stdin:
            if not line.strip():
                break
            one(line.strip())


if __name__ == "__main__":
    main()
