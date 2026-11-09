"""
Prepare the external WELFake dataset as an unseen test/training set.

    python prepare_external.py --isot ../Prototype/data/raw

Steps:
 1. drop empty rows and exact duplicates
 2. drop anything whose title OR text also appears in the ISOT data (no overlap)
 3. detect the label direction empirically: WELFake documents 0=fake/1=real but
    users report it is flipped. We score a sample with the ISOT-trained model and
    flip if agreement is below 50%. The result is printed so it can be audited.
Output: external/welfake_clean.csv with columns title,text,label (1=real, 0=fake)
"""

import argparse
import glob
import os

import joblib
import pandas as pd

import fnd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isot", required=True)
    ap.add_argument("--src", default=None)
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--out", default="external/welfake_clean.csv")
    a = ap.parse_args()

    src = a.src or glob.glob("external/data/*.parquet")[0]
    df = pd.read_parquet(src)
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    n0 = len(df)
    df = df[df["text"].str.strip().str.len() > 0].drop_duplicates(["title", "text"])
    n1 = len(df)

    isot = fnd.load_raw(a.isot)
    seen_titles = set(isot["title"].str.strip().str.lower())
    seen_text = set(isot["text"].str.strip().str.lower().str[:300])
    overlap = df["title"].str.strip().str.lower().isin(seen_titles) | \
        df["text"].str.strip().str.lower().str[:300].isin(seen_text)
    df = df[~overlap].reset_index(drop=True)
    print(f"rows: {n0} -> {n1} after dedup -> {len(df)} after removing ISOT overlap")

    bundle = joblib.load(a.model)
    s = df.sample(2000, random_state=0)
    X = fnd.build_field(s, bundle["field"], bundle["max_words"], True)
    agree = float((bundle["model"].predict(X) == s["label"].to_numpy()).mean())
    print(f"agreement of ISOT model with raw WELFake labels: {agree:.3f}")
    if agree < 0.5:
        df["label"] = 1 - df["label"]
        print("labels look FLIPPED in WELFake -> flipped so that 1 = real, 0 = fake")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df[["title", "text", "label"]].to_csv(a.out, index=False)
    print("saved", a.out, df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
