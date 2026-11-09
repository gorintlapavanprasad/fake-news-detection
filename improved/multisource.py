"""
Load every source as a DataFrame with columns title, text, label (1 = real, 0 = fake).

Sources
  ISOT       Reuters vs fake-news websites (US politics, 2016-2017)      Kaggle
  WELFake    mixed outlets, overlap with ISOT removed                    HF davanstrien/WELFake
  GossipCop  celebrity / entertainment news, human-written (HF/HR files) HF Jinyan1/GossipCop
  PolitiFact fact-checked political stories, human-written (HF/HR)       HF Jinyan1/PolitiFact
  COVID      health claims and tweets about COVID-19                     HF nanyy1025/covid_fake_news

Machine-written GossipCop/PolitiFact variants (MF/MR) are left out on purpose:
they test a different problem (detecting generated text).
"""

import glob
import os

import pandas as pd

import fnd


def _frame(df, label=None):
    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str) if "title" in df else ""
    df["text"] = df["text"].fillna("").astype(str)
    if label is not None:
        df["label"] = label
    df = df[df["text"].str.strip().str.len() > 0]
    return df[["title", "text", "label"]].drop_duplicates(["title", "text"]).reset_index(drop=True)


def _hf_pair(root, name):
    fake = glob.glob(os.path.join(root, name, "data", "HF-*.parquet"))
    real = glob.glob(os.path.join(root, name, "data", "HR-*.parquet"))
    parts = [_frame(pd.read_parquet(p), 0) for p in fake] + [_frame(pd.read_parquet(p), 1) for p in real]
    return pd.concat(parts, ignore_index=True)


def load_sources(isot_dir, ext_dir):
    """Returns an ordered dict name -> DataFrame."""
    out = {}
    isot = fnd.load_raw(isot_dir)[["title", "text", "label"]]
    out["ISOT"] = isot
    out["WELFake"] = _frame(pd.read_csv(os.path.join(ext_dir, "welfake_clean.csv")))
    out["GossipCop"] = _hf_pair(ext_dir, "GossipCop")
    out["PolitiFact"] = _hf_pair(ext_dir, "PolitiFact")
    covid = []
    for f in ("train.csv", "val.csv", "test.csv"):
        d = pd.read_csv(os.path.join(ext_dir, "covid", f))
        d = d.rename(columns={"tweet": "text"})
        d["label"] = (d["label"].str.lower() == "real").astype(int)
        covid.append(d[["text", "label"]])
    out["COVID"] = _frame(pd.concat(covid, ignore_index=True).assign(title=""))
    return out
