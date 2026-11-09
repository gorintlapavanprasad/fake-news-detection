"""
Core library for the improved fake-news pipeline.

Fixes over the original scripts:
- Works locally (no Colab paths); data directory is a parameter.
- Removes duplicate articles BEFORE splitting (the dataset has many).
- Optionally strips "source giveaways" (e.g. "WASHINGTON (Reuters) -") that let a
  model score ~99% without learning anything about truthfulness.
- Label convention is explicit: 1 = REAL, 0 = FAKE, everywhere.
"""

import os
import re

import pandas as pd

LABEL_NAMES = {0: "FAKE", 1: "REAL"}

# Words that identify the publisher/format rather than the truthfulness of a story.
LEAK_WORDS = {
    "reuters", "getty", "images", "image", "via", "pic", "twitter", "com",
    "featured", "video", "watch", "photo", "screenshot", "wire", "century",
    "21st", "breaking", "read", "more", "source",
}
_DATELINE = re.compile(r"^.{0,120}?\(reuters\)\s*[-\u2013\u2014]?\s*", re.IGNORECASE)
# Outlet name tags such as "... - The New York Times" or "... | Breitbart".
_OUTLET_TAG = re.compile(r"\s[-|]\s*(the )?(new york times|breitbart|washington post)\b.*$",
                         re.IGNORECASE)
_URL = re.compile(r"http\S+|www\.\S+")
_NON_ALPHA = re.compile(r"[^a-z\s]")


def load_raw(data_dir: str) -> pd.DataFrame:
    """Read Fake.csv + True.csv, label them, drop empties and exact duplicates."""
    fake = pd.read_csv(os.path.join(data_dir, "Fake.csv"))
    true = pd.read_csv(os.path.join(data_dir, "True.csv"))
    fake["label"], true["label"] = 0, 1
    df = pd.concat([fake, true], ignore_index=True)
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[(df["text"].str.strip().str.len() > 0)]
    df = df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)
    df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce")
    return df


def clean(text: str, strip_leaks: bool = True) -> str:
    """Lowercase, drop URLs/punctuation/digits; optionally remove giveaway tokens."""
    if not isinstance(text, str):
        return ""
    if strip_leaks:
        text = _DATELINE.sub("", text)
        text = _OUTLET_TAG.sub("", text)
    text = _URL.sub(" ", text.lower())
    text = _NON_ALPHA.sub(" ", text)
    words = text.split()
    if strip_leaks:
        words = [w for w in words if w not in LEAK_WORDS]
    return " ".join(words)


def build_field(df: pd.DataFrame, field: str = "title+text", max_words: int = 0,
                strip_leaks: bool = True) -> pd.Series:
    """Pick which part of the article the model sees, then clean it."""
    if field == "title":
        raw = df["title"]
    elif field == "text":
        raw = df["text"]
    else:
        raw = df["title"] + " " + df["text"]
    out = raw.map(lambda t: clean(t, strip_leaks))
    if max_words:
        out = out.map(lambda t: " ".join(t.split()[:max_words]))
    return out
