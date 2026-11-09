import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "improved"))
import fnd  # noqa: E402


def test_clean_basic():
    assert fnd.clean("Hello, World! 123 http://x.com/a") == "hello world"


def test_clean_handles_non_string():
    assert fnd.clean(None) == ""
    assert fnd.clean(float("nan")) == ""


def test_dateline_and_outlet_are_stripped():
    out = fnd.clean("WASHINGTON (Reuters) - The Senate voted on Tuesday.")
    assert "reuters" not in out and out.startswith("the senate")
    assert "york" not in fnd.clean("Big win for team - The New York Times")


def test_leaks_kept_when_requested():
    assert "reuters" in fnd.clean("WASHINGTON (Reuters) - hi there", strip_leaks=False)


def test_build_field_modes_and_truncation():
    df = pd.DataFrame({"title": ["Alpha beta"], "text": ["gamma delta epsilon"]})
    assert fnd.build_field(df, "title").iloc[0] == "alpha beta"
    assert fnd.build_field(df, "text").iloc[0] == "gamma delta epsilon"
    assert fnd.build_field(df, "title+text", max_words=3).iloc[0] == "alpha beta gamma"


def test_load_raw_dedupes_and_labels(tmp_path):
    fake = pd.DataFrame({"title": ["a", "a", "b"], "text": ["x", "x", ""],
                         "subject": ["s"] * 3, "date": ["Dec 31, 2017"] * 3})
    true = pd.DataFrame({"title": ["c"], "text": ["y"], "subject": ["s"],
                         "date": ["December 31, 2017"]})
    fake.to_csv(tmp_path / "Fake.csv", index=False)
    true.to_csv(tmp_path / "True.csv", index=False)
    df = fnd.load_raw(str(tmp_path))
    assert len(df) == 2                      # duplicate and empty-text rows dropped
    assert dict(zip(df.title, df.label)) == {"a": 0, "c": 1}


def test_label_names():
    assert fnd.LABEL_NAMES == {0: "FAKE", 1: "REAL"}


@pytest.mark.parametrize("text", ["", "   ", "!!!", "12345"])
def test_degenerate_inputs_clean_to_empty(text):
    assert fnd.clean(text) == ""
