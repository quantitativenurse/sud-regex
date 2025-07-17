import os
import re
import tempfile

import pandas as pd
import pytest

from SUDRegex import helper


def test_remove_line_break_simple():
    txt = "Hello$+$\nWorld"
    out = helper.remove_line_break(txt)
    assert out == "Hello World"


def test_remove_line_break_multiple_markers():
    txt = "A+++B and C$$$D"
    out = helper.remove_line_break(txt, break_markers=["+++", r"\$\$\$"])
    assert out == "A B and C D"


def test_remove_tobacco_mentions():
    txt = "Patient denies Tobacco use: None today."
    masked = helper.remove_tobacco_mentions(txt)
    assert "Tobacco: [Redacted]" in masked


def test_set_terms_and_regex_search_file_basic(tmp_path):
    # prepare small df
    df = pd.DataFrame(
        {"note_id": [1, 2], "note_text": ["foo bar baz", "no match here"]}
    )
    metadata = df[["note_id"]].copy()
    # compile a simple pat
    pat = re.compile("foo")
    helper.set_terms(["bar"])  # for substance detection later
    df_searched = helper.regex_search_file(
        pat, "foo", df.copy(), metadata, preview=False
    )
    # note 1 has 1 match, note 2 has 0
    assert df_searched.loc[df_searched.note_id == 1, "foo"].iloc[0] == 1
    assert df_searched.loc[df_searched.note_id == 2, "foo"].iloc[0] == 0


def test_check_for_substance_detects_term():
    # build df_searched with an initial 'foo' count >0
    df = pd.DataFrame({"note_id": [1], "note_text": ["abc foo bar xyz"], "foo": [1]})
    helper.set_terms(["bar"])
    out = helper.check_for_substance(re.compile("foo"), "foo", "foo_SUB", df.copy())
    # should find 'bar' near 'foo'
    assert out.loc[out.note_id == 1, "foo_SUB"].iloc[0] == 1


def test_check_negation_flags_negation():
    df = pd.DataFrame(
        {"note_id": [1], "note_text": ["patient has not foo now"], "foo": [1]}
    )
    out = helper.check_negation(re.compile("foo"), "foo", "foo_NEG", df.copy())
    # 'not' before foo → marked as negated → 0
    assert out.loc[out.note_id == 1, "foo_NEG"].iloc[0] == 0


def test_check_common_false_positives_filters_family_history():
    df = pd.DataFrame(
        {"note_id": [1], "note_text": ["family history of asthma"], "hist": [1]}
    )
    out = helper.check_common_false_positives(
        re.compile("history"), df.copy(), "hist", ["family"]
    )
    # 'family history' should be filtered, so hist → 0
    assert out.loc[out.note_id == 1, "hist"].iloc[0] == 0


def test_discharge_instructions_removes_discharge_context():
    df = pd.DataFrame(
        {
            "note_id": [1, 2],
            "note_text": ["discharge instructions foo", "no discharge here foo"],
            "foo": [1, 1],
        }
    )
    out = helper.discharge_instructions(re.compile("foo"), df.copy(), "foo")
    # first row in discharge context → 0, second remains 1
    assert out.loc[out.note_id == 1, "foo"].iloc[0] == 0
    assert out.loc[out.note_id == 2, "foo"].iloc[0] == 1
