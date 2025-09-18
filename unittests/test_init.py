import importlib.util
import os
import re
import tempfile

import pandas as pd
import pytest

import SUDRegex


def test_version_and_all():
    assert isinstance(SUDRegex.__version__, str)
    for name in ["extract_df", "remove_line_break", "check_for_substance"]:
        assert name in SUDRegex.__all__


def test_import_python_object(tmp_path):
    # write a small module
    mod_file = tmp_path / "mymod.py"
    mod_file.write_text("x = 123\ny = 'hello'")
    # import x
    x = SUDRegex._import_python_object(str(mod_file), "x")
    y = SUDRegex._import_python_object(str(mod_file), "y")
    # import whole module
    mod = SUDRegex._import_python_object(str(mod_file), None)
    assert x == 123
    assert y == "hello"
    assert hasattr(mod, "x") and hasattr(mod, "y")


def test_extract_df_basic(tmp_path):
    # prepare a small dataframe and a trivial checklist
    df = pd.DataFrame({"note_id": [1, 2], "note_text": ["apple or orange", "banana only"]})
    # checklist: look for 'apple' only
    checklist = {
        "apple_chk": {
            "pat": re.compile("apple"),
            "col_name": "apple_chk",
            "negation": False,
            "substance": False,
            "preview": False,
        }
    }
    out = SUDRegex.extract_df(
        df,
        checklist,
        terms=["irrelevant"],
        remove_linebreaks=False,
        keys=None,
        parallel=False,
        debug=False,
    )
    # apple appears only in note_id 1
    assert out.loc[out.note_id.astype(str) == "1", "apple_chk"].iloc[0] >= 1
    assert out.loc[out.note_id.astype(str) == "2", "apple_chk"].iloc[0] == 0
