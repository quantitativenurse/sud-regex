import importlib.util
import os
import re
import tempfile

import pandas as pd
import pytest

import sudregex


def test_version_and_all():
    assert isinstance(sudregex.__version__, str)
    for name in ["extract_df", "remove_line_break", "check_for_substance"]:
        assert name in sudregex.__all__


def test_import_python_object(tmp_path):
    # write a small module
    mod_file = tmp_path / "mymod.py"
    mod_file.write_text("x = 123\ny = 'hello'")
    # import x
    x = sudregex._import_python_object(str(mod_file), "x")
    y = sudregex._import_python_object(str(mod_file), "y")
    # import whole module
    mod = sudregex._import_python_object(str(mod_file), None)
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
    out = sudregex.extract_df(
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


def _mk_checklist(item_name="foo_chk", pat=re.compile("foo"), **flags):
    # helper: build a minimal checklist item with toggles
    return {
        item_name: {
            "pat": pat,
            "col_name": item_name,
            "negation": flags.get("negation", False),
            "substance": flags.get("substance", False),
            "preview": flags.get("preview", False),
        }
    }
def test_extract_df_negation_scope_left_vs_right():
    df = pd.DataFrame({
        "note_id": ["1", "2"],
        "note_text": ["not foo here", "foo not here"],
    })
    checklist = _mk_checklist(negation=True)

    out_left = sudregex.extract_df(
        df, checklist, terms=["__dummy__"],
        negation_scope="left", include_note_text=True, remove_linebreaks=False
    )
    out_right = sudregex.extract_df(
        df, checklist, terms=["__dummy__"],
        negation_scope="right", include_note_text=True, remove_linebreaks=False
    )

    # Assert on the NEGATED (final) mask, not the base count
    assert out_left.loc[out_left.note_id == "1", "foo_chk_NEG"].iloc[0] == 0   # "not" to the left → drop
    assert out_right.loc[out_right.note_id == "2", "foo_chk_NEG"].iloc[0] == 0 # "not" to the right → drop



def test_extract_df_discharge_toggle():
    df = pd.DataFrame({
        "note_id": ["1", "2"],
        "note_text": ["discharge instructions: foo only.", "regular note foo present"],
    })
    checklist = _mk_checklist()

    out_exclude = sudregex.extract_df(
        df, checklist, terms=["__dummy__"],
        exclude_discharge_mentions=True, include_note_text=True, remove_linebreaks=False
    )
    out_include = sudregex.extract_df(
        df, checklist, terms=["__dummy__"],
        exclude_discharge_mentions=False, include_note_text=True, remove_linebreaks=False
    )

    # When excluding discharge mentions, row 1 should be 0; when including, it should be >0
    assert out_exclude.loc[out_exclude.note_id == "1", "foo_chk"].iloc[0] == 0
    assert out_include.loc[out_include.note_id == "1", "foo_chk"].iloc[0] > 0


def test_extract_df_include_note_text_flag_controls_column():
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["foo"]})
    checklist = _mk_checklist()

    out_no_text = sudregex.extract_df(
        df, checklist, terms=["__dummy__"], include_note_text=False, remove_linebreaks=False
    )
    out_with_text = sudregex.extract_df(
        df, checklist, terms=["__dummy__"], include_note_text=True, remove_linebreaks=False
    )

    assert "note_text" not in out_no_text.columns
    assert "note_text" in out_with_text.columns


def test_extract_df_requires_terms_or_termslist():
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["foo"]})
    checklist = _mk_checklist()
    with pytest.raises(ValueError):
        sudregex.extract_df(df, checklist)  # neither terms nor termslist/terms_active provided


def test_extract_df_id_dtype_is_string():
    df = pd.DataFrame({"note_id": [101], "note_text": ["foo foo"]})
    checklist = _mk_checklist()
    out = sudregex.extract_df(
        df, checklist, terms=["__dummy__"], include_note_text=False, remove_linebreaks=False
    )
    assert out["note_id"].dtype.name == "string"
