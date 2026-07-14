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
    mod_file = tmp_path / "mymod.py"
    mod_file.write_text("x = 123\ny = 'hello'")
    x = sudregex._import_python_object(str(mod_file), "x")
    y = sudregex._import_python_object(str(mod_file), "y")
    assert x == 123
    assert y == "hello"


def test_extract_df_basic(tmp_path):
    df = pd.DataFrame({"note_id": [1, 2], "note_text": ["apple or orange", "banana only"]})
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
    assert out.loc[out.note_id.astype(str) == "1", "apple_chk"].iloc[0] >= 1
    assert out.loc[out.note_id.astype(str) == "2", "apple_chk"].iloc[0] == 0


def _mk_checklist(item_name="foo_chk", pat=re.compile("foo"), **flags):
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
    df = pd.DataFrame(
        {
            "note_id": ["1", "2"],
            "note_text": ["not foo here", "foo not here"],
        }
    )
    checklist = _mk_checklist(negation=True)

    out_left = sudregex.extract_df(
        df, checklist, terms=["__dummy__"], negation_scope="left", include_note_text=True, remove_linebreaks=False
    )
    out_right = sudregex.extract_df(
        df, checklist, terms=["__dummy__"], negation_scope="right", include_note_text=True, remove_linebreaks=False
    )

    assert out_left.loc[out_left.note_id == "1", "foo_chk_NEG"].iloc[0] == 0
    assert out_right.loc[out_right.note_id == "2", "foo_chk_NEG"].iloc[0] == 0


def test_extract_df_discharge_toggle():
    df = pd.DataFrame(
        {
            "note_id": ["1", "2"],
            "note_text": ["discharge instructions: foo only.", "regular note foo present"],
        }
    )
    checklist = _mk_checklist()

    out_exclude = sudregex.extract_df(
        df,
        checklist,
        terms=["__dummy__"],
        exclude_discharge_mentions=True,
        include_note_text=True,
        remove_linebreaks=False,
    )
    out_include = sudregex.extract_df(
        df,
        checklist,
        terms=["__dummy__"],
        exclude_discharge_mentions=False,
        include_note_text=True,
        remove_linebreaks=False,
    )

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
        sudregex.extract_df(df, checklist)


def test_extract_df_id_dtype_is_string():
    df = pd.DataFrame({"note_id": [101], "note_text": ["foo foo"]})
    checklist = _mk_checklist()
    out = sudregex.extract_df(df, checklist, terms=["__dummy__"], include_note_text=False, remove_linebreaks=False)
    assert out["note_id"].dtype.name == "string"


def test_extract_df_custom_id_name_and_person_roundtrip():
    df = pd.DataFrame(
        {
            "doc_oid": ["A", "B"],
            "patient_sid": ["P1", "P2"],
            "note_text": ["foo bar", "no hits"],
        }
    )
    checklist = {
        "foo_chk": {
            "pat": re.compile(r"\bfoo\b"),
            "col_name": "foo_chk",
            "negation": False,
            "substance": False,
            "preview": False,
        }
    }
    out = sudregex.extract_df(
        df,
        checklist,
        terms=["__dummy__"],
        id_column="doc_oid",
        person_column="patient_sid",
        include_note_text=False,
        remove_linebreaks=False,
    )
    cols = list(out.columns)
    assert cols[0] == "patient_sid" and cols[1] == "doc_oid"


def test_extract_df_error_when_person_without_id():
    df = pd.DataFrame({"patient_sid": ["P1"], "note_text": ["foo"]})
    checklist = {
        "c": {"pat": re.compile("foo"), "col_name": "c", "negation": False, "substance": False, "preview": False}
    }
    with pytest.raises(ValueError):
        sudregex.extract_df(
            df, checklist, terms=["__dummy__"], id_column=None, person_column="patient_sid", remove_linebreaks=False
        )


def test_extract_df_reattach_person_and_previews_return_df(tmp_path):
    df = pd.DataFrame({"note_id": ["1"], "person_id": ["P1"], "note_text": ["foo is here."]})
    checklist = {
        "foo_chk": {
            "pat": re.compile(r"\bfoo\b"),
            "col_name": "foo_chk",
            "negation": False,
            "substance": False,
            "preview": True,
        }
    }
    res, prev = sudregex.extract_df(
        df,
        checklist,
        terms=["__dummy__"],
        person_column="person_id",
        include_note_text=True,
        preview_count=1,
        return_previews_df=True,
        remove_linebreaks=False,
    )
    assert "person_id" in res.columns
    assert not prev.empty
    assert set(["item_key", "span_start", "span_end", "snippet"]).issubset(set(prev.columns))


def test_termslist_loading_from_file(tmp_path):
    terms_file = tmp_path / "my_terms.py"
    terms_file.write_text("opioid_terms = ['morphine','oxycodone']")
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["patient on morphine drip"]})
    checklist = {
        "morph_chk": {
            "pat": re.compile(r"\bmorphine\b"),
            "col_name": "morph_chk",
            "negation": False,
            "substance": False,
            "preview": False,
        }
    }
    out = sudregex.extract_df(
        df,
        checklist,
        termslist=str(terms_file),
        terms_active="opioid_terms",
        remove_linebreaks=False,
    )
    assert out["morph_chk"].iloc[0] > 0


def test_termslist_loading_from_dict():
    tdict = {"opioid_terms": ["morphine", "oxycodone"]}
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["oxycodone 5mg po"]})
    checklist = {
        "oxy_chk": {
            "pat": re.compile(r"\boxycodone\b"),
            "col_name": "oxy_chk",
            "negation": False,
            "substance": False,
            "preview": False,
        }
    }
    out = sudregex.extract_df(
        df,
        checklist,
        termslist=tdict,
        terms_active="opioid_terms",
        remove_linebreaks=False,
    )
    assert out["oxy_chk"].iloc[0] > 0


def test_extract_df_with_checklist_path(tmp_path):
    chk_file = tmp_path / "chk.py"
    chk_file.write_text(
        "import re\n"
        "pattern_library = {\n"
        "  'apple_chk': {\n"
        "    'pat': re.compile(r'\\bapple\\b'),\n"
        "    'col_name': 'apple_chk',\n"
        "    'negation': False,\n"
        "    'substance': False,\n"
        "    'preview': False,\n"
        "  }\n"
        "}\n"
    )
    df = pd.DataFrame({"note_id": ["1", "2"], "note_text": ["apple pie", "banana only"]})
    out = sudregex.extract_df(
        df,
        pattern_library=str(chk_file),
        terms=["__dummy__"],
        remove_linebreaks=False,
    )
    assert out.loc[out.note_id == "1", "apple_chk"].iloc[0] > 0
    assert out.loc[out.note_id == "2", "apple_chk"].iloc[0] == 0


def test_extract_no_header_csv_roundtrip(tmp_path):
    p = tmp_path / "notes_noheader.txt"
    rows = [
        "P1\t!^!\tN1\t!^!\tfoo bar",
        "P2\t!^!\tN2\t!^!\tbaz only",
    ]
    p.write_text("\n".join(rows))
    chk_file = tmp_path / "chk2.py"
    chk_file.write_text(
        "import re\n"
        "pattern_library = {'foo_chk': {'pat': re.compile(r'\\bfoo\\b'), 'col_name': 'foo_chk', 'negation': False, 'substance': False, 'preview': False}}\n"
    )
    out_csv = tmp_path / "out.csv"

    ok = sudregex.extract(
        in_file=str(p),
        out_file=str(out_csv),
        pattern_library=str(chk_file),
        separator=r"\t!\^!\t",
        terms=["__dummy__"],
        has_header=False,
        no_header_columns=["patient_id", "note_id", "note_text"],
        person_column="patient_id",
        id_column="note_id",
        include_note_text=False,
        remove_linebreaks=False,
        chunk_size=10,
    )
    assert ok is True
    out_df = pd.read_csv(out_csv)
    assert list(out_df.columns[:2]) == ["patient_id", "note_id"]


def test_identifier_column_order_in_output():
    df = pd.DataFrame({"note_id": ["1"], "person_col": ["PX"], "enc_id": ["E1"], "note_text": ["foo"]})
    checklist = _mk_checklist()
    out = sudregex.extract_df(
        df,
        checklist,
        terms=["__dummy__"],
        person_column="person_col",
        remove_linebreaks=False,
    )
    cols = list(out.columns)
    assert cols[0] == "person_col" and cols[1] == "note_id"


# ============================================================
# run_sudregex() dispatcher tests
#
# These cover the environment="local"|"databricks" wrapper added in 0.1.8.
# None of these require pyspark to be installed: the "local" path never
# touches Spark, and the "databricks" error-path tests raise before any
# pyspark import happens (run_databricks() checks spark is None as its
# first statement, before the try/except pyspark import). Tests requiring
# an actual SparkSession live in unittests/test_spark.py instead.
# ============================================================


def test_run_sudregex_default_environment_is_local():
    """run_sudregex() with no environment= specified should behave exactly like extract_df()."""
    df = pd.DataFrame({"note_id": ["1", "2"], "note_text": ["apple pie", "banana only"]})
    checklist = _mk_checklist(item_name="apple_chk", pat=re.compile(r"\bapple\b"))

    via_extract_df = sudregex.extract_df(
        df, checklist, terms=["__dummy__"], remove_linebreaks=False, include_note_text=False
    )
    via_run_sudregex = sudregex.run_sudregex(
        df, checklist, terms=["__dummy__"], remove_linebreaks=False, include_note_text=False
    )

    pd.testing.assert_frame_equal(
        via_extract_df.reset_index(drop=True),
        via_run_sudregex.reset_index(drop=True),
    )


def test_run_sudregex_explicit_local_matches_extract_df():
    """run_sudregex(..., environment='local') should be identical to calling extract_df() directly."""
    df = pd.DataFrame(
        {
            "note_id": ["1", "2", "3"],
            "note_text": ["patient denies foo use", "foo confirmed", "no mention here"],
        }
    )
    checklist = _mk_checklist(negation=True)

    via_extract_df = sudregex.extract_df(
        df,
        checklist,
        terms=["__dummy__"],
        negation_scope="left",
        include_note_text=True,
        remove_linebreaks=False,
    )
    via_run_sudregex = sudregex.run_sudregex(
        df,
        checklist,
        environment="local",
        terms=["__dummy__"],
        negation_scope="left",
        include_note_text=True,
        remove_linebreaks=False,
    )

    pd.testing.assert_frame_equal(
        via_extract_df.reset_index(drop=True),
        via_run_sudregex.reset_index(drop=True),
    )


def test_run_sudregex_local_ignores_spark_kwarg():
    """
    Passing spark= on the local path should be silently ignored (popped from kwargs)
    rather than raising, so callers can flip environments by changing one argument
    without conditionally omitting spark=.
    """
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["foo here"]})
    checklist = _mk_checklist()

    out = sudregex.run_sudregex(
        df,
        checklist,
        environment="local",
        spark="not_a_real_sparksession",
        terms=["__dummy__"],
        remove_linebreaks=False,
    )
    assert out.loc[out.note_id == "1", "foo_chk"].iloc[0] >= 1


def test_run_sudregex_unsupported_environment_raises():
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["foo"]})
    checklist = _mk_checklist()
    with pytest.raises(ValueError, match="Unsupported environment"):
        sudregex.run_sudregex(df, checklist, environment="banana", terms=["__dummy__"])


def test_run_sudregex_databricks_without_spark_raises():
    """
    This should raise before ever touching pyspark: run_sudregex('databricks') lazily
    imports sudregex.spark (which itself has no pyspark import at module load time)
    and run_databricks() checks for spark=None as its very first statement. So this
    test must pass even in an environment with no pyspark installed at all.
    """
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["foo"]})
    checklist = _mk_checklist()
    with pytest.raises(ValueError, match="requires an active SparkSession"):
        sudregex.run_sudregex(df, checklist, environment="databricks", spark=None, terms=["__dummy__"])


def test_run_sudregex_environment_case_insensitive():
    """environment='LOCAL' / 'Local' should behave the same as 'local'."""
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["foo"]})
    checklist = _mk_checklist()

    out_lower = sudregex.run_sudregex(df, checklist, environment="local", terms=["__dummy__"], remove_linebreaks=False)
    out_upper = sudregex.run_sudregex(df, checklist, environment="LOCAL", terms=["__dummy__"], remove_linebreaks=False)

    pd.testing.assert_frame_equal(out_lower.reset_index(drop=True), out_upper.reset_index(drop=True))


def test_run_sudregex_in_dunder_all():
    assert "run_sudregex" in sudregex.__all__


def test_run_sudregex_is_public_attribute():
    assert callable(sudregex.run_sudregex)
