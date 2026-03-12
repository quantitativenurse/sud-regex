import re

import pandas as pd
import pytest

from sudregex.validation import (
    _helper_negation,
    import_python_object,
    parse_text,
    validate_checklist,
    validate_rows,
)


def _mk_checklist():
    return {
        "foo_chk": {
            "pat": re.compile(r"\bfoo\b"),
            "col_name": "foo_chk",
            "negation": False,
            "substance": False,
            "preview": False,
        }
    }


def _mk_neg_checklist():
    return {
        "foo_chk": {
            "pat": re.compile(r"\bfoo\b"),
            "col_name": "foo_chk",
            "negation": True,
            "substance": False,
            "preview": False,
        }
    }


def _mk_substance_checklist():
    return {
        "foo_chk": {
            "pat": re.compile(r"\bfoo\b"),
            "col_name": "foo_chk",
            "negation": False,
            "substance": True,
            "preview": False,
        }
    }


def _mk_common_fp_checklist():
    return {
        "foo_chk": {
            "pat": re.compile(r"\bfoo\b"),
            "col_name": "foo_chk",
            "negation": False,
            "substance": False,
            "preview": False,
            "common_fp": ["family history"],
        }
    }


def test_import_python_object_success(tmp_path):
    mod_file = tmp_path / "mymod.py"
    mod_file.write_text("checklist = {'a': 1}\nvalue = 42\n")

    checklist = import_python_object(str(mod_file), "checklist")
    value = import_python_object(str(mod_file), "value")

    assert checklist == {"a": 1}
    assert value == 42


def test_import_python_object_missing_var_raises(tmp_path):
    mod_file = tmp_path / "mymod.py"
    mod_file.write_text("x = 1\n")

    with pytest.raises(AttributeError, match="Expected `checklist`"):
        import_python_object(str(mod_file), "checklist")


def test_parse_text_pipe_format(tmp_path):
    p = tmp_path / "examples.txt"
    p.write_text("foo_chk | 1 | patient has foo\n")

    df = parse_text(str(p))

    assert list(df.columns) == ["item_code", "item_key", "expected", "note_text"]
    assert df.loc[0, "item_key"] == "foo_chk"
    assert df.loc[0, "expected"] == 1
    assert df.loc[0, "note_text"] == "patient has foo"


def test_parse_text_bang_format(tmp_path):
    p = tmp_path / "examples.txt"
    p.write_text("foo_chk !^! 0 !^! patient denies foo\n")

    df = parse_text(str(p))

    assert df.loc[0, "item_key"] == "foo_chk"
    assert df.loc[0, "expected"] == 0


def test_parse_text_invalid_expected_raises(tmp_path):
    p = tmp_path / "examples.txt"
    p.write_text("foo_chk | 2 | patient has foo\n")

    with pytest.raises(ValueError, match="expected must be '0' or '1'"):
        parse_text(str(p))


def test_parse_text_bad_format_raises(tmp_path):
    p = tmp_path / "examples.txt"
    p.write_text("this line has no valid delimiter\n")

    with pytest.raises(ValueError, match="expected 'item_key \\| expected \\| note_text'"):
        parse_text(str(p))


def test_helper_negation_left_only_true_when_cue_on_left():
    text = "patient does not foo today"
    span = re.search(r"\bfoo\b", text).span()
    assert _helper_negation(text, span) is True


def test_helper_negation_left_only_false_when_cue_on_right():
    text = "foo not present"
    span = re.search(r"\bfoo\b", text).span()
    assert _helper_negation(text, span) is False


def test_helper_negation_false_when_no_cue():
    text = "patient has foo today"
    span = re.search(r"\bfoo\b", text).span()
    assert _helper_negation(text, span) is False


def test_validate_rows_basic_accepts_match():
    checklist = _mk_checklist()
    df = pd.DataFrame([{"item_key": "foo_chk", "item_code": "foo_chk", "expected": 1, "note_text": "patient has foo"}])

    detailed, by_item, previews = validate_rows(checklist, df)

    assert detailed["actual_match"].iloc[0] == 1
    assert detailed["mismatch"].iloc[0] == 0
    assert by_item.loc[by_item["item_key"] == "foo_chk", "tp"].iloc[0] == 1
    assert previews is None


def test_validate_rows_negation_rejects_hit():
    checklist = _mk_neg_checklist()
    df = pd.DataFrame(
        [{"item_key": "foo_chk", "item_code": "foo_chk", "expected": 0, "note_text": "patient not foo today"}]
    )

    detailed, by_item, _ = validate_rows(checklist, df)

    assert detailed["actual_match"].iloc[0] == 0
    assert "negated" in detailed["failure_reason"].iloc[0]
    assert by_item.loc[by_item["item_key"] == "foo_chk", "fn"].iloc[0] == 0


def test_validate_rows_common_fp_rejects_hit():
    checklist = _mk_common_fp_checklist()
    df = pd.DataFrame(
        [{"item_key": "foo_chk", "item_code": "foo_chk", "expected": 0, "note_text": "family history foo"}]
    )

    detailed, _, _ = validate_rows(checklist, df)

    assert detailed["actual_match"].iloc[0] == 0
    assert "common_fp" in detailed["failure_reason"].iloc[0]


def test_validate_rows_substance_required_rejects_without_vocab_match():
    checklist = _mk_substance_checklist()
    df = pd.DataFrame(
        [{"item_key": "foo_chk", "item_code": "foo_chk", "expected": 0, "note_text": "patient has foo only"}]
    )

    detailed, _, _ = validate_rows(
        checklist,
        df,
        substance_terms=["opioid"],
        substance_window_chars=50,
    )

    assert detailed["actual_match"].iloc[0] == 0
    assert "needs_substance" in detailed["failure_reason"].iloc[0]


def test_validate_rows_substance_required_accepts_with_vocab_match():
    checklist = _mk_substance_checklist()
    df = pd.DataFrame(
        [{"item_key": "foo_chk", "item_code": "foo_chk", "expected": 1, "note_text": "opioid foo use noted"}]
    )

    detailed, _, _ = validate_rows(
        checklist,
        df,
        substance_terms=["opioid"],
        substance_window_chars=50,
    )

    assert detailed["actual_match"].iloc[0] == 1


def test_validate_rows_unknown_item_key():
    checklist = _mk_checklist()
    df = pd.DataFrame(
        [{"item_key": "missing_chk", "item_code": "missing_chk", "expected": 0, "note_text": "patient has foo"}]
    )

    detailed, _, _ = validate_rows(checklist, df)

    assert detailed["actual_match"].iloc[0] == 0
    assert detailed["failure_reason"].iloc[0] == "unknown_item_key"


def test_validate_rows_collect_previews():
    checklist = {
        "foo_chk": {
            "pat": re.compile(r"\bfoo\b"),
            "col_name": "foo_chk",
            "negation": False,
            "substance": False,
            "preview": True,
        }
    }
    df = pd.DataFrame(
        [
            {
                "note_id": "N1",
                "item_key": "foo_chk",
                "item_code": "foo_chk",
                "expected": 1,
                "note_text": "patient has foo today",
            }
        ]
    )

    detailed, by_item, previews = validate_rows(checklist, df, collect_previews=True)

    assert detailed["actual_match"].iloc[0] == 1
    assert previews is not None
    assert not previews.empty
    assert set(["item_key", "note_id", "match_text", "snippet"]).issubset(previews.columns)
    assert previews["match_text"].iloc[0] == "foo"


def test_validate_checklist_dataframe_input():
    checklist = _mk_checklist()
    df = pd.DataFrame([{"item_key": "foo_chk", "item_code": "foo_chk", "expected": 1, "note_text": "patient has foo"}])

    detailed, by_item = validate_checklist(checklist, df)

    assert detailed["actual_match"].iloc[0] == 1
    assert "TOTAL" in set(by_item["item_key"])


def test_validate_checklist_examples_file_input(tmp_path):
    chk_file = tmp_path / "chk.py"
    chk_file.write_text(
        "import re\n"
        "checklist = {\n"
        "    'foo_chk': {\n"
        "        'pat': re.compile(r'\\bfoo\\b'),\n"
        "        'col_name': 'foo_chk',\n"
        "        'negation': False,\n"
        "        'substance': False,\n"
        "        'preview': False,\n"
        "    }\n"
        "}\n"
    )

    ex_file = tmp_path / "examples.txt"
    ex_file.write_text("foo_chk | 1 | patient has foo\n")

    detailed, by_item = validate_checklist(str(chk_file), str(ex_file))

    assert detailed["actual_match"].iloc[0] == 1
    assert by_item.loc[by_item["item_key"] == "foo_chk", "tp"].iloc[0] == 1


def test_validate_checklist_return_previews_and_write_csvs(tmp_path):
    checklist = {
        "foo_chk": {
            "pat": re.compile(r"\bfoo\b"),
            "col_name": "foo_chk",
            "negation": False,
            "substance": False,
            "preview": True,
        }
    }
    df = pd.DataFrame(
        [
            {
                "note_id": "N1",
                "item_key": "foo_chk",
                "item_code": "foo_chk",
                "expected": 1,
                "note_text": "patient has foo today",
            }
        ]
    )

    out_csv = tmp_path / "detailed.csv"
    by_item_csv = tmp_path / "by_item.csv"
    previews_csv = tmp_path / "previews.csv"

    detailed, by_item, previews = validate_checklist(
        checklist,
        df,
        out_csv=str(out_csv),
        by_item_csv=str(by_item_csv),
        return_previews=True,
        previews_csv=str(previews_csv),
    )

    assert out_csv.exists()
    assert by_item_csv.exists()
    assert previews_csv.exists()
    assert not previews.empty
    assert detailed["actual_match"].iloc[0] == 1
