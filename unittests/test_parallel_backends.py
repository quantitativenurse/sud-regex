import re

import pandas as pd
import pytest

import sudregex


def _mk_checklist(item_name="foo_chk", pat=re.compile(r"\bfoo\b"), **flags):
    return {
        item_name: {
            "pat": pat,
            "col_name": item_name,
            "negation": flags.get("negation", False),
            "substance": flags.get("substance", False),
            "preview": flags.get("preview", False),
        }
    }


def test_extract_df_invalid_parallel_backend_raises():
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["foo"]})
    checklist = _mk_checklist()

    with pytest.raises(ValueError, match="Unsupported parallel_backend"):
        sudregex.extract_df(
            df,
            checklist,
            terms=["__dummy__"],
            parallel=True,
            parallel_backend="not_a_backend",
            remove_linebreaks=False,
        )


def test_extract_df_invalid_n_workers_raises_zero():
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["foo"]})
    checklist = _mk_checklist()

    with pytest.raises(ValueError, match="n_workers must be a positive integer"):
        sudregex.extract_df(
            df,
            checklist,
            terms=["__dummy__"],
            parallel=True,
            parallel_backend="loky",
            n_workers=0,
            remove_linebreaks=False,
        )


def test_extract_df_invalid_n_workers_raises_negative():
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["foo"]})
    checklist = _mk_checklist()

    with pytest.raises(ValueError, match="n_workers must be a positive integer"):
        sudregex.extract_df(
            df,
            checklist,
            terms=["__dummy__"],
            parallel=True,
            parallel_backend="loky",
            n_workers=-2,
            remove_linebreaks=False,
        )


def test_extract_df_grid_column_aliases_to_person_column():
    df = pd.DataFrame(
        {
            "note_id": ["1", "2"],
            "grid": ["G1", "G2"],
            "note_text": ["foo here", "bar only"],
        }
    )
    checklist = _mk_checklist()

    out = sudregex.extract_df(
        df,
        checklist,
        terms=["__dummy__"],
        grid_column="grid",
        remove_linebreaks=False,
    )

    assert "grid" in out.columns
    assert out.columns[0] == "grid"
    assert out.columns[1] == "note_id"


def test_extract_df_serial_vs_loky_exact_match():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3", "P4"],
            "note_id": ["1", "2", "3", "4"],
            "note_text": [
                "patient reports foo use",
                "not foo currently",
                "foo again later",
                "regular note without match",
            ],
        }
    )
    checklist = _mk_checklist(negation=True)

    out_serial = sudregex.extract_df(
        df,
        checklist,
        terms=["__dummy__"],
        person_column="patient_id",
        id_column="note_id",
        parallel=False,
        remove_linebreaks=False,
    )

    out_loky = sudregex.extract_df(
        df,
        checklist,
        terms=["__dummy__"],
        person_column="patient_id",
        id_column="note_id",
        parallel=True,
        parallel_backend="loky",
        n_workers=2,
        remove_linebreaks=False,
    )

    assert out_serial.shape == out_loky.shape
    assert list(out_serial.columns) == list(out_loky.columns)
    assert out_serial.equals(out_loky)


def test_extract_df_serial_vs_pandarallel_exact_match():
    pytest.importorskip("pandarallel")

    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3", "P4"],
            "note_id": ["1", "2", "3", "4"],
            "note_text": [
                "patient reports foo use",
                "not foo currently",
                "foo again later",
                "regular note without match",
            ],
        }
    )
    checklist = _mk_checklist(negation=True)

    out_serial = sudregex.extract_df(
        df,
        checklist,
        terms=["__dummy__"],
        person_column="patient_id",
        id_column="note_id",
        parallel=False,
        remove_linebreaks=False,
    )

    out_parallel = sudregex.extract_df(
        df,
        checklist,
        terms=["__dummy__"],
        person_column="patient_id",
        id_column="note_id",
        parallel=True,
        parallel_backend="pandarallel",
        n_workers=2,
        remove_linebreaks=False,
    )

    assert out_serial.shape == out_parallel.shape
    assert list(out_serial.columns) == list(out_parallel.columns)
    assert out_serial.equals(out_parallel)


def test_extract_file_matches_extract_df(tmp_path):
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "note_id": ["N1", "N2"],
            "note_text": ["foo here", "no foo now"],
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

    chk_file = tmp_path / "chk.py"
    chk_file.write_text(
        "import re\n"
        "pattern_library = {\n"
        "    'foo_chk': {\n"
        "        'pat': re.compile(r'\\bfoo\\b'),\n"
        "        'col_name': 'foo_chk',\n"
        "        'negation': False,\n"
        "        'substance': False,\n"
        "        'preview': False,\n"
        "    }\n"
        "}\n"
    )

    in_file = tmp_path / "notes.csv"
    out_file = tmp_path / "out.csv"
    df.to_csv(in_file, index=False)

    out_df = sudregex.extract_df(
        df,
        checklist,
        terms=["__dummy__"],
        person_column="patient_id",
        id_column="note_id",
        remove_linebreaks=False,
    ).copy()

    ok = sudregex.extract(
        in_file=str(in_file),
        out_file=str(out_file),
        pattern_library=str(chk_file),
        separator=",",
        terms=["__dummy__"],
        person_column="patient_id",
        id_column="note_id",
        remove_linebreaks=False,
    )

    assert ok is True

    out_file_df = pd.read_csv(out_file)
    out_file_df["patient_id"] = out_file_df["patient_id"].astype("string")
    out_file_df["note_id"] = out_file_df["note_id"].astype("string")

    assert out_df.shape == out_file_df.shape
    assert list(out_df.columns) == list(out_file_df.columns)
    assert out_df.equals(out_file_df)