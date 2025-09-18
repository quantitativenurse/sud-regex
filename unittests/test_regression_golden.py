import csv
import importlib.util as iu
import io
import re
from pathlib import Path

import pandas as pd
import pytest

import SUDRegex

_fc = Path(__file__).with_name("fail_checklist.py")
spec = iu.spec_from_file_location("fail_checklist", _fc)
_mod = iu.module_from_spec(spec)
spec.loader.exec_module(_mod)
checklist_override = getattr(_mod, "checklist")
# make this true to use the override checklist and test failures
USE_OVERRIDE = False
CHECKLIST = checklist_override if USE_OVERRIDE else SUDRegex.checklist_abc

REQUIRED_HEADERS = {"grid", "note_id", "note_text"}
HEADER_SYNONYMS = {
    "grid": {"grid"},
    "note_id": {"note_id", "id", "note id", "noteid"},
    "note_text": {"note_text", "text", "note text", "notetext"},
}

# ---------- IO helpers ----------


def _best_header_rename(cols):
    norm = {c: c.strip().lower() for c in cols}
    rename = {}
    for want, alts in HEADER_SYNONYMS.items():
        for c, n in norm.items():
            if n in alts:
                rename[c] = want
                break
    return rename


def _autodetect_sep(sample_text: str) -> str:
    if "!^!" in sample_text:
        return r"\s*!\^!\s*"
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", "\t", "|", ";"])
        return dialect.delimiter
    except Exception:
        pass
    candidates = [",", "\t", "|", ";"]
    lines = [ln for ln in sample_text.splitlines() if ln.strip()][:10]
    best, best_score = None, -1e9
    for cand in candidates:
        counts = [len(ln.split(cand)) for ln in lines] or [1]
        score = -pd.Series(counts).var()
        if score > best_score:
            best, best_score = cand, score
    return best or ","


def _resolve_sep(sep_opt: str, text: str):
    if sep_opt == "auto":
        return _autodetect_sep(text), "python"  # python engine handles regex/tabs well
    if sep_opt.startswith("re:"):
        return sep_opt[3:], "python"
    return ("\t" if sep_opt == "\\t" else sep_opt), "python"


def _read_notes(path: Path, sep_opt: str):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        raise AssertionError(f"Notes file is empty: {path}")
    sep, engine = _resolve_sep(sep_opt, text)
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine=engine, dtype=str)
    except Exception as e:
        head = "\n".join(text.splitlines()[:6])
        raise AssertionError(
            f"Failed to parse notes with sep='{sep_opt}' (resolved='{sep}').\n"
            f"Head:\n{head}\n"
            f"Try --notes-sep auto, ',', '\\t', '|', ';', or regex like --notes-sep 're:\\s*!\\^!\\s*'.\n"
            f"Original error: {e}"
        )
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=_best_header_rename(df.columns))
    missing = REQUIRED_HEADERS - set(df.columns)
    if missing:
        raise AssertionError(f"Notes missing columns: {missing}. Found: {list(df.columns)}")
    for c in REQUIRED_HEADERS:
        df[c] = df[c].astype("string").str.strip()
    return df, sep


def _normalize_numeric_str(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    res = s.astype(str).copy()
    mask = out.notna()
    res.loc[mask] = out.loc[mask].map(lambda x: str(int(x)) if float(x).is_integer() else str(float(x)))
    res = res.str.replace(r"(\.\d*?)0+$", r"\1", regex=True).str.replace(r"\.$", "", regex=True)
    return res


def _postprocess_colorder(df: pd.DataFrame) -> pd.DataFrame:
    if "grid" in df.columns:
        cols = [c for c in df.columns if c != "grid"] + ["grid"]
        return df[cols]
    return df


# ---------- Extraction runners ----------


def _run_extract_df(df_notes: pd.DataFrame, terms: list, terms_active=None) -> pd.DataFrame:
    out = SUDRegex.extract_df(
        df=df_notes,
        checklist=CHECKLIST,
        terms=terms if terms else None,
        termslist=SUDRegex.default_termslist if (terms_active and not terms) else None,
        terms_active=terms_active,
        remove_linebreaks=True,
        id_column="note_id",
        grid_column="grid",
        include_note_text=False,
        debug=False,
        parallel=False,
    )
    return _postprocess_colorder(out)


def _run_extract_file(notes_path: Path, sep_str: str, tmpdir: Path, terms: list, terms_active=None) -> pd.DataFrame:
    out_csv = tmpdir / "extract_out.csv"
    SUDRegex.extract(
        in_file=str(notes_path),
        out_file=str(out_csv),
        checklist=CHECKLIST,
        separator=sep_str,
        terms=terms if terms else None,
        termslist=SUDRegex.default_termslist if (terms_active and not terms) else None,
        terms_active=terms_active,
        parallel=False,
        include_note_text=False,
        remove_linebreaks=True,
        note_column="note_text",
        id_column="note_id",
        grid_column="grid",
        keep_columns=None,
        debug=False,
        has_header=True,
    )
    # Collect output(s). If chunking ever created parts, concat them.
    parts = sorted(tmpdir.glob("extract_out_part_*.csv"))
    if parts:
        dfs = [pd.read_csv(p, dtype=str, low_memory=False) for p in parts]
        res = pd.concat(dfs, ignore_index=True)
    else:
        res = pd.read_csv(out_csv, dtype=str, low_memory=False)
    for c in res.columns:
        res[c] = res[c].astype("string")
    return _postprocess_colorder(res)


# ---------- Comparer ----------


def _compare_frames(
    actual: pd.DataFrame, expected: pd.DataFrame, key="note_id", exclude=("note_text",), also_binary=True
):
    for df in (actual, expected):
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            df[c] = df[c].astype("string")

    only_a = sorted(set(actual[key]) - set(expected[key]))
    only_b = sorted(set(expected[key]) - set(actual[key]))
    if only_a or only_b:
        raise AssertionError(
            f"note_id mismatch. only_in_actual={len(only_a)}, only_in_expected={len(only_b)}.\n"
            f"Examples actual-only: {only_a[:5]}; expected-only: {only_b[:5]}"
        )

    common_cols = sorted(set(actual.columns).intersection(expected.columns))
    common_cols = [c for c in common_cols if c not in exclude and c != key]

    left = actual[[key] + common_cols].sort_values(key).reset_index(drop=True).copy()
    right = expected[[key] + common_cols].sort_values(key).reset_index(drop=True).copy()

    for c in common_cols:
        left[c] = _normalize_numeric_str(left[c])
        right[c] = _normalize_numeric_str(right[c])

    mism = []
    for c in common_cols:
        neq = (left[c] != right[c]) & ~(left[c].isna() & right[c].isna())
        if neq.any():
            sub = pd.DataFrame(
                {
                    key: left.loc[neq, key],
                    "column": c,
                    "actual": left.loc[neq, c].astype(str).values,
                    "expected": right.loc[neq, c].astype(str).values,
                }
            )
            mism.append(sub)
    exact_mism = (
        pd.concat(mism, ignore_index=True) if mism else pd.DataFrame(columns=[key, "column", "actual", "expected"])
    )

    binary_mism = pd.DataFrame(columns=[key, "column", "actual_bin", "expected_bin"])
    if also_binary:
        bins = []
        for c in common_cols:
            a = pd.to_numeric(left[c], errors="coerce").fillna(0)
            b = pd.to_numeric(right[c], errors="coerce").fillna(0)
            neq = (a > 0).astype(int) != (b > 0).astype(int)
            if neq.any():
                sub = pd.DataFrame(
                    {
                        key: left.loc[neq, key],
                        "column": c,
                        "actual_bin": (a > 0).astype(int).loc[neq].values,
                        "expected_bin": (b > 0).astype(int).loc[neq].values,
                    }
                )
                bins.append(sub)
        if bins:
            binary_mism = pd.concat(bins, ignore_index=True)

    ok = exact_mism.empty
    return ok, exact_mism, binary_mism, len(common_cols)


def _write_artifacts(
    dirpath: Path,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    exact_mism: pd.DataFrame,
    binary_mism: pd.DataFrame,
    label: str,
):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / f"{label}_actual.csv").write_text(actual.to_csv(index=False), encoding="utf-8")
    (dirpath / f"{label}_expected.csv").write_text(expected.to_csv(index=False), encoding="utf-8")
    exact_mism.to_csv(dirpath / f"{label}_exact_mismatches.csv", index=False)
    binary_mism.to_csv(dirpath / f"{label}_binary_mismatches.csv", index=False)


def _resolve_terms(cfg):
    if cfg.get("terms"):
        return [t for t in cfg["terms"].split(";") if t.strip()], None
    # Default to the standard terms group so extract_df/extract can call set_terms(...)
    return [], (cfg.get("terms_active") or "opioid_terms")


# ---------- Tests ----------


def test_golden_extract_df(test_cfg, tmp_path):
    notes_df, sep_str = _read_notes(test_cfg["notes_file"], test_cfg["notes_sep"])
    terms, terms_active = _resolve_terms(test_cfg)

    actual_df = _run_extract_df(notes_df, terms, terms_active)
    expected = pd.read_csv(test_cfg["expected_csv"], dtype=str, low_memory=False)
    expected = _postprocess_colorder(expected)

    if test_cfg["regen_golden"] and test_cfg["regen_from"] == "df":
        test_cfg["expected_csv"].parent.mkdir(parents=True, exist_ok=True)
        actual_df.to_csv(test_cfg["expected_csv"], index=False)
        pytest.skip(f"--regen-golden used (from df); wrote {test_cfg['expected_csv']}")

    ok, exact_mism, binary_mism, ncols = _compare_frames(actual_df, expected, also_binary=test_cfg["check_binary"])
    _write_artifacts(Path("unittests/artifacts"), actual_df, expected, exact_mism, binary_mism, "df_vs_golden")
    assert ok, (
        f"\nextract_df vs GOLDEN mismatch (cols compared={ncols}).\n"
        f"See unittests/artifacts/df_vs_golden_exact_mismatches.csv"
    )


def test_golden_extract_file(test_cfg, tmp_path):
    notes_df, sep_str = _read_notes(test_cfg["notes_file"], test_cfg["notes_sep"])
    terms, terms_active = _resolve_terms(test_cfg)

    actual_ex = _run_extract_file(test_cfg["notes_file"], sep_str, tmp_path, terms, terms_active)
    expected = pd.read_csv(test_cfg["expected_csv"], dtype=str, low_memory=False)
    expected = _postprocess_colorder(expected)

    if test_cfg["regen_golden"] and test_cfg["regen_from"] == "extract":
        test_cfg["expected_csv"].parent.mkdir(parents=True, exist_ok=True)
        actual_ex.to_csv(test_cfg["expected_csv"], index=False)
        pytest.skip(f"--regen-golden used (from extract); wrote {test_cfg['expected_csv']}")

    ok, exact_mism, binary_mism, ncols = _compare_frames(actual_ex, expected, also_binary=test_cfg["check_binary"])
    _write_artifacts(Path("unittests/artifacts"), actual_ex, expected, exact_mism, binary_mism, "extract_vs_golden")
    assert ok, (
        f"\nextract (file) vs GOLDEN mismatch (cols compared={ncols}).\n"
        f"See unittests/artifacts/extract_vs_golden_exact_mismatches.csv"
    )


def test_extract_parity_with_extract_df(test_cfg, tmp_path):
    """Direct parity check: extract() output equals extract_df() output on the same notes."""
    notes_df, sep_str = _read_notes(test_cfg["notes_file"], test_cfg["notes_sep"])
    terms, terms_active = _resolve_terms(test_cfg)

    df_out = _run_extract_df(notes_df, terms, terms_active)
    ex_out = _run_extract_file(test_cfg["notes_file"], sep_str, tmp_path, terms, terms_active)

    ok, exact_mism, binary_mism, ncols = _compare_frames(df_out, ex_out, also_binary=test_cfg["check_binary"])
    _write_artifacts(Path("unittests/artifacts"), df_out, ex_out, exact_mism, binary_mism, "df_vs_extract")
    assert ok, (
        f"\nextract_df vs extract mismatch (cols compared={ncols}).\n"
        f"See unittests/artifacts/df_vs_extract_exact_mismatches.csv"
    )
