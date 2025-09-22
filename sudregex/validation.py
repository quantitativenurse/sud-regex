# SUDRegex/validation.py
from __future__ import annotations

import importlib.util
import os
import re
from typing import Any, Dict, List, Tuple, Union

import pandas as pd


# ---------- loaders ----------
def import_python_object(file_path: str, varname: str = "checklist"):
    """Import a variable (e.g., checklist) from a Python file."""
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, varname)


def _compile_pattern(raw_pat):
    """
    Compile the checklist pattern exactly as provided.
    No auto-fixes (we validate the checklist 'as-is').
    """
    if hasattr(raw_pat, "pattern"):  # already compiled
        return raw_pat
    return re.compile(str(raw_pat), flags=re.IGNORECASE | re.MULTILINE)


# ---------- example file parsing ----------
def parse_text(path: str) -> pd.DataFrame:
    """
    Supported line formats (no header):

      1) Pipe-delimited:   item_key | expected | note_text
      2) Bang-delimited:   item_key !^! expected !^! note_text

    Notes:
      - expected must be '0' or '1'
      - item_key must match keys in the checklist (e.g., '1a', '10', '11b')
    """
    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            if "!^!" in raw:
                parts = [p.strip() for p in raw.split("!^!", 2)]
            elif "|" in raw:
                parts = [p.strip() for p in raw.split("|", 2)]
            else:
                raise ValueError(
                    f"Line {ln}: expected 'item_key | expected | note_text' or "
                    f"'item_key !^! expected !^! note_text'. Got: {raw}"
                )

            if len(parts) != 3:
                raise ValueError(f"Line {ln}: expected 3 fields. Got: {raw}")

            item_key, expected_str, note_text = parts

            if expected_str not in {"0", "1"}:
                raise ValueError(f"Line {ln}: expected must be '0' or '1' (got '{expected_str}').")
            expected = int(expected_str)

            rows.append(
                dict(
                    item_code=item_key,  # keep 'item_code' for detailed output compatibility
                    item_key=item_key,  # used to look up the checklist pattern
                    expected=expected,
                    note_text=note_text,
                )
            )

    return pd.DataFrame(rows)


# ---------- core validation ----------
def validate_rows(checklist: Dict[str, Any], df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run pure-regex validation against df rows.

    Input df must have:
      - item_key (or item_code with the same value)
      - expected ∈ {0,1}
      - note_text

    Returns:
      detailed_rows:   item_code, expected, note_text, actual_match, mismatch
      by_item_summary: item_key, n, correct, accuracy + a 'TOTAL' row
    """
    # Pre-compile patterns per item_key exactly as-written in the checklist
    patterns = {key: _compile_pattern(obj["pat"]) for key, obj in checklist.items()}

    # Ensure item_key exists (if caller only provided item_code)
    if "item_key" not in df.columns and "item_code" in df.columns:
        df = df.copy()
        df["item_key"] = df["item_code"]

    def apply_match(row):
        key = row["item_key"]
        text = row["note_text"]
        pat = patterns.get(key)
        if pat is None:
            return 0  # unknown item -> no match
        return 1 if re.search(pat, text) else 0

    detailed = df.copy()
    detailed["actual_match"] = detailed.apply(apply_match, axis=1)
    detailed["mismatch"] = (detailed["actual_match"] != detailed["expected"].astype(int)).astype(int)

    # Keep item_key for the by-item summary, but do not include it in the final detailed CSV
    detailed_out = detailed[["item_code", "expected", "note_text", "actual_match", "mismatch"]].copy()

    # per-item summary
    work = detailed[["item_key", "expected", "actual_match"]].copy()
    work["expected"] = work["expected"].astype(int)
    work["correct"] = (work["expected"] == work["actual_match"]).astype(int)

    by_item = work.groupby("item_key").agg(n=("correct", "size"), correct=("correct", "sum")).reset_index()
    by_item["accuracy"] = by_item["correct"] / by_item["n"]

    # overall totals
    total_n = int(work["correct"].size)
    total_correct = int(work["correct"].sum())
    total_acc = total_correct / total_n if total_n else float("nan")
    total_row = pd.DataFrame([{"item_key": "TOTAL", "n": total_n, "correct": total_correct, "accuracy": total_acc}])
    by_item = pd.concat([by_item, total_row], ignore_index=True)

    return detailed_out, by_item


# ---------- public API ----------
def validate_checklist(
    checklist: Union[str, Dict[str, Any]],
    examples: Union[str, pd.DataFrame],
    out_csv: str | None = None,
    by_item_csv: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Programmatic API.

    Parameters
    ----------
    checklist : path to checklist.py OR a dict loaded by caller
    examples  : path to examples file OR a DataFrame with columns:
                - item_key | expected | note_text   (or item_code instead of item_key)
                expected must be 0/1
    out_csv   : optional path to write detailed CSV
    by_item_csv: optional path to write by-item summary CSV

    Returns
    -------
    (detailed_df, by_item_df)
    """
    if isinstance(checklist, str):
        checklist = import_python_object(checklist, "checklist")

    if isinstance(examples, str):
        df = parse_text(examples)
    else:
        # accept either item_key or item_code
        req_any = [{"item_key", "expected", "note_text"}, {"item_code", "expected", "note_text"}]
        if not any(r.issubset(set(examples.columns)) for r in req_any):
            raise ValueError(
                "examples DataFrame must contain either "
                "{item_key, expected, note_text} or {item_code, expected, note_text}"
            )
        df = examples.copy()
        if "item_key" not in df.columns:
            df["item_key"] = df["item_code"]
        if "item_code" not in df.columns:
            df["item_code"] = df["item_key"]

    detailed, by_item = validate_rows(checklist, df)

    if out_csv:
        detailed.to_csv(out_csv, index=False)
    if by_item_csv:
        by_item.to_csv(by_item_csv, index=False)

    return detailed, by_item
