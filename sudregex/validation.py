# sudregex/validation.py
from __future__ import annotations

import importlib.util
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

# ---------------- Types & built-ins ----------------

# A negation function returns True if the hit (span) is negated in `text`
NegationFn = Callable[[str, Tuple[int, int]], bool]

# ---------------- Negation (MATCHES HELPER) ----------------
# Helper uses *left-only* window (span chars before the match) and a simple vocab:
#   window = text[max(0, start - span) : stop]
# Terms: ["no ", "not ", "denie", "denial", "doubt", "never", "negative", "without", "neg", "didn't"]

_HELPER_NEG_TERMS: List[str] = [
    "no ",
    "not ",
    "denie",  # matches denies/denied/denying
    "denial",
    "doubt",
    "never",
    "negative",
    "without",
    "neg",  # lab shorthand
    "didn't",
]
_HELPER_NEG_RE = re.compile("|".join(re.escape(t) for t in _HELPER_NEG_TERMS), re.IGNORECASE | re.MULTILINE)


def _helper_negation(text: str, span: Tuple[int, int], left_chars: int = 65) -> bool:
    """
    MATCHES helper.check_negation logic:
    - LEFT-ONLY window of `left_chars` before the match, up to the end of the match.
    - Returns True if any helper-negation term is found in that window.
    """
    s, e = span
    left = max(0, s - left_chars)
    window = text[left:e]
    return _HELPER_NEG_RE.search(window) is not None


# ---------------- Loaders ----------------


def import_python_object(file_path: str, varname: str = "checklist"):
    """Import a variable (e.g., checklist) from a Python file."""
    import os

    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, varname):
        raise AttributeError(f"Expected `{varname}` in {file_path}, but it wasn’t found.")
    return getattr(mod, varname)


def _compile_pattern(raw_pat):
    """
    Compile the checklist pattern exactly as provided.
    No auto-fixes (we validate the checklist 'as-is').
    """
    if hasattr(raw_pat, "pattern"):  # already compiled
        return raw_pat
    return re.compile(str(raw_pat), flags=re.IGNORECASE | re.MULTILINE)


# ---------------- Example file parsing ----------------


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


# ---------------- Helpers: common_fp & substance gate ----------------


def _compile_common_fp(fps: Optional[Sequence[str]]) -> Sequence[re.Pattern]:
    if not fps:
        return ()
    out: List[re.Pattern] = []
    for f in fps:
        # allow users to supply either plain substrings or regex-y snippets
        pat = re.compile(str(f), flags=re.IGNORECASE | re.MULTILINE)
        out.append(pat)
    return tuple(out)


def _is_common_fp_hit(text: str, span: Tuple[int, int], fp_pats: Sequence[re.Pattern], window_chars: int) -> bool:
    if not fp_pats:
        return False
    s, e = span
    left = max(0, s - window_chars)
    right = min(len(text), e + window_chars)
    ctx = text[left:right]
    return any(p.search(ctx) for p in fp_pats)


def _compile_terms(terms: Optional[Sequence[str]]) -> Sequence[re.Pattern]:
    if not terms:
        return ()
    # escape terms (treat as literals) and match case-insensitively
    return tuple(re.compile(re.escape(t), re.IGNORECASE | re.MULTILINE) for t in terms)


def _is_substance_hit(text: str, span: Tuple[int, int], term_pats: Sequence[re.Pattern], window_chars: int) -> bool:
    if not term_pats:
        return False
    s, e = span
    left = max(0, s - window_chars)
    right = min(len(text), e + window_chars)
    ctx = text[left:right]
    return any(p.search(ctx) for p in term_pats)


# ---------------- Preview helpers ----------------


def _first_substance_term(
    text: str, span: Tuple[int, int], term_pats: Sequence[re.Pattern], window_chars: int
) -> Optional[str]:
    if not term_pats:
        return None
    s, e = span
    left = max(0, s - window_chars)
    right = min(len(text), e + window_chars)
    ctx = text[left:right]
    for p in term_pats:
        m = p.search(ctx)
        if m:
            return m.group(0)
    return None


def _first_helper_negation_cue(window_text: str) -> Optional[str]:
    m = _HELPER_NEG_RE.search(window_text)
    return m.group(0) if m else None


def _make_snippet(text: str, s: int, e: int, left_chars: int, right_chars: int) -> tuple[str, int, int]:
    start = max(0, s - left_chars)
    end = min(len(text), e + right_chars)
    return text[start:end], start, end


# ---------------- Core validation ----------------


def validate_rows(
    checklist: Dict[str, Any],
    df: pd.DataFrame,
    negation_fn: Optional[NegationFn] = None,
    fp_window_chars: int = 120,  # context window for common_fp checks
    substance_terms: Optional[Sequence[str]] = None,  # optional global vocab
    substance_window_chars: int = 120,  # window around hit for substance vocab
    *,
    collect_previews: bool = False,  # if True, return previews dataframe
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Negation-aware, common_fp-aware, and (optionally) substance-gated validator.
    Optionally collects preview snippets for items with `preview: True` in checklist.

    Negation behavior: matches helper.check_negation (LEFT-ONLY, 65 chars by default).
    Returns (detailed_out, by_item, previews_df) where previews_df may be None.
    """
    # default to HELPER-STYLE negation if not supplied
    if negation_fn is None:
        negation_fn = _helper_negation

    term_pats = _compile_terms(substance_terms)

    # Pre-compile the main pattern and per-item settings
    patterns: Dict[str, re.Pattern] = {}
    item_cfg: Dict[str, Dict[str, Any]] = {}  # key -> dict with flags + compiled fp
    any_item_has_common_fp = False
    any_item_needs_substance = False
    for key, obj in checklist.items():
        pat = _compile_pattern(obj["pat"])
        fps = _compile_common_fp(obj.get("common_fp"))
        needs_sub = bool(obj.get("substance") or obj.get("opioid"))
        item_cfg[key] = {
            "negation": bool(obj.get("negation", False)),
            "common_fp": fps,
            "needs_substance": needs_sub,
            "preview": bool(obj.get("preview", False)),
            # Optional per-item overrides (safe defaults)
            "substance_window": int(obj.get("substance_window", substance_window_chars)),
            "common_fp_window": int(obj.get("common_fp_window", fp_window_chars)),
            "neg_left_span": int(obj.get("neg_left_span", 65)),  # match helper default 65
        }
        patterns[key] = pat
        any_item_has_common_fp = any_item_has_common_fp or bool(fps)
        any_item_needs_substance = any_item_needs_substance or needs_sub

    # Ensure item_key exists (if caller only provided item_code)
    if "item_key" not in df.columns and "item_code" in df.columns:
        df = df.copy()
        df["item_key"] = df["item_code"]

    # Diagnostics (stored raw; we may or may not export them)
    diag_cols = {
        "raw_hits": [],
        "substance_nearby": [],
        "common_fp_nearby": [],
        # Extra diagnostics (accepted vs any)
        "accepted_substance_nearby": [],
        "accepted_common_fp_nearby": [],
        "failure_reason": [],
        # Applicability (for n/a rendering)
        "substance_applicable": [],
        "common_fp_applicable": [],
    }

    # Previews (optional)
    previews: List[Dict[str, Any]] = []

    # Negation preview window (LEFT ONLY per helper); no right-side expansion
    def _neg_window_sizes(use_neg: bool, neg_left_span: int) -> tuple[int, int]:
        return (neg_left_span if use_neg else 0, 0)

    def apply_match(row):
        key = row["item_key"]
        text = row["note_text"]
        note_id = row["note_id"] if "note_id" in row else None
        pat = patterns.get(key)
        if pat is None:
            diag_cols["raw_hits"].append(0)
            diag_cols["substance_nearby"].append(False)
            diag_cols["common_fp_nearby"].append(False)
            diag_cols["accepted_substance_nearby"].append(False)
            diag_cols["accepted_common_fp_nearby"].append(False)
            diag_cols["failure_reason"].append("unknown_item_key")
            diag_cols["substance_applicable"].append(False)
            diag_cols["common_fp_applicable"].append(False)
            return 0

        cfg = item_cfg[key]
        use_neg = cfg["negation"]
        fp_pats = cfg["common_fp"]
        needs_sub = cfg["needs_substance"]
        want_preview = cfg["preview"]
        sw = cfg["substance_window"]
        fw = cfg["common_fp_window"]
        nl = cfg["neg_left_span"]

        # Applicability flags (row-level)
        # Substance is applicable ONLY if item needs it AND global vocab exists.
        substance_applicable = bool(needs_sub and term_pats)
        # Common FP is applicable if the item configured any patterns.
        common_fp_applicable = bool(fp_pats)

        # Ensure preview snippet is NOT less than any gating window
        neg_left_req, neg_right_req = _neg_window_sizes(use_neg, nl)
        left_req = max(sw if substance_applicable else 0, fw if common_fp_applicable else 0, neg_left_req)
        right_req = max(sw if substance_applicable else 0, fw if common_fp_applicable else 0, neg_right_req)

        matches = list(re.finditer(pat, text))
        raw_hits = len(matches)
        any_substance = False
        any_common_fp = False
        failure_reasons: List[str] = []

        for m in matches:
            span = m.span()

            # Only compute substance window if applicable for this item
            sub_hit = substance_applicable and _is_substance_hit(text, span, term_pats, sw)
            # Only compute common_fp window if applicable for this item
            fp_hit = common_fp_applicable and _is_common_fp_hit(text, span, fp_pats, fw)

            # HELPER-STYLE negation (left-only)
            neg_hit = use_neg and negation_fn(text, span)

            any_substance = any_substance or bool(sub_hit)
            any_common_fp = any_common_fp or bool(fp_hit)

            # --- PREVIEW capture (BEFORE gating drops a match) ---
            if collect_previews and want_preview:
                snippet, snip_s, snip_e = _make_snippet(text, span[0], span[1], left_req, right_req)
                neg_window = text[max(0, span[0] - nl) : span[1]] if use_neg else ""
                previews.append(
                    {
                        "item_key": key,
                        "note_id": note_id,
                        "match_span_start": span[0],
                        "match_span_end": span[1],
                        "match_text": text[span[0] : span[1]],
                        "snippet": snippet,  # focused context (not full note)
                        "snippet_start": snip_s,
                        "snippet_end": snip_e,
                        "substance_term": (
                            _first_substance_term(text, span, term_pats, sw) if substance_applicable else None
                        ),
                        "negation_cue": _first_helper_negation_cue(neg_window) if use_neg else None,
                        "common_fp_hit": bool(fp_hit) if common_fp_applicable else None,
                    }
                )

            # --- Gating ---
            if needs_sub and term_pats and not sub_hit:
                failure_reasons.append("needs_substance")
                continue
            if neg_hit:
                failure_reasons.append("negated")
                continue
            if fp_hit:
                failure_reasons.append("common_fp")
                continue

            # accepted
            diag_cols["raw_hits"].append(raw_hits)
            diag_cols["substance_nearby"].append(bool(any_substance))
            diag_cols["common_fp_nearby"].append(bool(any_common_fp))
            diag_cols["accepted_substance_nearby"].append(bool(sub_hit))
            diag_cols["accepted_common_fp_nearby"].append(bool(fp_hit))
            diag_cols["failure_reason"].append("")
            diag_cols["substance_applicable"].append(substance_applicable)
            diag_cols["common_fp_applicable"].append(common_fp_applicable)
            return 1

        # none accepted
        diag_cols["raw_hits"].append(raw_hits)
        diag_cols["substance_nearby"].append(bool(any_substance))
        diag_cols["common_fp_nearby"].append(bool(any_common_fp))
        diag_cols["accepted_substance_nearby"].append(False)
        diag_cols["accepted_common_fp_nearby"].append(False)
        diag_cols["failure_reason"].append(";".join(sorted(set(failure_reasons))) if matches else "no_raw_hit")
        diag_cols["substance_applicable"].append(substance_applicable)
        diag_cols["common_fp_applicable"].append(common_fp_applicable)
        return 0

    detailed = df.copy()
    detailed["actual_match"] = detailed.apply(apply_match, axis=1)
    detailed["mismatch"] = (detailed["actual_match"] != detailed["expected"].astype(int)).astype(int)

    # Attach diagnostics (raw)
    detailed["raw_hits"] = diag_cols["raw_hits"]
    detailed["substance_nearby"] = diag_cols["substance_nearby"]
    detailed["common_fp_nearby"] = diag_cols["common_fp_nearby"]
    detailed["accepted_substance_nearby"] = diag_cols["accepted_substance_nearby"]
    detailed["accepted_common_fp_nearby"] = diag_cols["accepted_common_fp_nearby"]
    detailed["failure_reason"] = diag_cols["failure_reason"]
    detailed["_substance_applicable"] = diag_cols["substance_applicable"]
    detailed["_common_fp_applicable"] = diag_cols["common_fp_applicable"]

    # Convert booleans → tri-state: "n/a" | 1 | 0 (only if columns are included)
    def _na1_0(val: bool, applicable: bool):
        if not applicable:
            return "n/a"
        return 1 if bool(val) else 0

    # Decide whether to include diagnostic columns at all:
    # - Substance columns only if: at least one item needs substance AND a global vocab was passed.
    # - Common FP columns only if: at least one item configured common_fp patterns.
    show_substance_cols = bool(any_item_needs_substance and term_pats)
    show_common_fp_cols = bool(any_item_has_common_fp)

    if show_substance_cols:
        detailed["substance_nearby"] = [
            _na1_0(v, a) for v, a in zip(detailed["substance_nearby"], detailed["_substance_applicable"])
        ]
        detailed["accepted_substance_nearby"] = [
            _na1_0(v, a) for v, a in zip(detailed["accepted_substance_nearby"], detailed["_substance_applicable"])
        ]
    if show_common_fp_cols:
        detailed["common_fp_nearby"] = [
            _na1_0(v, a) for v, a in zip(detailed["common_fp_nearby"], detailed["_common_fp_applicable"])
        ]
        detailed["accepted_common_fp_nearby"] = [
            _na1_0(v, a) for v, a in zip(detailed["accepted_common_fp_nearby"], detailed["_common_fp_applicable"])
        ]

    # Build output columns (keep note_id if present)
    base_cols = [
        "item_code",
        "item_key",
        "expected",
        "note_text",
        "actual_match",
        "mismatch",
        "raw_hits",
        "failure_reason",
    ]
    if show_substance_cols:
        base_cols[6:6] = ["substance_nearby", "accepted_substance_nearby"]  # insert before raw_hits
    if show_common_fp_cols:
        insert_at = 8 if show_substance_cols else 6
        base_cols[insert_at:insert_at] = ["common_fp_nearby", "accepted_common_fp_nearby"]
    if "note_id" in detailed.columns:
        base_cols.insert(2, "note_id")  # after item_key

    detailed_out = detailed[base_cols].copy()

    # per-item summary with precision/recall/F1
    work = detailed[["item_key", "expected", "actual_match"]].copy().astype({"expected": int, "actual_match": int})
    work["tp"] = ((work.expected == 1) & (work.actual_match == 1)).astype(int)
    work["fp"] = ((work.expected == 0) & (work.actual_match == 1)).astype(int)
    work["fn"] = ((work.expected == 1) & (work.actual_match == 0)).astype(int)

    by_item = (
        work.groupby("item_key")
        .agg(
            n=("expected", "size"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
        )
        .reset_index()
    )
    by_item["precision"] = by_item["tp"] / (by_item["tp"] + by_item["fp"]).replace(0, pd.NA)
    by_item["recall"] = by_item["tp"] / (by_item["tp"] + by_item["fn"]).replace(0, pd.NA)
    by_item["f1"] = (2 * by_item["precision"] * by_item["recall"]) / (by_item["precision"] + by_item["recall"])

    # overall totals
    total_n = int(work["item_key"].size)
    total_tp = int(work["tp"].sum())
    total_fp = int(work["fp"].sum())
    total_fn = int(work["fn"].sum())
    tot_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else float("nan")
    tot_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else float("nan")
    tot_f1 = (2 * tot_prec * tot_rec) / (tot_prec + tot_rec) if (tot_prec + tot_rec) else float("nan")
    total_row = pd.DataFrame(
        [
            {
                "item_key": "TOTAL",
                "n": total_n,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "precision": tot_prec,
                "recall": tot_rec,
                "f1": tot_f1,
            }
        ]
    )
    by_item = pd.concat([by_item, total_row], ignore_index=True)

    previews_df = pd.DataFrame(previews) if (collect_previews and previews) else None
    return detailed_out, by_item, previews_df


# ---------------- Public API ----------------


def validate_checklist(
    checklist: Union[str, Dict[str, Any]],
    examples: Union[str, pd.DataFrame],
    out_csv: str | None = None,
    by_item_csv: str | None = None,
    *,
    negation_fn: Optional[NegationFn] = None,  # optional override
    fp_window_chars: int = 120,
    substance_terms: Optional[Sequence[str]] = None,  # optional global vocab
    substance_window_chars: int = 120,
    return_previews: bool = False,  # opt into previews
    previews_csv: str | None = None,  # optional CSV path
) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Programmatic API.

    Parameters
    ----------
    checklist : path to checklist.py OR a dict loaded by caller
    examples  : path to examples file OR a DataFrame with columns:
                - item_key | expected | note_text   (or item_code instead of item_key)
                expected must be 0/1; optional `note_id` will be preserved in outputs.
    out_csv   : optional path to write detailed CSV
    by_item_csv: optional path to write by-item summary CSV
    negation_fn: optional function (text, span) -> bool (True if negated).
                 If not provided, a helper-style LEFT-ONLY 65-char window is used.
    fp_window_chars: character window around a hit to check `common_fp` (default 120).
    substance_terms: list of vocab strings to require near hits for items that
                     set `opioid: True` or `substance: True` in the checklist.
    substance_window_chars: context window around hit for substance vocab (default 120).
    return_previews: if True, return a third DataFrame of preview snippets for items with preview=True.
    previews_csv: optional path to write previews CSV when return_previews=True.
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

    detailed, by_item, previews = validate_rows(
        checklist,
        df,
        negation_fn=negation_fn or _helper_negation,  # ensure helper-style default
        fp_window_chars=fp_window_chars,
        substance_terms=substance_terms,
        substance_window_chars=substance_window_chars,
        collect_previews=return_previews,
    )

    if out_csv:
        detailed.to_csv(out_csv, index=False)
    if by_item_csv:
        by_item.to_csv(by_item_csv, index=False)
    if return_previews and previews is not None:
        if previews_csv:
            previews.to_csv(previews_csv, index=False)
        return detailed, by_item, previews

    return detailed, by_item
