"""
Helper file so that functions are stored separately from main execution file.
Refactor: unified gating utilities (substance/opioid, negation, common FP), plus
preview/highlighting helpers. Public API preserved; negation scope is opt-in.
"""

import os
import re
from typing import Iterable, List
from typing import Pattern as RePattern
from typing import Tuple, Union

import pandas as pd

# ============================================================
# Global flags & storage
# ============================================================

PRINT = False  # set by callers (debug mode)

# term storage (_terms by main)
TERMS_LIST: list[str] = []
TERMS_COMPILED: list[re.Pattern] = []

# Default scanning windows (tuned to current behavior)
WIN_SUBSTANCE = 100
WIN_NEGATION = 65
WIN_CFP = 65  # common false positive
WIN_DISCHARGE = 250

# ============================================================
# Logging / Debug helper
# ============================================================


def _dbg(msg: str):
    if PRINT:
        print(msg)


# ============================================================
# Regex & text utilities
# ============================================================


def _finditer(pat: RePattern | str, text: str):
    if isinstance(pat, re.Pattern):
        return pat.finditer(text)
    return re.finditer(pat, text, flags=re.IGNORECASE | re.MULTILINE)


def _search(term: RePattern | str, text: str):
    if isinstance(term, re.Pattern):
        return term.search(text) is not None
    return re.search(term, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def _compile_terms(terms: Iterable[str]) -> list[re.Pattern]:
    return [re.compile(re.escape(t), re.IGNORECASE | re.MULTILINE) for t in terms]


def set_terms(terms: list[str]) -> None:
    """
    Call once at startup to tell this helper which vocabulary to use.
    """
    global TERMS_LIST, TERMS_COMPILED
    TERMS_LIST = terms or []
    TERMS_COMPILED = _compile_terms(TERMS_LIST)
    _dbg(f"Using terms: {TERMS_LIST}")


def _window(text: str, start: int, stop: int, left: int, right: int) -> Tuple[int, int, str]:
    """
    Safe windowing around a span [start:stop), returning (L, R, slice).
    """
    L = max(0, start - max(0, left))
    R = min(len(text), stop + max(0, right))
    return L, R, text[L:R]


# ============================================================
# Note pre-processing
# ============================================================

# Memory usage (best-effort; harmless if unavailable)
try:
    total_memory, used_memory, free_memory = map(int, os.popen("free -t -m").readlines()[-1].split()[1:])
except Exception:
    total_memory, used_memory, free_memory = (0, 0, 0)


def remove_line_break(
    text: Union[str, bytes],
    break_markers: Union[str, List[str]] = r"\$\+\$",
    replacement: str = " ",
) -> str:
    s = text.decode() if isinstance(text, (bytes, bytearray)) else str(text)

    # Build regex pattern for markers
    if isinstance(break_markers, str):
        pattern = break_markers  # treat literal as a regex pattern
    else:
        parts = []
        for m in break_markers:
            if m and m[0] == "\\":
                parts.append(m)
            else:
                parts.append(re.escape(m))
        pattern = "|".join(parts)

    # 1) remove the markers
    s = re.sub(pattern, replacement, s)
    # 2) collapse any run of whitespace to a single space
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# Highlighting (for previews)
# ============================================================

NEGATION_CUES = [
    r"\bno\b",
    r"\bnot\b",
    r"\bnever\b",
    r"\bwithout\b",
    r"\bden(y|ies|ied|ying)\b",
    r"\bnegative(?:\s+for)?\b",
    r"\bneg(?:ative)?\.?\b",
    r"\bdidn['’]t\b",
]


def _first_span(pats: list[re.Pattern], text: str) -> tuple[int, int] | None:
    for p in pats:
        m = p.search(text)
        if m:
            return m.span()
    return None


def _apply_style(s: str, span: tuple[int, int], style: str, kind: str) -> str:
    a, b = span
    if a is None or b is None or a < 0 or b > len(s) or a >= b:
        return s

    if style == "ansi":  # inverse video
        code = "\x1b[7m"
        reset = "\x1b[0m"
        return s[:a] + code + s[a:b] + reset + s[b:]

    if style == "html":
        cls = {"hit": "hit", "sub": "sub", "neg": "neg"}.get(kind, "hit")
        return s[:a] + f"<mark class='{cls}'>" + s[a:b] + "</mark>" + s[b:]

    # default bracket markers
    tag = {"hit": "[[", "sub": "{{", "neg": "(("}.get(kind, "[[")
    end = {"hit": "]]", "sub": "}}", "neg": "))"}.get(kind, "]]")
    return s[:a] + tag + s[a:b] + end + s[b:]


def _highlight_snippet(
    snippet: str,
    rel_hit: tuple[int, int] | None,
    sub_span: tuple[int, int] | None,
    neg_span: tuple[int, int] | None,
    style: str = "brackets",
) -> str:
    s = snippet
    if rel_hit:
        s = _apply_style(s, rel_hit, style, "hit")

    def _refind(original: str, a: int, b: int) -> tuple[int, int] | None:
        piece = original[a:b]
        j = s.find(piece)
        return (j, j + len(piece)) if j >= 0 else None

    if sub_span:
        sub2 = _refind(snippet, *sub_span)
        if sub2:
            s = _apply_style(s, sub2, style, "sub")

    if neg_span:
        neg2 = _refind(snippet, *neg_span)
        if neg2:
            s = _apply_style(s, neg2, style, "neg")

    return s


# ============================================================
# Generic gating utilities (reusable)
# ============================================================


def gate_by_terms(
    df: pd.DataFrame,
    pat,
    in_col: str,
    out_col: str,
    terms: Iterable[str] | Iterable[re.Pattern],
    left_chars: int,
    right_chars: int,
    policy: str = "require",  # "require": keep rows with ANY term in window; "exclude": drop rows with ANY term
    note_col: str = "note_text",
) -> pd.DataFrame:
    """
    Generic gate around matches in `in_col` by scanning a window near each match.
    - 'require': out_col = 1 iff there exists a match for pat AND a term is present in its window
    - 'exclude': out_col = 1 iff there exists a match for pat AND NO term is present in its window
    """
    assert policy in {"require", "exclude"}
    df = df.copy()

    # If the input mask column isn't present, there's nothing to scan.
    if in_col not in df.columns:
        df[out_col] = 0
        return df

    hits = df[df[in_col].fillna(0).astype(int) > 0].copy()
    if hits.empty:
        # ensure out_col exists for downstream consumers
        df[out_col] = 0
        return df

    # Compile terms if needed
    term_pats = []
    for t in terms or []:
        if hasattr(t, "search"):
            term_pats.append(t)  # already compiled
        else:
            term_pats.append(re.compile(re.escape(str(t)), re.IGNORECASE | re.MULTILINE))

    def _row_ok(text: str) -> int:
        for m in _finditer(pat, text):
            s, e = m.span()
            L, R, ctx = _window(text, s, e, left_chars, right_chars)
            found = any(p.search(ctx) for p in term_pats) if term_pats else False
            if policy == "require" and found:
                return 1
            if policy == "exclude" and not found:
                return 1
        return 0

    hits[out_col] = hits[note_col].apply(_row_ok)

    # Avoid suffix collisions when out_col == in_col (in-place pruning)
    df.drop(columns=[out_col], errors="ignore", inplace=True)
    df = df.merge(hits[["note_id", out_col]], on="note_id", how="left")

    df[out_col] = df[out_col].fillna(0).astype(int)
    _dbg(f"[GATE] {out_col}: policy={policy}, left={left_chars}, right={right_chars}, terms={len(term_pats)}")
    return df


def gate_by_cues_left(
    df: pd.DataFrame,
    pat,
    in_col: str,
    out_col: str,
    cues,
    left_chars: int,
    note_col: str = "note_text",
) -> pd.DataFrame:
    # left-only behavior preserved (back-compat convenience)
    return gate_by_cues(
        df=df,
        pat=pat,
        in_col=in_col,
        out_col=out_col,
        cues=cues,
        left_chars=left_chars,
        right_chars=0,
        note_col=note_col,
    )


def gate_by_cues(
    df: pd.DataFrame,
    pat,
    in_col: str,
    out_col: str,
    cues,
    left_chars: int,
    right_chars: int,
    note_col: str = "note_text",
) -> pd.DataFrame:
    """
    Negation-style gate scanning left/right/both sides.
    out_col = 1 iff there exists a match for pat AND NO cue is found in the scanned window(s).
    """
    df = df.copy()
    hits = df[df[in_col].fillna(0).astype(int) > 0].copy()
    if hits.empty:
        df[out_col] = 0
        return df

    cue_pats = [c if hasattr(c, "search") else re.compile(str(c), re.IGNORECASE | re.MULTILINE) for c in (cues or [])]

    def _row_ok(text: str) -> int:
        for m in _finditer(pat, text):
            s, e = m.span()
            L, R, ctx = _window(text, s, e, left_chars, right_chars)
            left = ctx[: s - L]
            right = ctx[e - L :]
            # keep iff NO cue appears in the scanned sides
            if not any(p.search(left) for p in cue_pats) and not any(p.search(right) for p in cue_pats):
                return 1
        return 0

    hits[out_col] = hits[note_col].apply(_row_ok)
    df = df.merge(hits[["note_id", out_col]], on="note_id", how="left")
    df[out_col] = df[out_col].fillna(0).astype(int)
    _dbg(f"[GATE] {out_col}: negation-window left={left_chars}, right={right_chars}, cues={len(cue_pats)}")
    return df


# ============================================================
# Public gates (back-compatible names)
# ============================================================


def check_for_substance(pat, col_name, col_name_substance, df_searched, span=WIN_SUBSTANCE):
    """
    Require at least one substance term near a hit.
    (keeps existing signature)
    """
    return gate_by_terms(
        df=df_searched,
        pat=pat,
        in_col=col_name,
        out_col=col_name_substance,
        terms=TERMS_LIST,
        left_chars=span,
        right_chars=span,
        policy="require",
    )


def check_negation(
    pat,
    col_name,
    col_name_negated,
    df_searched,
    t=None,
    neg=True,
    span=WIN_NEGATION,
    *,
    side: str = "left",  # "left" (default), "right", or "both"
):
    """
    Keep hits that are NOT negated by cues in the chosen scope.
    side: "left" (default, back-compatible), "right", or "both".
    """
    t = t or []
    cues = ["no ", "not ", "denie", "denial", "doubt", "never", "negative", "without", "neg", "didn't"]
    if not neg:
        cues = []
    cues.extend(t)

    side = (side or "left").lower()
    if side == "left":
        return gate_by_cues_left(df_searched, pat, col_name, col_name_negated, cues, left_chars=span)
    if side == "right":
        return gate_by_cues(df_searched, pat, col_name, col_name_negated, cues, left_chars=0, right_chars=span)
    # both
    return gate_by_cues(df_searched, pat, col_name, col_name_negated, cues, left_chars=span, right_chars=span)


def check_common_false_positives(pat, df_searched, col_name_fp, common_fp, span=WIN_CFP):
    """
    Exclude hits if ANY common-fp term appears near them.
    (keeps existing signature)
    """
    return gate_by_terms(
        df=df_searched,
        pat=pat,
        in_col=col_name_fp,  # note: input mask is the active mask
        out_col=col_name_fp,  # in-place pruning behavior matches original API
        terms=common_fp or [],
        left_chars=span,
        right_chars=span,
        policy="exclude",
    )


def discharge_instructions(pat, df_searched, col_name_discharge, span=WIN_DISCHARGE):
    """
    Exclude hits that occur within discharge-instruction contexts.
    (keeps existing signature)
    """
    discharge_terms = ["discharge instructions", "no results for"]
    return gate_by_terms(
        df=df_searched,
        pat=pat,
        in_col=col_name_discharge,
        out_col=col_name_discharge,
        terms=discharge_terms,
        left_chars=span,
        right_chars=span,
        policy="exclude",
    )


# ============================================================
# PREVIEWS (gated + highlighting)
# ============================================================


def write_previews_for_item(
    df_searched: pd.DataFrame,
    item_key: str,
    pat,
    mask_col: str,
    note_col: str = "note_text",
    note_id_col: str = "note_id",
    n_notes: int | None = 10,
    left_chars: int = 120,
    right_chars: int = 120,
    csv_path: str | None = None,
    outfile: str | None = None,
    *,
    highlight: bool = True,
    highlight_style: str = "brackets",  # 'brackets' | 'ansi' | 'html'
):
    """
    Emit preview snippets from rows where mask_col == 1.
    Adds 'snippet_marked' with highlight on hit + first nearby substance + first negation cue.
    RETURNS: list of dict rows (so callers can aggregate to a DataFrame).
    """
    pat_iter = (
        pat.finditer
        if hasattr(pat, "finditer")
        else (lambda t: re.finditer(pat, t, flags=re.IGNORECASE | re.MULTILINE))
    )
    sub_pats = TERMS_COMPILED[:] if TERMS_COMPILED else []
    neg_pats = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in NEGATION_CUES]

    rows = []
    hits = df_searched[df_searched[mask_col].fillna(0).astype(int) > 0]
    if hits.empty:
        return rows  # return empty

    sample = (
        hits
        if (n_notes is None or (isinstance(n_notes, int) and n_notes < 0))
        else hits.sample(min(int(n_notes), len(hits)), random_state=123)
    )

    fh = open(outfile, "a", encoding="utf-8") if outfile else None
    try:
        for _, r in sample.iterrows():
            text = r.get(note_col, "") or ""
            nid = r.get(note_id_col, None)
            found = False
            for m in pat_iter(text):
                s, e = m.span()
                L, R, snippet = _window(text, s, e, left_chars, right_chars)
                rel_hit = (s - L, e - L)
                sub_span = _first_span(sub_pats, snippet) if sub_pats else None
                neg_span = _first_span(neg_pats, snippet)

                snippet_marked = (
                    _highlight_snippet(snippet, rel_hit, sub_span, neg_span, style=highlight_style)
                    if highlight
                    else snippet
                )

                rows.append(
                    {
                        "item_key": item_key,
                        "note_id": nid,
                        "span_start": s,
                        "span_end": e,
                        "snippet": snippet,
                        "snippet_marked": snippet_marked,
                    }
                )

                if fh:
                    tag = f"~~~ {nid} ~~~" if nid is not None else "~~~ row ~~~"
                    fh.write(tag + "\n" + snippet_marked + "\n\n")

                found = True
                break

            if not found and fh:
                tag = f"~~~ {nid} ~~~" if nid is not None else "~~~ row ~~~"
                fh.write(tag + "\n[no span found in text]\n\n")
    finally:
        if fh:
            fh.close()

    if csv_path:
        header = not os.path.exists(csv_path)
        pd.DataFrame(rows).to_csv(csv_path, mode=("w" if header else "a"), index=False, header=header)

    return rows  # <<< NEW: return collected preview rows


# Legacy/base-only preview (kept for completeness)
def previews_batch(
    checklist,
    df_summarized,
    n_notes: int = 2,
    span: int = 300,
    outfile: str | None = None,
    *,
    return_df: bool = False,
    csv_path: str | None = None,
):
    """
    Collect preview snippets for items with `preview: True` using the BASE column only.
    Schema: ['item_key', 'note_id', 'span_start', 'span_end', 'snippet']
    """
    import re
    import sys

    import pandas as pd

    class _Writer:
        def __init__(self, path: str | None):
            self.path = path
            self._orig = None
            self._fh = None

        def __enter__(self):
            if self.path:
                self._orig = sys.stdout
                self._fh = open(self.path, "a", encoding="utf-8")
                sys.stdout = self._fh
            return self

        def __exit__(self, exc_type, exc, tb):
            if self._fh:
                sys.stdout = self._orig
                self._fh.close()

    if "note_text" not in df_summarized.columns:
        raise ValueError(
            "previews_batch requires 'note_text' in df_summarized. "
            "Run extract/extract_df with include_note_text=True or call before dropping note_text."
        )

    rows = []

    def _iter_matches(pat, text: str):
        if hasattr(pat, "finditer"):
            return pat.finditer(text)
        return re.finditer(pat, text, flags=re.IGNORECASE | re.MULTILINE)

    with _Writer(outfile):
        for item_key, cfg in checklist.items():
            if not cfg.get("preview"):
                continue

            pat = cfg["pat"]
            base = cfg["col_name"]
            if base not in df_summarized.columns:
                continue

            hits = df_summarized[df_summarized[base].fillna(0).astype(int) > 0]
            if hits.empty:
                continue

            if n_notes is None or (isinstance(n_notes, int) and n_notes < 0):
                sample = hits
            else:
                sample = hits.sample(min(int(n_notes), len(hits)), random_state=123)

            for _, r in sample.iterrows():
                note_id = str(r["note_id"]) if "note_id" in r else None
                text = r.get("note_text", "") or ""
                found = False
                for m in _iter_matches(pat, text):
                    s, e = m.span()
                    start = max(0, s - span)
                    end = min(len(text), e + span)
                    snippet = text[start:end]

                    if outfile:
                        tag = f"~~~ {note_id} ~~~" if note_id is not None else "~~~ row ~~~"
                        print(tag)
                        print(snippet)
                        print()

                    rows.append(
                        {
                            "item_key": item_key,
                            "note_id": note_id,
                            "span_start": s,
                            "span_end": e,
                            "snippet": snippet,
                        }
                    )
                    found = True
                    break
                if not found and outfile:
                    tag = f"~~~ {note_id} ~~~" if note_id is not None else "~~~ row ~~~"
                    print(tag)
                    print("[no span found in text]")
                    print()

    df = pd.DataFrame(rows, columns=["item_key", "note_id", "span_start", "span_end", "snippet"])
    if csv_path:
        df.to_csv(csv_path, index=False)
    if return_df:
        return df
    return None


# ============================================================
# Main extract + gated previews
# ============================================================


def regex_extract(
    checklist,
    df_to_analyze,
    metadata,
    preview_count,
    expected_row_count,
    exclude_discharge_mentions: bool = True,
    *,
    preview_span: int = 120,
    preview_csv: str | None = None,
    preview_file: str | None = None,
    negation_scope: str = "left",  # ← NEW: "left" (default), "right", "both"
    return_previews: bool = False,  # ← NEW
):
    """
    Applies the checklist of regex searches to the data frame, with optional substance and negation checks.
    Also writes gated previews when `preview: True` on a checklist item.

    negation_scope controls where negation cues are searched relative to the hit:
    - "left"  : ONLY to the left (current default behavior)
    - "right" : ONLY to the right
    - "both"  : symmetric window on both sides
    If return_previews=True, returns (metadata_df, previews_df).
    """
    # Validate scope early (friendlier error than silent mismatch)
    negation_scope = (negation_scope or "left").lower()
    if negation_scope not in {"left", "right", "both"}:
        raise ValueError("negation_scope must be 'left', 'right', or 'both'")

    _dbg(
        f"[DEBUG] Starting regex_extract: df_to_analyze.shape={df_to_analyze.shape}, "
        f"metadata.shape={metadata.shape}, expected_row_count={expected_row_count}"
    )

    previews_acc: list[dict] = []  # NEW: collect preview rows across items

    for i in checklist:
        _dbg(f"\n[DEBUG] Checklist item index: {i}")
        actual_rows = df_to_analyze.shape[0]
        _dbg(f"[DEBUG]  → df_to_analyze has {actual_rows} rows")
        assert actual_rows == expected_row_count, f"Row counts do not match ({actual_rows} != {expected_row_count})"

        pat = checklist[i]["pat"]
        col_name = checklist[i]["col_name"]
        _dbg(f"[DEBUG]  → pattern='{pat.pattern if hasattr(pat, 'pattern') else pat}', col_name='{col_name}'")

        # Treat either 'substance' OR 'opioid' as the gating flag
        has_substance = bool(checklist[i].get("substance") or checklist[i].get("opioid"))
        has_negation = bool(checklist[i].get("negation"))
        _dbg(f"[DEBUG]  → substance={has_substance}, negation={has_negation}")

        # Initial search
        _dbg(f"[DEBUG]  → Calling regex_search_file for '{col_name}'")
        df_searched = regex_search_file(pat, col_name, df_to_analyze, metadata, preview=True)
        base_sum = int(pd.to_numeric(df_searched[col_name], errors="coerce").fillna(0).sum())
        _dbg(f"[DEBUG]  → After regex_search_file: df_searched['{col_name}'].sum()={base_sum}")

        active_col = col_name

        # substance (+ optional negation) branch
        if has_substance:
            _dbg(f"[DEBUG]  → Entering substance branch for '{col_name}'")
            if base_sum > 0:
                df_searched = check_for_substance(
                    pat, col_name, f"{col_name}_SUBSTANCE_MATCHED", df_searched, span=WIN_SUBSTANCE
                )
                active_col = f"{col_name}_SUBSTANCE_MATCHED"
                sub_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
                _dbg(f"[DEBUG]    • After check_for_substance: {sub_sum} matches")

                if has_negation:
                    _dbg("[DEBUG]    • Entering negation checks")
                    if sub_sum > 0:
                        df_searched = check_negation(
                            pat,
                            active_col,
                            f"{active_col}_NEG",
                            df_searched,
                            t=[],
                            neg=True,
                            span=WIN_NEGATION,
                            side=negation_scope,  # ← scope injected here
                        )
                        active_col = f"{active_col}_NEG"
                        neg_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
                        _dbg(f"[DEBUG]    • After check_negation: {neg_sum} kept")
                    else:
                        df_searched[f"{active_col}_NEG"] = 0
                        active_col = f"{active_col}_NEG"
                        _dbg(f"[DEBUG]    • No substance matches; set {active_col}=0")
            else:
                df_searched[f"{col_name}_SUBSTANCE_MATCHED"] = 0
                if has_negation:
                    df_searched[f"{col_name}_SUBSTANCE_MATCHED_NEG"] = 0
                    active_col = f"{col_name}_SUBSTANCE_MATCHED_NEG"
                else:
                    active_col = f"{col_name}_SUBSTANCE_MATCHED"
                _dbg(f"[DEBUG]    • No initial matches; zeroed {active_col}")

        # negation-only branch
        elif has_negation:
            _dbg(f"[DEBUG]  → Entering negation-only branch for '{col_name}'")
            if base_sum > 0:
                df_searched = check_negation(
                    pat,
                    col_name,
                    f"{col_name}_NEG",
                    df_searched,
                    t=[],
                    neg=True,
                    span=WIN_NEGATION,
                    side=negation_scope,  # ← scope injected
                )
                active_col = f"{col_name}_NEG"
                neg_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
                _dbg(f"[DEBUG]    • After check_negation: {neg_sum} negated")
            else:
                df_searched[f"{col_name}_NEG"] = 0
                active_col = f"{col_name}_NEG"
                _dbg(f"[DEBUG]    • No initial matches; set {active_col}=0")

        else:
            _dbg(f"[DEBUG]  → No substance/negation flags for '{col_name}' (base branch)")

        # Pruning policy: prune if negation is present OR there is no substance gate
        should_prune = has_negation or not has_substance

        if active_col in df_searched.columns:
            pre_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
            _dbg(f"[DEBUG]    • Before pruning on {active_col}: {pre_sum} kept")

            if should_prune and pre_sum > 0:
                if exclude_discharge_mentions:
                    df_searched = discharge_instructions(pat, df_searched, active_col, span=WIN_DISCHARGE)
                    post_dis = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
                    _dbg(f"[DEBUG]    • After discharge_instructions on {active_col}: {post_dis} kept")

                common_fp = checklist[i].get("common_fp") or []
                if common_fp:
                    df_searched = check_common_false_positives(pat, df_searched, active_col, common_fp, span=WIN_CFP)
                    post_fp = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
                    _dbg(f"[DEBUG]    • After common FP pruning on {active_col}: {post_fp} kept")
            else:
                _dbg(
                    f"[DEBUG]    • Skipping pruning for {active_col} (matches={pre_sum}, has_substance={has_substance}, has_negation={has_negation})"
                )
        else:
            _dbg(f"[DEBUG]    • Skipping pruning (missing column {active_col})")

        if active_col not in df_searched.columns:
            df_searched[active_col] = 0

        # -------- PREVIEWS (gated): from final mask --------
        if checklist[i].get("preview"):
            _dbg(f"[DEBUG]  → Writing previews for '{col_name}' from mask '{active_col}'")
            sub_win = WIN_SUBSTANCE if has_substance else 0
            neg_win = WIN_NEGATION if has_negation else 0
            fp_win = WIN_CFP if (checklist[i].get("common_fp")) else 0
            dis_win = WIN_DISCHARGE if exclude_discharge_mentions else 0
            left_req = max(preview_span, sub_win, neg_win, fp_win, dis_win)
            right_req = left_req

            rows = write_previews_for_item(  # NEW: capture rows
                df_searched=df_searched,
                item_key=i,
                pat=pat,
                mask_col=active_col,
                n_notes=preview_count,
                left_chars=left_req,
                right_chars=right_req,
                csv_path=preview_csv,
                outfile=preview_file,
                highlight=True,
                highlight_style="brackets",
            )
            if rows:
                previews_acc.extend(rows)

        # Build merge column set, ensuring active_col is present
        merge_cols = ["note_id", col_name]
        if has_substance:
            merge_cols.append(f"{col_name}_SUBSTANCE_MATCHED")
        if has_negation and has_substance:
            merge_cols.append(f"{col_name}_SUBSTANCE_MATCHED_NEG")
        elif has_negation and not has_substance:
            merge_cols.append(f"{col_name}_NEG")
        if active_col not in merge_cols:
            merge_cols.append(active_col)

        for mc in merge_cols:
            if mc not in df_searched.columns:
                df_searched[mc] = 0

        cur_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
        _dbg(f"[DEBUG]  → SUMMARY '{col_name}': base={base_sum} | active({active_col})={cur_sum}")
        _dbg(f"[DEBUG]  → Merging columns {merge_cols} into metadata (active_col={active_col})")

        metadata = metadata.merge(df_searched[merge_cols], on="note_id", how="left")

    # Final merge for note_text
    metadata = metadata.merge(df_to_analyze[["note_id", "note_text"]], on="note_id", how="left")
    _dbg(f"[DEBUG] Finished regex_extract: metadata.shape={metadata.shape}")

    if return_previews:
        previews_df = (
            pd.DataFrame(previews_acc)
            if previews_acc
            else pd.DataFrame(columns=["item_key", "note_id", "span_start", "span_end", "snippet", "snippet_marked"])
        )
        return metadata, previews_df

    return metadata


# ============================================================
# Primitive search
# ============================================================


def regex_search_file(pat, new_col_name, df_to_search, metadata, preview=True):
    """
    Search the note text in each row for `pat` and count matches into `new_col_name`.
    Returns a df merged with `metadata` on note_id.
    """
    from re import Pattern

    counts = pd.DataFrame(columns=["note_id", "note_text", new_col_name])

    def pat_search(text):
        if isinstance(pat, Pattern):
            return len(pat.findall(text))
        else:
            return len(re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE))

    df_to_search["note_text"] = df_to_search["note_text"].apply(remove_line_break)
    df_to_search[new_col_name] = df_to_search["note_text"].apply(pat_search)
    counts = pd.concat([counts, df_to_search], sort=True)

    keep_cols = ["note_id", new_col_name]
    if preview:
        keep_cols.append("note_text")

    df_searched = metadata.merge(counts[keep_cols], how="left", on="note_id")
    return df_searched


# ============================================================
# Misc redactors
# ============================================================


def remove_tobacco_mentions(text):
    """Mask mentions of no tobacco use in the text."""
    tobacco_pattern = (
        r"(Tob(?:acco)?[ -]*(?:use)?[: -]*\b(None|Never|No\s+use|abstains|denies use,? never|ever)\b|"
        r"Smoking[: -]*None|Smoker[: -]*(?:never|no))"
    )
    return re.sub(tobacco_pattern, "Tobacco: [Redacted]", str(text), flags=re.IGNORECASE)


# ============================================================
# Console preview (legacy)
# ============================================================


def preview_string_matches(pat, col_name, df_searched, col_check=False, n_notes=10, span=100):
    """
    Console preview: print color-highlighted excerpts for debugging.
    """
    if col_check and col_name not in df_searched.columns:
        raise KeyError(f"Column '{col_name}' not found in df_searched")

    hits = df_searched[df_searched[col_name] > 0]
    hit_count = len(hits)
    if hit_count == 0 or n_notes <= 0:
        return

    k = min(n_notes, hit_count)
    matches = hits.sample(k, random_state=123)

    for i in range(matches.shape[0]):
        if PRINT:
            print(str(matches["note_id"].iloc[i]))
        text = matches["note_text"].iloc[i]
        for m in _finditer(pat, text):
            start, stop = m.span()
            L, R, snippet = _window(text, start, stop, span, span)

            # crude terminal highlight of the hit + any term/neg cues in the snippet
            marked = snippet
            # hit
            marked = (
                marked[0:span]
                + "\x1b[7m"  # inverse
                + marked[span : span + (stop - start)]
                + "\x1b[0m"
                + marked[span + (stop - start) :]
            )
            # substance terms
            for term in TERMS_LIST:
                x = re.search(term, marked, flags=re.IGNORECASE | re.MULTILINE)
                if x:
                    s2, e2 = x.span()
                    marked = marked[:s2] + "\x1b[7m" + marked[s2:e2] + "\x1b[0m" + marked[e2:]
            # neg cues
            for cue in ["no ", "not ", "denies", "denial", "doubt", "never", "negative for"]:
                x = re.search(cue, marked, flags=re.IGNORECASE | re.MULTILINE)
                if x:
                    s2, e2 = x.span()
                    marked = marked[:s2] + "\x1b[7m" + marked[s2:e2] + "\x1b[0m" + marked[e2:]

            if PRINT:
                print(marked)
                print("\n")
