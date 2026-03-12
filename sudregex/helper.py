"""
Helper utilities for regex-driven extraction, gating, and preview generation.

This module contains the reusable text-processing and row-wise matching logic
used by the public extraction APIs. It provides pattern counting, substance
and negation gating, false-positive and discharge-context pruning, preview
generation, and backend-aware row execution for serial and parallel workflows.
"""

import os
import re
from functools import partial
from typing import Iterable, List
from typing import Pattern as RePattern
from typing import Tuple, Union

import pandas as pd

# ============================================================
# Global flags and shared state
# ============================================================

# Debug logging flag, configured by the caller.
PRINT = False

# Active term vocabulary configured at runtime.
TERMS_LIST: list[str] = []
TERMS_COMPILED: list[re.Pattern] = []

# Default scan windows, in characters.
WIN_SUBSTANCE = 100
WIN_NEGATION = 65
WIN_CFP = 65
WIN_DISCHARGE = 250

# ============================================================
# Logging helper
# ============================================================


def _dbg(msg: str):
    """Print a debug message when debug mode is enabled."""
    if PRINT:
        print(msg)


# ============================================================
# Parallel apply helper
# ============================================================


def _apply_series(
    series: pd.Series,
    func,
    use_parallel: bool = False,
    parallel_backend: str | None = None,
    n_workers: int | None = None,
) -> pd.Series:
    """
    Apply a function to a pandas Series using the configured execution backend.

    Supported modes:
    - serial execution via pandas `.apply()`
    - `pandarallel` via `.parallel_apply()`
    - `loky` via joblib `Parallel(..., backend="loky")`
    """
    if not use_parallel or not parallel_backend:
        return series.apply(func)

    backend = parallel_backend.lower()

    if backend == "pandarallel":
        if hasattr(series, "parallel_apply"):
            return series.parallel_apply(func)
        return series.apply(func)

    if backend == "loky":
        try:
            from joblib import Parallel, delayed
        except ImportError as e:
            raise ImportError("joblib is required for parallel_backend='loky'") from e

        values = series.tolist()
        results = Parallel(n_jobs=n_workers or -1, backend="loky")(delayed(func)(value) for value in values)
        return pd.Series(results, index=series.index)

    raise ValueError(f"Unsupported parallel_backend={parallel_backend!r}")


# ============================================================
# Regex and pattern utilities
# ============================================================


def _finditer(pat: RePattern | str, text: str):
    """Return an iterator over pattern matches for a compiled or raw pattern."""
    if isinstance(pat, re.Pattern):
        return pat.finditer(text)
    return re.finditer(pat, text, flags=re.IGNORECASE | re.MULTILINE)


def _search(term: RePattern | str, text: str):
    """Return True if a compiled or raw pattern is found in the text."""
    if isinstance(term, re.Pattern):
        return term.search(text) is not None
    return re.search(term, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def _compile_terms(terms: Iterable[str]) -> list[re.Pattern]:
    """Compile a list of literal terms into case-insensitive regex patterns."""
    return [re.compile(re.escape(t), re.IGNORECASE | re.MULTILINE) for t in terms]


def _count_pattern_matches(pat, text: str) -> int:
    """
    Count all matches for a compiled regex or raw regex string in the text.
    """
    if isinstance(pat, re.Pattern):
        return len(pat.findall(text))
    return len(re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE))


def _pattern_to_payload(pat) -> tuple[str, int]:
    """
    Convert a compiled or raw pattern into a serializable `(pattern, flags)` payload.

    This is used by backends such as Loky that benefit from passing simple
    serializable inputs to worker processes.
    """
    if isinstance(pat, re.Pattern):
        return pat.pattern, pat.flags
    return str(pat), re.IGNORECASE | re.MULTILINE


def _compile_payload(payload: tuple[str, int]) -> re.Pattern:
    """Compile a `(pattern, flags)` payload back into a regex object."""
    pattern_text, flags = payload
    return re.compile(pattern_text, flags)


def _term_to_payload(term) -> tuple[str, int]:
    """
    Convert a term or compiled regex into a serializable `(pattern, flags)` payload.
    """
    if isinstance(term, re.Pattern):
        return term.pattern, term.flags
    return re.escape(str(term)), re.IGNORECASE | re.MULTILINE


def _count_pattern_matches_from_payload(text: str, pat_payload: tuple[str, int]) -> int:
    """Count pattern matches using a serialized regex payload."""
    pat = _compile_payload(pat_payload)
    return len(pat.findall(text))


def set_terms(terms: list[str]) -> None:
    """
    Set the active vocabulary used by helper-level gating and previews.
    """
    global TERMS_LIST, TERMS_COMPILED
    TERMS_LIST = terms or []
    TERMS_COMPILED = _compile_terms(TERMS_LIST)
    _dbg(f"Using terms: {TERMS_LIST}")


def _window(text: str, start: int, stop: int, left: int, right: int) -> Tuple[int, int, str]:
    """
    Return a safe context window around a match span.

    Returns:
    - left bound
    - right bound
    - text slice within those bounds
    """
    L = max(0, start - max(0, left))
    R = min(len(text), stop + max(0, right))
    return L, R, text[L:R]


# ============================================================
# Note preprocessing
# ============================================================


def remove_line_break(
    text: Union[str, bytes],
    break_markers: Union[str, List[str]] = r"\$\+\$",
    replacement: str = " ",
) -> str:
    """
    Replace configured break markers and collapse repeated whitespace.

    This is used to normalize note text before extraction.
    """
    s = text.decode() if isinstance(text, (bytes, bytearray)) else str(text)

    if isinstance(break_markers, str):
        pattern = break_markers
    else:
        parts = []
        for m in break_markers:
            if m and m[0] == "\\":
                parts.append(m)
            else:
                parts.append(re.escape(m))
        pattern = "|".join(parts)

    s = re.sub(pattern, replacement, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# Highlighting helpers for previews
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
    """Return the span of the first matching pattern in the text, if any."""
    for p in pats:
        m = p.search(text)
        if m:
            return m.span()
    return None


def _apply_style(s: str, span: tuple[int, int], style: str, kind: str) -> str:
    """
    Apply highlight markup to a span using the requested output style.
    """
    a, b = span
    if a is None or b is None or a < 0 or b > len(s) or a >= b:
        return s

    if style == "ansi":
        code = "\x1b[7m"
        reset = "\x1b[0m"
        return s[:a] + code + s[a:b] + reset + s[b:]

    if style == "html":
        cls = {"hit": "hit", "sub": "sub", "neg": "neg"}.get(kind, "hit")
        return s[:a] + f"<mark class='{cls}'>" + s[a:b] + "</mark>" + s[b:]

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
    """
    Highlight a preview snippet for the match, nearby substance term, and negation cue.
    """
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
# Row worker helpers
# ============================================================


def _gate_by_terms_row(
    text: str,
    *,
    pat_payload: tuple[str, int],
    term_payloads: list[tuple[str, int]],
    left_chars: int,
    right_chars: int,
    policy: str,
) -> int:
    """
    Evaluate a single row for term-based gating.

    Returns 1 when the row passes the configured gate policy, otherwise 0.
    """
    pat = _compile_payload(pat_payload)
    term_pats = [_compile_payload(payload) for payload in term_payloads]

    for m in pat.finditer(text):
        s, e = m.span()
        _, _, ctx = _window(text, s, e, left_chars, right_chars)
        found = any(p.search(ctx) for p in term_pats) if term_pats else False
        if policy == "require" and found:
            return 1
        if policy == "exclude" and not found:
            return 1
    return 0


def _gate_by_cues_row(
    text: str,
    *,
    pat_payload: tuple[str, int],
    cue_payloads: list[tuple[str, int]],
    left_chars: int,
    right_chars: int,
) -> int:
    """
    Evaluate a single row for cue-based gating.

    Returns 1 when the row is not negated within the requested window, otherwise 0.
    """
    pat = _compile_payload(pat_payload)
    cue_pats = [_compile_payload(payload) for payload in cue_payloads]

    for m in pat.finditer(text):
        s, e = m.span()
        L, _, ctx = _window(text, s, e, left_chars, right_chars)
        left = ctx[: s - L]
        right = ctx[e - L :]
        if not any(p.search(left) for p in cue_pats) and not any(p.search(right) for p in cue_pats):
            return 1
    return 0


# ============================================================
# Generic gating utilities
# ============================================================


def gate_by_terms(
    df: pd.DataFrame,
    pat,
    in_col: str,
    out_col: str,
    terms: Iterable[str] | Iterable[re.Pattern],
    left_chars: int,
    right_chars: int,
    policy: str = "require",
    note_col: str = "note_text",
    use_parallel: bool = False,
    parallel_backend: str | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Gate matches by scanning for term presence or absence near each hit.

    Policies:
    - `require`: keep a row only when at least one configured term appears near a hit
    - `exclude`: keep a row only when no configured term appears near a hit
    """
    if policy not in {"require", "exclude"}:
        raise ValueError("policy must be 'require' or 'exclude'")

    df = df.copy()

    if in_col not in df.columns:
        df[out_col] = 0
        return df

    hits = df[df[in_col].fillna(0).astype(int) > 0].copy()
    if hits.empty:
        df[out_col] = 0
        return df

    pat_payload = _pattern_to_payload(pat)
    term_payloads = [_term_to_payload(t) for t in (terms or [])]

    row_func = partial(
        _gate_by_terms_row,
        pat_payload=pat_payload,
        term_payloads=term_payloads,
        left_chars=left_chars,
        right_chars=right_chars,
        policy=policy,
    )

    hits[out_col] = _apply_series(
        hits[note_col],
        row_func,
        use_parallel=use_parallel,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
    )

    df.drop(columns=[out_col], errors="ignore", inplace=True)
    df = df.merge(hits[["note_id", out_col]], on="note_id", how="left")

    df[out_col] = df[out_col].fillna(0).astype(int)
    _dbg(f"[GATE] {out_col}: policy={policy}, left={left_chars}, right={right_chars}, terms={len(term_payloads)}")
    return df


def gate_by_cues_left(
    df: pd.DataFrame,
    pat,
    in_col: str,
    out_col: str,
    cues,
    left_chars: int,
    note_col: str = "note_text",
    use_parallel: bool = False,
    parallel_backend: str | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper for cue-based gating using only a left-side scan window.
    """
    return gate_by_cues(
        df=df,
        pat=pat,
        in_col=in_col,
        out_col=out_col,
        cues=cues,
        left_chars=left_chars,
        right_chars=0,
        note_col=note_col,
        use_parallel=use_parallel,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
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
    use_parallel: bool = False,
    parallel_backend: str | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Gate matches by scanning for cue terms in the surrounding context window.

    A row passes when a hit is present and no configured cue is found within the
    requested left/right scan window.
    """
    df = df.copy()
    hits = df[df[in_col].fillna(0).astype(int) > 0].copy()
    if hits.empty:
        df[out_col] = 0
        return df

    pat_payload = _pattern_to_payload(pat)
    cue_payloads = [_term_to_payload(c) for c in (cues or [])]

    row_func = partial(
        _gate_by_cues_row,
        pat_payload=pat_payload,
        cue_payloads=cue_payloads,
        left_chars=left_chars,
        right_chars=right_chars,
    )

    hits[out_col] = _apply_series(
        hits[note_col],
        row_func,
        use_parallel=use_parallel,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
    )

    df = df.merge(hits[["note_id", out_col]], on="note_id", how="left")
    df[out_col] = df[out_col].fillna(0).astype(int)
    _dbg(f"[GATE] {out_col}: negation-window left={left_chars}, right={right_chars}, cues={len(cue_payloads)}")
    return df


# ============================================================
# Public gate wrappers
# ============================================================


def check_for_substance(
    pat,
    col_name,
    col_name_substance,
    df_searched,
    span=WIN_SUBSTANCE,
    *,
    use_parallel: bool = False,
    parallel_backend: str | None = None,
    n_workers: int | None = None,
):
    """
    Require at least one configured substance term near a match.
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
        use_parallel=use_parallel,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
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
    side: str = "left",
    use_parallel: bool = False,
    parallel_backend: str | None = None,
    n_workers: int | None = None,
):
    """
    Exclude hits negated by configured cue terms within the requested scope.

    `side` may be:
    - `"left"`
    - `"right"`
    - `"both"`
    """
    t = t or []
    cues = ["no ", "not ", "denie", "denial", "doubt", "never", "negative", "without", "neg", "didn't"]
    if not neg:
        cues = []
    cues.extend(t)

    side = (side or "left").lower()
    if side == "left":
        return gate_by_cues_left(
            df_searched,
            pat,
            col_name,
            col_name_negated,
            cues,
            left_chars=span,
            use_parallel=use_parallel,
            parallel_backend=parallel_backend,
            n_workers=n_workers,
        )
    if side == "right":
        return gate_by_cues(
            df_searched,
            pat,
            col_name,
            col_name_negated,
            cues,
            left_chars=0,
            right_chars=span,
            use_parallel=use_parallel,
            parallel_backend=parallel_backend,
            n_workers=n_workers,
        )
    return gate_by_cues(
        df_searched,
        pat,
        col_name,
        col_name_negated,
        cues,
        left_chars=span,
        right_chars=span,
        use_parallel=use_parallel,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
    )


def check_common_false_positives(
    pat,
    df_searched,
    col_name_fp,
    common_fp,
    span=WIN_CFP,
    *,
    use_parallel: bool = False,
    parallel_backend: str | None = None,
    n_workers: int | None = None,
):
    """
    Exclude matches when common false-positive terms are found nearby.
    """
    return gate_by_terms(
        df=df_searched,
        pat=pat,
        in_col=col_name_fp,
        out_col=col_name_fp,
        terms=common_fp or [],
        left_chars=span,
        right_chars=span,
        policy="exclude",
        use_parallel=use_parallel,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
    )


def discharge_instructions(
    pat,
    df_searched,
    col_name_discharge,
    span=WIN_DISCHARGE,
    *,
    use_parallel: bool = False,
    parallel_backend: str | None = None,
    n_workers: int | None = None,
):
    """
    Exclude matches found within discharge-instruction contexts.
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
        use_parallel=use_parallel,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
    )


# ============================================================
# Preview generation
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
    highlight_style: str = "brackets",
):
    """
    Write preview snippets for rows where `mask_col == 1`.

    Each returned row includes:
    - item key
    - note identifier
    - match span
    - raw snippet
    - highlighted snippet
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
        return rows

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
                L, _, snippet = _window(text, s, e, left_chars, right_chars)
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

    return rows


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
    Generate legacy preview snippets for checklist items with `preview=True`.

    This helper operates on the base match column only and is retained for
    compatibility with earlier preview workflows.
    """
    import sys

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
# Main extraction pipeline
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
    negation_scope: str = "left",
    return_previews: bool = False,
    use_parallel: bool = False,
    parallel_backend: str | None = None,
    n_workers: int | None = None,
):
    """
    Apply the checklist to the prepared input DataFrame.

    This function:
    - computes base regex matches
    - applies substance and negation gating when configured
    - applies discharge-context and false-positive pruning when configured
    - optionally emits previews for checklist items marked with `preview=True`

    The input note text is assumed to have been normalized upstream.
    """
    negation_scope = (negation_scope or "left").lower()
    if negation_scope not in {"left", "right", "both"}:
        raise ValueError("negation_scope must be 'left', 'right', or 'both'")

    _dbg(
        f"[DEBUG] Starting regex_extract: df_to_analyze.shape={df_to_analyze.shape}, "
        f"metadata.shape={metadata.shape}, expected_row_count={expected_row_count}, "
        f"use_parallel={use_parallel}, parallel_backend={parallel_backend}, n_workers={n_workers}"
    )

    previews_acc: list[dict] = []

    for i in checklist:
        _dbg(f"\n[DEBUG] Checklist item index: {i}")
        actual_rows = df_to_analyze.shape[0]
        _dbg(f"[DEBUG]  → df_to_analyze has {actual_rows} rows")
        assert actual_rows == expected_row_count, f"Row counts do not match ({actual_rows} != {expected_row_count})"

        pat = checklist[i]["pat"]
        col_name = checklist[i]["col_name"]
        _dbg(f"[DEBUG]  → pattern='{pat.pattern if hasattr(pat, 'pattern') else pat}', col_name='{col_name}'")

        has_substance = bool(checklist[i].get("substance") or checklist[i].get("opioid"))
        has_negation = bool(checklist[i].get("negation"))
        _dbg(f"[DEBUG]  → substance={has_substance}, negation={has_negation}")

        _dbg(f"[DEBUG]  → Calling regex_search_file for '{col_name}'")
        df_searched = regex_search_file(
            pat,
            col_name,
            df_to_analyze,
            metadata,
            preview=True,
            use_parallel=use_parallel,
            parallel_backend=parallel_backend,
            n_workers=n_workers,
        )
        base_sum = int(pd.to_numeric(df_searched[col_name], errors="coerce").fillna(0).sum())
        _dbg(f"[DEBUG]  → After regex_search_file: df_searched['{col_name}'].sum()={base_sum}")

        active_col = col_name

        if has_substance:
            _dbg(f"[DEBUG]  → Entering substance branch for '{col_name}'")
            if base_sum > 0:
                df_searched = check_for_substance(
                    pat,
                    col_name,
                    f"{col_name}_SUBSTANCE_MATCHED",
                    df_searched,
                    span=WIN_SUBSTANCE,
                    use_parallel=use_parallel,
                    parallel_backend=parallel_backend,
                    n_workers=n_workers,
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
                            side=negation_scope,
                            use_parallel=use_parallel,
                            parallel_backend=parallel_backend,
                            n_workers=n_workers,
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
                    side=negation_scope,
                    use_parallel=use_parallel,
                    parallel_backend=parallel_backend,
                    n_workers=n_workers,
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

        should_prune = has_negation or not has_substance

        if active_col in df_searched.columns:
            pre_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
            _dbg(f"[DEBUG]    • Before pruning on {active_col}: {pre_sum} kept")

            if should_prune and pre_sum > 0:
                if exclude_discharge_mentions:
                    df_searched = discharge_instructions(
                        pat,
                        df_searched,
                        active_col,
                        span=WIN_DISCHARGE,
                        use_parallel=use_parallel,
                        parallel_backend=parallel_backend,
                        n_workers=n_workers,
                    )
                    post_dis = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
                    _dbg(f"[DEBUG]    • After discharge_instructions on {active_col}: {post_dis} kept")

                common_fp = checklist[i].get("common_fp") or []
                if common_fp:
                    df_searched = check_common_false_positives(
                        pat,
                        df_searched,
                        active_col,
                        common_fp,
                        span=WIN_CFP,
                        use_parallel=use_parallel,
                        parallel_backend=parallel_backend,
                        n_workers=n_workers,
                    )
                    post_fp = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
                    _dbg(f"[DEBUG]    • After common FP pruning on {active_col}: {post_fp} kept")
            else:
                _dbg(
                    f"[DEBUG]    • Skipping pruning for {active_col} "
                    f"(matches={pre_sum}, has_substance={has_substance}, has_negation={has_negation})"
                )
        else:
            _dbg(f"[DEBUG]    • Skipping pruning (missing column {active_col})")

        if active_col not in df_searched.columns:
            df_searched[active_col] = 0

        if checklist[i].get("preview"):
            _dbg(f"[DEBUG]  → Writing previews for '{col_name}' from mask '{active_col}'")
            sub_win = WIN_SUBSTANCE if has_substance else 0
            neg_win = WIN_NEGATION if has_negation else 0
            fp_win = WIN_CFP if checklist[i].get("common_fp") else 0
            dis_win = WIN_DISCHARGE if exclude_discharge_mentions else 0
            left_req = max(preview_span, sub_win, neg_win, fp_win, dis_win)
            right_req = left_req

            rows = write_previews_for_item(
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


def regex_search_file(
    pat,
    new_col_name,
    df_to_search,
    metadata,
    preview=True,
    use_parallel: bool = False,
    parallel_backend: str | None = None,
    n_workers: int | None = None,
):
    """
    Count pattern matches per note and merge the result into the metadata frame.

    The input note text is assumed to have been normalized upstream.
    """
    work = df_to_search[["note_id", "note_text"]].copy()

    pat_payload = _pattern_to_payload(pat)
    count_func = partial(_count_pattern_matches_from_payload, pat_payload=pat_payload)

    work[new_col_name] = _apply_series(
        work["note_text"],
        count_func,
        use_parallel=use_parallel,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
    )

    keep_cols = ["note_id", new_col_name]
    if preview:
        keep_cols.append("note_text")

    df_searched = metadata.merge(work[keep_cols], how="left", on="note_id")
    return df_searched


# ============================================================
# Miscellaneous redactors
# ============================================================


def remove_tobacco_mentions(text):
    """Mask mentions of no tobacco use in the text."""
    tobacco_pattern = (
        r"(Tob(?:acco)?[ -]*(?:use)?[: -]*\b(None|Never|No\s+use|abstains|denies use,? never|ever)\b|"
        r"Smoking[: -]*None|Smoker[: -]*(?:never|no))"
    )
    return re.sub(tobacco_pattern, "Tobacco: [Redacted]", str(text), flags=re.IGNORECASE)


# ============================================================
# Console preview helper
# ============================================================


def preview_string_matches(pat, col_name, df_searched, col_check=False, n_notes=10, span=100):
    """
    Print color-highlighted match excerpts for interactive debugging.
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
            _, _, snippet = _window(text, start, stop, span, span)

            marked = snippet
            marked = (
                marked[0:span]
                + "\x1b[7m"
                + marked[span : span + (stop - start)]
                + "\x1b[0m"
                + marked[span + (stop - start) :]
            )

            for term in TERMS_LIST:
                x = re.search(term, marked, flags=re.IGNORECASE | re.MULTILINE)
                if x:
                    s2, e2 = x.span()
                    marked = marked[:s2] + "\x1b[7m" + marked[s2:e2] + "\x1b[0m" + marked[e2:]

            for cue in ["no ", "not ", "denies", "denial", "doubt", "never", "negative for"]:
                x = re.search(cue, marked, flags=re.IGNORECASE | re.MULTILINE)
                if x:
                    s2, e2 = x.span()
                    marked = marked[:s2] + "\x1b[7m" + marked[s2:e2] + "\x1b[0m" + marked[e2:]

            if PRINT:
                print(marked)
                print("\n")
