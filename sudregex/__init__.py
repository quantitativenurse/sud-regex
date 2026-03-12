# sudregex/__init__.py
"""
Public package interface for sudregex.

This module exposes the main extraction APIs for file-based and in-memory
workflows, along with selected helper utilities at the package root. It also
coordinates shared runtime setup, input normalization, identifier handling,
and parallel backend selection before delegating row-wise extraction work to
the helper layer.
"""
from importlib.metadata import PackageNotFoundError, version
from typing import List, Optional, Tuple, Union

import pandas as pd

from . import helper as _helper
from .checklist import checklist as checklist_abc
from .termslist import termslist as default_termslist
from .validation import import_python_object as _import_python_object
from .validation import validate_checklist as validation

__all__ = [
    "__version__",
    "extract",
    "extract_df",
    "remove_line_break",
    "remove_tobacco_mentions",
    "set_terms",
    "regex_extract",
    "check_for_substance",
    "check_negation",
    "check_common_false_positives",
    "discharge_instructions",
    "preview_string_matches",
    "previews_batch",
    "checklist_abc",
    "default_termslist",
    "validation",
    "write_previews_for_item",
]

# Re-export selected helper functions at the package root for convenience.
remove_line_break = _helper.remove_line_break
remove_tobacco_mentions = _helper.remove_tobacco_mentions
set_terms = _helper.set_terms
regex_extract = _helper.regex_extract
check_for_substance = _helper.check_for_substance
check_negation = _helper.check_negation
check_common_false_positives = _helper.check_common_false_positives
discharge_instructions = _helper.discharge_instructions
preview_string_matches = _helper.preview_string_matches
previews_batch = _helper.previews_batch
write_previews_for_item = _helper.write_previews_for_item

try:
    __version__ = version("sudregex")
except PackageNotFoundError:
    __version__ = "0.0.dev"

# Supported execution backends for row-wise parallel work.
SUPPORTED_PARALLEL_BACKENDS = {"pandarallel", "loky"}


def _build_terms(terms=None, termslist=None, terms_active=None) -> List[str]:
    """
    Build the active vocabulary from either:

    - explicit `terms`, or
    - (`termslist`, `terms_active`) where `termslist` is a module object,
      a dictionary, or a Python file path, and `terms_active` identifies one
      or more groups defined in that source.

    If both grouped terms and explicit `terms` are provided, the explicit terms
    are appended to the resolved group terms.
    """
    out: List[str] = []

    if termslist and terms_active:
        if isinstance(terms_active, str):
            groups = [g.strip() for g in terms_active.split(",") if g.strip()]
        else:
            groups = list(terms_active)

        if isinstance(termslist, dict):
            for grp in groups:
                if grp not in termslist:
                    raise ValueError(f"Term group {grp!r} not found in provided dict.")
                out.extend(termslist[grp])
        else:
            if isinstance(termslist, str):
                import importlib.util
                import os

                module_name = os.path.splitext(os.path.basename(termslist))[0]
                spec = importlib.util.spec_from_file_location(module_name, termslist)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load module from {termslist}")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            else:
                mod = termslist

            for grp in groups:
                if not hasattr(mod, grp):
                    raise ValueError(f"Term group {grp!r} not found in {getattr(mod, '__file__', 'termslist module')}.")
                out.extend(getattr(mod, grp))

        if terms:
            out.extend(list(terms))

    elif terms:
        out = list(terms)
    else:
        raise ValueError("You must supply either (termslist & terms_active) or explicit terms=[...].")

    return out


def _normalize_identifier_args(
    *,
    id_column: Optional[str],
    grid_column: Optional[str],
    person_column: Optional[str],
    extra_id_columns: Optional[List[str]],
) -> tuple[Optional[str], Optional[str], List[str]]:
    """
    Normalize identifier-related arguments.

    - `grid_column` is treated as a deprecated alias for `person_column`
    - `extra_id_columns` are de-duplicated against the primary note and person
      identifier columns
    """
    if grid_column and not person_column:
        person_column = grid_column

    extra_id_columns = list(extra_id_columns or [])
    extra_id_columns = [c for c in extra_id_columns if c not in {id_column, person_column}]
    return id_column, person_column, extra_id_columns


def _resolve_checklist(checklist, keys=None):
    """
    Resolve a checklist object from either a Python object or a file path.

    If `keys` is provided, return only the requested checklist entries that are
    present in the resolved checklist.
    """
    checklist_obj = _import_python_object(checklist, "checklist") if isinstance(checklist, str) else checklist
    if keys:
        checklist_obj = {k: checklist_obj[k] for k in keys if k in checklist_obj}
    return checklist_obj


def _initialize_runtime(
    *,
    checklist,
    terms,
    termslist,
    terms_active,
    debug: bool,
    keys=None,
):
    """
    Initialize shared runtime state for both file-based and DataFrame-based APIs.

    Responsibilities:
    - set helper debug mode
    - resolve the checklist
    - build the active vocabulary
    - register terms with the helper layer
    """
    hlp = _helper
    hlp.PRINT = debug

    checklist_obj = _resolve_checklist(checklist, keys=keys)
    terms_list = _build_terms(terms=terms, termslist=termslist, terms_active=terms_active)
    hlp.set_terms(terms_list)
    return checklist_obj


def _init_parallel(
    *,
    parallel: bool,
    parallel_backend: str = "pandarallel",
    n_workers: Optional[int],
    debug: bool,
) -> tuple[bool, Optional[str], Optional[int]]:
    """
    Validate and initialize parallel execution settings.

    Returns:
        (False, None, None) for serial execution
        (True, backend, n_workers) for parallel execution

    Notes:
    - `pandarallel` requires explicit initialization
    - `loky` uses joblib and does not require global initialization here
    """
    if not parallel:
        return False, None, None

    backend = (parallel_backend or "pandarallel").lower()
    if backend not in SUPPORTED_PARALLEL_BACKENDS:
        raise ValueError(
            f"Unsupported parallel_backend={backend!r}. " f"Choose one of {sorted(SUPPORTED_PARALLEL_BACKENDS)}."
        )

    if n_workers is not None and (not isinstance(n_workers, int) or n_workers <= 0):
        raise ValueError("n_workers must be a positive integer")

    if backend == "pandarallel":
        try:
            from pandarallel import pandarallel  # type: ignore

            if n_workers is not None:
                pandarallel.initialize(progress_bar=False, nb_workers=n_workers)
            else:
                pandarallel.initialize(progress_bar=False)
            return True, "pandarallel", n_workers
        except ImportError:
            if debug:
                print("[DEBUG] pandarallel not installed; falling back to serial mode")
            return False, None, None

    if backend == "loky":
        try:
            import joblib  # noqa: F401

            return True, "loky", n_workers
        except ImportError:
            if debug:
                print("[DEBUG] joblib/loky not installed; falling back to serial mode")
            return False, None, None

    return False, None, None


def _apply_series(
    series: pd.Series,
    func,
    use_parallel: bool,
    parallel_backend: Optional[str],
    n_workers: Optional[int],
) -> pd.Series:
    """
    Apply a function to a Series using the configured execution backend.

    Behavior:
    - serial mode uses pandas `.apply()`
    - `pandarallel` uses `.parallel_apply()` directly for prep-time operations
    - `loky` delegates to the helper-layer backend-aware dispatcher
    """
    if not use_parallel or not parallel_backend:
        return series.apply(func)

    backend = parallel_backend.lower()

    if backend == "pandarallel":
        if hasattr(series, "parallel_apply"):
            return series.parallel_apply(func)
        return series.apply(func)

    if backend == "loky":
        return _helper._apply_series(
            series=series,
            func=func,
            use_parallel=True,
            parallel_backend="loky",
            n_workers=n_workers,
        )

    return series.apply(func)


def _required_columns(
    *,
    note_column: str,
    id_column: Optional[str],
    person_column: Optional[str],
    extra_id_columns: Optional[List[str]],
) -> set[str]:
    """
    Compute the set of required input columns for the current extraction call.
    """
    req = {note_column}
    if id_column:
        req.add(id_column)
    if person_column:
        req.add(person_column)
    req.update([c for c in (extra_id_columns or []) if c])
    return req


def _prepare_work_df(
    *,
    df: pd.DataFrame,
    note_column: str,
    id_column: Optional[str],
    person_column: Optional[str],
    extra_id_columns: Optional[List[str]],
    keep_columns=None,
) -> pd.DataFrame:
    """
    Validate, subset, and normalize the working DataFrame.

    This function:
    - verifies required columns
    - keeps required columns plus any requested passthrough columns
    - coerces identifier/text fields to pandas string dtype
    - drops rows missing note text or, when configured, note identifiers
    """
    req = _required_columns(
        note_column=note_column,
        id_column=id_column,
        person_column=person_column,
        extra_id_columns=extra_id_columns,
    )

    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required cols: {missing}")

    keep = list(req)
    if keep_columns:
        keep += [c for c in keep_columns if c not in keep]

    work = df[keep].copy()
    for c in req:
        if c in work.columns:
            work[c] = work[c].astype("string")

    subset_keys = [note_column] + ([id_column] if id_column else [])
    work.dropna(subset=subset_keys, inplace=True)

    return work


def _build_crosswalk(
    *,
    work: pd.DataFrame,
    id_column: Optional[str],
    person_column: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Build a note-to-person crosswalk for identifier reattachment after extraction.
    """
    cross_cols = [c for c in [id_column, person_column] if c and c in work.columns]
    if not cross_cols:
        return None

    crosswalk = work[cross_cols].drop_duplicates(subset=[id_column] if id_column else None).copy()
    for c in cross_cols:
        crosswalk[c] = crosswalk[c].astype("string")
    return crosswalk


def _group_notes_and_extras(
    *,
    work: pd.DataFrame,
    note_column: str,
    id_column: Optional[str],
    extra_id_columns: Optional[List[str]],
    keep_columns=None,
):
    """
    Group note text by `id_column` and attach the first value of passthrough columns.

    When `id_column` is present, all rows sharing that identifier are combined into
    a single note text value via `" ".join(...)`. Additional identifier or passthrough
    columns are grouped with `.first()` and merged back into the grouped result.
    """
    extras_df = None

    if id_column:
        work[id_column] = work[id_column].astype("string")
        grouped = work.groupby([id_column])[note_column].apply(" ".join).reset_index()
        grouped[id_column] = grouped[id_column].astype("string")
    else:
        passthrough_cols = [note_column] + list(extra_id_columns or [])
        grouped = work[passthrough_cols].copy()

    extras_list = list(extra_id_columns or []) + list(keep_columns or [])
    extras_list = [c for c in extras_list if c not in {note_column, id_column}]

    if id_column and extras_list:
        extras_df = work.groupby([id_column])[extras_list].first().reset_index()
        extras_df[id_column] = extras_df[id_column].astype("string")
        grouped = grouped.merge(extras_df, on=id_column, how="left")

    return grouped, extras_df


def _build_metadata(grouped: pd.DataFrame, id_column: Optional[str]) -> tuple[pd.DataFrame, int]:
    """
    Build the metadata frame used as the base merge target for extracted signals.

    Returns:
    - metadata DataFrame
    - expected row count for downstream consistency checks
    """
    expected = grouped.shape[0]
    meta = (
        grouped[[id_column]].copy() if id_column and id_column in grouped.columns else pd.DataFrame(index=grouped.index)
    )
    if id_column and id_column in meta.columns:
        meta[id_column] = meta[id_column].astype("string")
    return meta, expected


def _rename_for_helper(
    *,
    grouped: pd.DataFrame,
    meta: pd.DataFrame,
    extras_df: Optional[pd.DataFrame],
    id_column: Optional[str],
):
    """
    Normalize the note identifier column name for helper-layer compatibility.

    The helper extraction layer expects the note identifier column to be named
    `note_id`. If the caller uses a different identifier name, rename it here
    and preserve enough state to restore the original name later.
    """
    orig_id_col = id_column
    tmp_renamed = False

    if id_column and id_column != "note_id":
        if id_column in grouped.columns:
            grouped = grouped.rename(columns={id_column: "note_id"})
            tmp_renamed = True
        if id_column in meta.columns:
            meta = meta.rename(columns={id_column: "note_id"})
        if extras_df is not None and id_column in extras_df.columns:
            extras_df = extras_df.rename(columns={id_column: "note_id"})

    return grouped, meta, extras_df, orig_id_col, tmp_renamed


def _prepare_grouped_input(
    *,
    df: pd.DataFrame,
    note_column: str,
    id_column: Optional[str],
    person_column: Optional[str],
    extra_id_columns: Optional[List[str]],
    keep_columns=None,
    remove_linebreaks: bool,
    use_parallel: bool,
    parallel_backend: Optional[str],
    n_workers: Optional[int],
):
    """
    Prepare grouped input for extraction.

    This shared path is used by both `extract()` and `extract_df()` and is
    responsible for:
    - validating input columns
    - building identifier crosswalks
    - optionally normalizing note text
    - grouping rows by note identifier
    - constructing helper-compatible metadata
    """
    work = _prepare_work_df(
        df=df,
        note_column=note_column,
        id_column=id_column,
        person_column=person_column,
        extra_id_columns=extra_id_columns,
        keep_columns=keep_columns,
    )

    if person_column and not id_column:
        raise ValueError("person_column requires id_column to be set for crosswalk reattachment.")

    crosswalk = _build_crosswalk(
        work=work,
        id_column=id_column,
        person_column=person_column,
    )

    if remove_linebreaks:
        work[note_column] = _apply_series(
            work[note_column],
            _helper.remove_line_break,
            use_parallel=use_parallel,
            parallel_backend=parallel_backend,
            n_workers=n_workers,
        )

    grouped, extras_df = _group_notes_and_extras(
        work=work,
        note_column=note_column,
        id_column=id_column,
        extra_id_columns=extra_id_columns,
        keep_columns=keep_columns,
    )

    meta, expected = _build_metadata(grouped, id_column)

    grouped, meta, extras_df, orig_id_col, tmp_renamed = _rename_for_helper(
        grouped=grouped,
        meta=meta,
        extras_df=extras_df,
        id_column=id_column,
    )

    return {
        "grouped": grouped,
        "meta": meta,
        "expected": expected,
        "crosswalk": crosswalk,
        "extras_df": extras_df,
        "orig_id_col": orig_id_col,
        "tmp_renamed": tmp_renamed,
    }


def _run_grouped_extraction(
    *,
    checklist_obj,
    grouped: pd.DataFrame,
    meta: pd.DataFrame,
    expected: int,
    exclude_discharge_mentions: bool,
    preview_count: int,
    preview_span: int,
    preview_csv: Optional[str],
    preview_file: Optional[str],
    negation_scope: str,
    return_previews_df: bool,
    use_parallel: bool,
    parallel_backend: Optional[str],
    n_workers: Optional[int],
):
    """
    Execute helper-layer extraction on prepared grouped input.

    This wrapper passes the resolved execution mode and preview configuration
    into `helper.regex_extract()` and optionally returns the preview DataFrame.
    """
    if return_previews_df:
        return _helper.regex_extract(
            checklist=checklist_obj,
            df_to_analyze=grouped,
            metadata=meta,
            preview_count=preview_count,
            expected_row_count=expected,
            exclude_discharge_mentions=exclude_discharge_mentions,
            preview_span=preview_span,
            preview_csv=preview_csv,
            preview_file=preview_file or "note_previews.txt",
            negation_scope=negation_scope,
            return_previews=True,
            use_parallel=use_parallel,
            parallel_backend=parallel_backend,
            n_workers=n_workers,
        )

    res = _helper.regex_extract(
        checklist=checklist_obj,
        df_to_analyze=grouped,
        metadata=meta,
        preview_count=preview_count,
        expected_row_count=expected,
        exclude_discharge_mentions=exclude_discharge_mentions,
        preview_span=preview_span,
        preview_csv=preview_csv,
        preview_file=preview_file or "note_previews.txt",
        negation_scope=negation_scope,
        use_parallel=use_parallel,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
    )
    return res, None


def _finalize_result(
    *,
    res: pd.DataFrame,
    id_column: Optional[str],
    orig_id_col: Optional[str],
    tmp_renamed: bool,
    crosswalk: Optional[pd.DataFrame],
    person_column: Optional[str],
    extra_id_columns: Optional[List[str]],
    include_note_text: bool,
    note_column: str,
):
    """
    Finalize the extracted result DataFrame for public return/output.

    This step restores caller-facing identifier names, reattaches person-level
    identifiers when needed, drops note text when not requested, and orders
    identifier columns at the front of the result.
    """
    if tmp_renamed and "note_id" in res.columns and orig_id_col:
        res = res.rename(columns={"note_id": orig_id_col})

    if id_column and id_column in res.columns:
        res[id_column] = res[id_column].astype("string")

    if id_column and crosswalk is not None:
        res = res.merge(crosswalk.drop_duplicates(subset=[id_column]), on=id_column, how="left")

    if not include_note_text and note_column in res.columns:
        res = res.drop(columns=[note_column])

    id_front = [c for c in [person_column, id_column, *(extra_id_columns or [])] if c and c in res.columns]
    other_cols = [c for c in res.columns if c not in id_front]
    res = res[id_front + other_cols]

    return res


def _finalize_previews(
    *,
    previews_df: Optional[pd.DataFrame],
    crosswalk: Optional[pd.DataFrame],
    tmp_renamed: bool,
    orig_id_col: Optional[str],
):
    """
    Finalize preview output by reattaching caller-facing identifiers when needed.
    """
    if previews_df is None or previews_df.empty:
        return previews_df

    if "note_id" not in previews_df.columns or crosswalk is None:
        return previews_df

    previews_df = previews_df.copy()
    previews_df["note_id"] = previews_df["note_id"].astype("string")

    if tmp_renamed and orig_id_col:
        previews_df = previews_df.merge(crosswalk, left_on="note_id", right_on=orig_id_col, how="left")
        previews_df.drop(columns=[orig_id_col], inplace=True, errors="ignore")
    else:
        previews_df = previews_df.merge(crosswalk, on="note_id", how="left")

    return previews_df


def extract(
    in_file,
    out_file,
    checklist,
    separator="",
    terms=None,
    termslist=None,
    terms_active=None,
    parallel=False,
    parallel_backend: str = "pandarallel",
    include_note_text=False,
    nrows=None,
    chunk_size=None,
    remove_linebreaks=True,
    note_column="note_text",
    id_column="note_id",
    grid_column="grid",
    person_column: Optional[str] = None,
    extra_id_columns: Optional[List[str]] = None,
    keep_columns=None,
    debug: bool = False,
    has_header: bool = True,
    no_header_columns: Optional[List[str]] = None,
    n_workers: Optional[int] = None,
    exclude_discharge_mentions: bool = True,
    preview_count: int = 0,
    preview_span: int = 120,
    preview_file: Optional[str] = None,
    preview_csv: Optional[str] = None,
    negation_scope: str = "left",
):
    """
    Run regex extraction on a file and write the result to CSV.

    Output behavior:
    - if a single batch is produced, write directly to `out_file`
    - if multiple batches are produced, write to numbered part files
    """
    import os
    import time

    def _part_filename(path: str, idx: int) -> str:
        base, ext = os.path.splitext(path)
        ext = ext or ".csv"
        return f"{base}_part_{idx}{ext}"

    id_column, person_column, extra_id_columns = _normalize_identifier_args(
        id_column=id_column,
        grid_column=grid_column,
        person_column=person_column,
        extra_id_columns=extra_id_columns,
    )

    checklist_obj = _initialize_runtime(
        checklist=checklist,
        terms=terms,
        termslist=termslist,
        terms_active=terms_active,
        debug=debug,
    )

    use_parallel, resolved_parallel_backend, resolved_n_workers = _init_parallel(
        parallel=parallel,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
        debug=debug,
    )

    nrows = int(nrows) if nrows is not None else None
    chunk_size = int(chunk_size) if chunk_size is not None else int(1e6)

    read_kwargs = dict(sep=separator, engine="python", nrows=nrows, chunksize=chunk_size)
    if not has_header:
        if no_header_columns:
            names = [c.strip() for c in no_header_columns if c and c.strip()]
        else:
            names = [c for c in (id_column, person_column, note_column) if c]
        read_kwargs.update(header=None, names=names)

    dtype_map = {}
    for c in [person_column, id_column, note_column, *extra_id_columns]:
        if c:
            dtype_map[c] = "string"
    if not has_header and no_header_columns:
        for c in no_header_columns:
            if c:
                dtype_map[c] = "string"
    if dtype_map:
        read_kwargs.update(dtype=dtype_map)

    start = time.time()

    first_result = None
    batch_count = 0
    multi_output_started = False

    for chunk in pd.read_csv(in_file, **read_kwargs):
        prepared = _prepare_grouped_input(
            df=chunk,
            note_column=note_column,
            id_column=id_column,
            person_column=person_column,
            extra_id_columns=extra_id_columns,
            keep_columns=keep_columns,
            remove_linebreaks=remove_linebreaks,
            use_parallel=use_parallel,
            parallel_backend=resolved_parallel_backend,
            n_workers=resolved_n_workers,
        )

        res, _ = _run_grouped_extraction(
            checklist_obj=checklist_obj,
            grouped=prepared["grouped"],
            meta=prepared["meta"],
            expected=prepared["expected"],
            exclude_discharge_mentions=exclude_discharge_mentions,
            preview_count=preview_count,
            preview_span=preview_span,
            preview_csv=preview_csv,
            preview_file=preview_file,
            negation_scope=negation_scope,
            return_previews_df=False,
            use_parallel=use_parallel,
            parallel_backend=resolved_parallel_backend,
            n_workers=resolved_n_workers,
        )

        result = _finalize_result(
            res=res,
            id_column=id_column,
            orig_id_col=prepared["orig_id_col"],
            tmp_renamed=prepared["tmp_renamed"],
            crosswalk=prepared["crosswalk"],
            person_column=person_column,
            extra_id_columns=extra_id_columns,
            include_note_text=include_note_text,
            note_column=note_column,
        )

        if batch_count == 0:
            first_result = result
            batch_count = 1
            continue

        if not multi_output_started:
            first_result.to_csv(_part_filename(out_file, 0), index=False)
            multi_output_started = True

        result.to_csv(_part_filename(out_file, batch_count), index=False)
        batch_count += 1

    if batch_count == 0:
        pd.DataFrame().to_csv(out_file, index=False)
    elif batch_count == 1:
        first_result.to_csv(out_file, index=False)

    elapsed = round(time.time() - start, 2)
    print(f"Elapsed time: {elapsed}s")
    return True


def extract_df(
    df,
    checklist,
    note_column="note_text",
    terms=None,
    termslist=None,
    terms_active=None,
    remove_linebreaks=True,
    keys=None,
    parallel=False,
    parallel_backend: str = "pandarallel",
    debug: bool = False,
    id_column="note_id",
    grid_column=None,
    person_column: Optional[str] = None,
    extra_id_columns: Optional[List[str]] = None,
    include_note_text: bool = False,
    n_workers: Optional[int] = None,
    exclude_discharge_mentions: bool = True,
    preview_count: int = 0,
    preview_span: int = 120,
    preview_file: Optional[str] = None,
    preview_csv: Optional[str] = None,
    negation_scope: str = "left",
    return_previews_df: bool = False,
) -> Union["pd.DataFrame", Tuple["pd.DataFrame", "pd.DataFrame"]]:
    """
    Run regex extraction on an in-memory DataFrame.

    This function shares the same preparation, extraction, and finalization
    pipeline as `extract()`, so notebook and file-based workflows remain
    behaviorally aligned.
    """
    checklist_obj = _initialize_runtime(
        checklist=checklist,
        terms=terms,
        termslist=termslist,
        terms_active=terms_active,
        debug=debug,
        keys=keys,
    )

    id_column, person_column, extra_id_columns = _normalize_identifier_args(
        id_column=id_column,
        grid_column=grid_column,
        person_column=person_column,
        extra_id_columns=extra_id_columns,
    )

    use_parallel, resolved_parallel_backend, resolved_n_workers = _init_parallel(
        parallel=parallel,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
        debug=debug,
    )

    prepared = _prepare_grouped_input(
        df=df,
        note_column=note_column,
        id_column=id_column,
        person_column=person_column,
        extra_id_columns=extra_id_columns,
        keep_columns=None,
        remove_linebreaks=remove_linebreaks,
        use_parallel=use_parallel,
        parallel_backend=resolved_parallel_backend,
        n_workers=resolved_n_workers,
    )

    res, previews_df = _run_grouped_extraction(
        checklist_obj=checklist_obj,
        grouped=prepared["grouped"],
        meta=prepared["meta"],
        expected=prepared["expected"],
        exclude_discharge_mentions=exclude_discharge_mentions,
        preview_count=preview_count,
        preview_span=preview_span,
        preview_csv=preview_csv,
        preview_file=preview_file,
        negation_scope=negation_scope,
        return_previews_df=return_previews_df,
        use_parallel=use_parallel,
        parallel_backend=resolved_parallel_backend,
        n_workers=resolved_n_workers,
    )

    result = _finalize_result(
        res=res,
        id_column=id_column,
        orig_id_col=prepared["orig_id_col"],
        tmp_renamed=prepared["tmp_renamed"],
        crosswalk=prepared["crosswalk"],
        person_column=person_column,
        extra_id_columns=extra_id_columns,
        include_note_text=include_note_text,
        note_column=note_column,
    )

    if return_previews_df:
        previews_df = _finalize_previews(
            previews_df=previews_df,
            crosswalk=prepared["crosswalk"],
            tmp_renamed=prepared["tmp_renamed"],
            orig_id_col=prepared["orig_id_col"],
        )
        return result, previews_df

    return result
