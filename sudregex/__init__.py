# sudregex/__init__.py
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

# re-export useful helper functions
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


def _build_terms(terms=None, termslist=None, terms_active=None) -> List[str]:
    """
    Build the active terms list from either:
      • explicit `terms`, or
      • (`termslist`, `terms_active`) where `termslist` is a module object, a dict, or a file path,
        and `terms_active` is a group name or comma-separated list of group names defined in that module/dict.
    """
    out: List[str] = []
    if termslist and terms_active:
        # normalize active groups
        if isinstance(terms_active, str):
            groups = [g.strip() for g in terms_active.split(",") if g.strip()]
        else:
            groups = list(terms_active)

        if isinstance(termslist, dict):
            # termslist is already a dict of lists
            for grp in groups:
                if grp not in termslist:
                    raise ValueError(f"Term group {grp!r} not found in provided dict.")
                out.extend(termslist[grp])

        else:
            # termslist is a module object OR a file path; load safely with importlib if it's a path
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
                mod = termslist  # already a module object

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


def extract(
    in_file,
    out_file,
    checklist,
    separator="",
    terms=None,
    termslist=None,
    terms_active=None,
    parallel=False,
    include_note_text=False,
    nrows=None,
    chunk_size=None,
    remove_linebreaks=True,
    note_column="note_text",  # text column
    id_column="note_id",  # NOTE/document identifier (can be overridden)
    grid_column="grid",  # DEPRECATED alias for person identifier
    person_column: Optional[str] = None,  # person identifier (wins over grid_column)
    extra_id_columns: Optional[List[str]] = None,  # additional identifiers to carry through
    keep_columns=None,
    debug: bool = False,
    has_header: bool = True,  # header present by default
    no_header_columns: Optional[List[str]] = None,  # if no header, user-provided names in file order
    n_workers: Optional[int] = None,  # optional pandarallel worker count
    exclude_discharge_mentions: bool = True,  # default behavior
    preview_count: int = 0,  # 0 = previews off
    preview_span: int = 120,
    preview_file: Optional[str] = None,  # human-readable previews
    preview_csv: Optional[str] = None,  # CSV previews path
    negation_scope: str = "left",  # "left" | "right" | "both"
):
    """
    Run regex extraction on a file and save results to CSV.

    Identifiers:
      - person_column: replaces deprecated grid_column (if both provided, person_column wins)
      - id_column: note/document identifier (free name; a shim maps to 'note_id' for helper)
      - extra_id_columns: any additional identifier columns to carry through

    Header handling:
      - has_header=True (default): read column names from file header.
      - has_header=False: if no_header_columns is provided, use exactly those names in file order;
        otherwise fallback assumes file order is [id_column, person_column, note_column].
    """
    import time

    hlp = _helper
    hlp.PRINT = debug

    # Deprecation shim: grid_column → person_column
    if grid_column and not person_column:
        person_column = grid_column

    # sanitize extra ids
    extra_id_columns = list(extra_id_columns or [])
    extra_id_columns = [c for c in extra_id_columns if c not in {id_column, person_column}]

    # -- load checklist object --
    checklist_obj = _import_python_object(checklist, "checklist") if isinstance(checklist, str) else checklist

    # -- build terms and set globally for helper --
    terms_list = _build_terms(terms=terms, termslist=termslist, terms_active=terms_active)
    hlp.set_terms(terms_list)

    # parse numerics / defaults
    nrows = int(nrows) if nrows is not None else None
    chunk_size = int(chunk_size) if chunk_size is not None else int(1e6)

    # optional parallel init
    use_parallel = False
    if parallel:
        try:
            from pandarallel import pandarallel  # type: ignore

            if n_workers is not None:
                if not isinstance(n_workers, int) or n_workers <= 0:
                    raise ValueError("n_workers must be a positive integer")
                pandarallel.initialize(progress_bar=False, nb_workers=n_workers)
            else:
                pandarallel.initialize(progress_bar=False)
            use_parallel = True
        except ImportError:
            if debug:
                print("[DEBUG] pandarallel not installed; falling back to single-core .apply()")
            use_parallel = False

    # ---- build read_csv kwargs (header/no-header aware) ----
    read_kwargs = dict(sep=separator, engine="python", nrows=nrows, chunksize=chunk_size)
    if not has_header:
        if no_header_columns:
            names = [c.strip() for c in no_header_columns if c and c.strip()]
        else:
            # Fallback guess: [id, person, note]
            names = [c for c in (id_column, person_column, note_column) if c]
        read_kwargs.update(header=None, names=names)

    # dtype map (ensure identifiers/text are strings)
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
    part = 0

    for chunk in pd.read_csv(in_file, **read_kwargs):
        # ---- validation of required cols ----
        req = {note_column}
        if id_column:
            req.add(id_column)
        if person_column:
            req.add(person_column)
        req.update({c for c in extra_id_columns if c})
        missing = req - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing required cols: {missing}")

        # normalize dtypes
        for col in req:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("string")

        # subset + drop NAs (note text must exist; id if provided)
        keep = list(req)
        if keep_columns:
            keep += [c for c in keep_columns if c not in keep]
        chunk = chunk[keep]
        subset_keys = [note_column] + ([id_column] if id_column else [])
        chunk.dropna(subset=subset_keys, inplace=True)

        # crosswalk (id ↔ person)
        cross_cols = [c for c in [id_column, person_column] if c and c in chunk.columns]
        crosswalk = chunk[cross_cols].drop_duplicates(subset=[id_column] if id_column else None).copy()
        for c in cross_cols:
            crosswalk[c] = crosswalk[c].astype("string")

        # text cleanup
        if remove_linebreaks:
            if use_parallel and hasattr(chunk[note_column], "parallel_apply"):
                chunk[note_column] = chunk[note_column].parallel_apply(hlp.remove_line_break)
            else:
                chunk[note_column] = chunk[note_column].apply(hlp.remove_line_break)

        # group notes by id (or pass through if no id)
        if id_column:
            chunk[id_column] = chunk[id_column].astype("string")
            grouped = chunk.groupby([id_column])[note_column].apply(" ".join).reset_index()
            grouped[id_column] = grouped[id_column].astype("string")
        else:
            grouped = chunk[[note_column] + extra_id_columns].copy()

        # bring extras back (first per id)
        if id_column and (extra_id_columns or keep_columns):
            extras_list = list(extra_id_columns or []) + list(keep_columns or [])
            extras_list = [c for c in extras_list if c not in {note_column, id_column}]
            if extras_list:
                extras = chunk.groupby([id_column])[extras_list].first().reset_index()
                extras[id_column] = extras[id_column].astype("string")
                grouped = grouped.merge(extras, on=id_column, how="left")

        EXPECTED = grouped.shape[0]
        meta = (
            grouped[[id_column]].copy()
            if id_column and id_column in grouped.columns
            else pd.DataFrame(index=grouped.index)
        )
        if id_column and id_column in meta.columns:
            meta[id_column] = meta[id_column].astype("string")

        out_fname = out_file if chunk_size == 1 else out_file.replace(".csv", f"_part_{part}.csv")

        # ---- helper compatibility: ensure 'note_id' exists ----
        orig_id_col = id_column
        tmp_renamed = False
        if id_column and id_column != "note_id":
            if id_column in grouped.columns:
                grouped = grouped.rename(columns={id_column: "note_id"})
                tmp_renamed = True
            if id_column in meta.columns:
                meta = meta.rename(columns={id_column: "note_id"})

        # ---- run extraction (helper handles previews & negation scope) ----
        result = hlp.regex_extract(
            checklist=checklist_obj,
            df_to_analyze=grouped,
            metadata=meta,
            preview_count=preview_count,
            expected_row_count=EXPECTED,
            exclude_discharge_mentions=exclude_discharge_mentions,
            preview_span=preview_span,
            preview_csv=preview_csv,
            preview_file=preview_file or "note_previews.txt",
            negation_scope=negation_scope,
        )

        # rename id back if we mapped to 'note_id'
        if tmp_renamed and "note_id" in result.columns:
            result = result.rename(columns={"note_id": orig_id_col})

        # ensure merge key dtype
        if id_column and id_column in result.columns:
            result[id_column] = result[id_column].astype("string")

        # reattach person id via crosswalk
        if id_column and cross_cols:
            result = result.merge(crosswalk.drop_duplicates(subset=[id_column]), on=id_column, how="left")

        # drop raw text unless requested
        if not include_note_text and note_column in result.columns:
            result.drop(columns=[note_column], inplace=True)

        # reorder: identifiers first (person, id, extras)
        id_front = [c for c in [person_column, id_column, *extra_id_columns] if c and c in result.columns]
        other_cols = [c for c in result.columns if c not in id_front]
        result = result[id_front + other_cols]

        result.to_csv(out_fname, index=False)
        part += 1

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
    import pandas as pd

    import sudregex.helper as hlp

    hlp.PRINT = debug

    # grid_column -> person_column shim
    if grid_column and not person_column:
        person_column = grid_column

    # --- enforce terms and set globally for helper (raises if neither provided) ---
    terms_list = _build_terms(terms=terms, termslist=termslist, terms_active=terms_active)
    hlp.set_terms(terms_list)

    # sanitize extras
    extra_id_columns = list(extra_id_columns or [])
    extra_id_columns = [c for c in extra_id_columns if c not in {id_column, person_column}]

    # --- resolve checklist ---
    checklist_obj = _import_python_object(checklist, "checklist") if isinstance(checklist, str) else checklist
    if keys:
        checklist_obj = {k: checklist_obj[k] for k in keys if k in checklist_obj}

    # --- normalize & validate required columns ---
    req = {note_column}
    if id_column:
        req.add(id_column)
    if person_column:
        req.add(person_column)
    req.update(extra_id_columns)

    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required cols: {missing}")

    work = df[list(req)].copy()
    for c in req:
        work[c] = work[c].astype("string")
    work.dropna(subset=[note_column] + ([id_column] if id_column else []), inplace=True)

    if person_column and not id_column:
        raise ValueError("person_column requires id_column to be set for crosswalk reattachment.")

    # --- crosswalk BEFORE extraction (person ↔ id) ---
    crosswalk = None
    cross_cols = [c for c in [id_column, person_column] if c]
    if cross_cols:
        crosswalk = work[cross_cols].drop_duplicates(subset=[id_column] if id_column else None).copy()
        for c in cross_cols:
            crosswalk[c] = crosswalk[c].astype("string")

    # --- optional parallel init ---
    use_parallel = False
    if parallel:
        try:
            from pandarallel import pandarallel  # type: ignore

            if n_workers is not None:
                if not isinstance(n_workers, int) or n_workers <= 0:
                    raise ValueError("n_workers must be a positive integer")
                pandarallel.initialize(progress_bar=False, nb_workers=n_workers)
            else:
                pandarallel.initialize(progress_bar=False)
            use_parallel = True
        except ImportError:
            if debug:
                print("pandarallel not installed; falling back to .apply()")
            use_parallel = False

    def _apply(series, func):
        if use_parallel and hasattr(series, "parallel_apply"):
            return series.parallel_apply(func)
        return series.apply(func)

    # --- text preprocessing ---
    if remove_linebreaks:
        work[note_column] = _apply(work[note_column], hlp.remove_line_break)

    # --- group by id (like extract()) ---
    if id_column:
        grouped = work.groupby([id_column])[note_column].apply(" ".join).reset_index()
    else:
        grouped = work[[note_column]].copy()

    # capture extras to re-attach AFTER extraction
    extras_df = None
    if id_column and extra_id_columns:
        extras_df = work.groupby([id_column])[extra_id_columns].first().reset_index()
        # (optional) also merge into grouped so previews could see them upstream
        grouped = grouped.merge(extras_df, on=id_column, how="left")

    EXPECTED = grouped.shape[0]
    meta = (
        grouped[[id_column]].copy() if id_column and id_column in grouped.columns else pd.DataFrame(index=grouped.index)
    )
    if id_column and id_column in meta.columns:
        meta[id_column] = meta[id_column].astype("string")

    # --- helper compatibility: temporarily map id → 'note_id' ---
    orig_id_col = id_column
    tmp_renamed = False
    if id_column and id_column != "note_id":
        if id_column in grouped.columns:
            grouped = grouped.rename(columns={id_column: "note_id"})
            tmp_renamed = True
        if id_column in meta.columns:
            meta = meta.rename(columns={id_column: "note_id"})
        # also align extras_df's key if we attached it to grouped earlier
        if extras_df is not None and id_column in extras_df.columns:
            extras_df = extras_df.rename(columns={id_column: "note_id"})

    # --- extraction ---
    if return_previews_df:
        res, previews_df = hlp.regex_extract(
            checklist=checklist_obj,
            df_to_analyze=grouped,
            metadata=meta,
            preview_count=preview_count,
            expected_row_count=EXPECTED,
            exclude_discharge_mentions=exclude_discharge_mentions,
            preview_span=preview_span,
            preview_csv=preview_csv,
            preview_file=preview_file or "note_previews.txt",
            negation_scope=negation_scope,
            return_previews=True,
        )
    else:
        res = hlp.regex_extract(
            checklist=checklist_obj,
            df_to_analyze=grouped,
            metadata=meta,
            preview_count=preview_count,
            expected_row_count=EXPECTED,
            exclude_discharge_mentions=exclude_discharge_mentions,
            preview_span=preview_span,
            preview_csv=preview_csv,
            preview_file=preview_file or "note_previews.txt",
            negation_scope=negation_scope,
        )

    # --- restore original id column name ---
    if tmp_renamed and "note_id" in res.columns:
        res = res.rename(columns={"note_id": orig_id_col})
    # ensure dtype
    if id_column and id_column in res.columns:
        res[id_column] = res[id_column].astype("string")

    # --- reattach person id ---
    if id_column and crosswalk is not None:
        res = res.merge(crosswalk.drop_duplicates(subset=[id_column]), on=id_column, how="left")

    # --- reattach EXTRA IDs ---
    if id_column and extras_df is not None:
        # if we renamed, extras_df currently keyed on 'note_id' — align to current result key
        if tmp_renamed and "note_id" in extras_df.columns:
            extras_df = extras_df.rename(columns={"note_id": id_column})
        res = res.merge(extras_df, on=id_column, how="left")

    # --- attach person to previews if requested ---
    if return_previews_df and "note_id" in locals().get("previews_df", pd.DataFrame()).columns:
        previews_df["note_id"] = previews_df["note_id"].astype("string")
        if tmp_renamed and orig_id_col:
            # previews_df currently has 'note_id'; add person via crosswalk, then drop tmp id if desired
            previews_df = previews_df.merge(crosswalk, left_on="note_id", right_on=orig_id_col, how="left")
            previews_df.drop(columns=[orig_id_col], inplace=True, errors="ignore")
        else:
            previews_df = previews_df.merge(crosswalk, on="note_id", how="left")

    # --- drop raw note text unless requested ---
    if not include_note_text and note_column in res.columns:
        res.drop(columns=[note_column], inplace=True)

    # --- reorder: identifiers first (person, id, extras) ---
    id_front = [c for c in [person_column, id_column, *extra_id_columns] if c and c in res.columns]
    other_cols = [c for c in res.columns if c not in id_front]
    res = res[id_front + other_cols]

    return (res, previews_df) if return_previews_df else res
