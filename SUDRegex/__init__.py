# SUDRegex/__init__.py

__version__ = "0.1.0"

import importlib.util
import os

from .checklist import checklist as checklist_abc
from .helper import (
    check_common_false_positives,
    check_for_substance,
    check_negation,
    discharge_instructions,
    preview_string_matches,
    regex_extract,
    remove_line_break,
    remove_tobacco_mentions,
    set_terms,
)
from .termslist import termslist as default_termslist
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
    "checklist_abc",
    "default_termslist",
    "validation",
]


def _import_python_object(file_path, varname):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if varname and not hasattr(module, varname):
        raise AttributeError(f"{file_path} has no attribute '{varname}'")
    return getattr(module, varname) if varname else module


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
    note_column="note_text",
    id_column="note_id",
    grid_column="grid",
    keep_columns=None,
    debug: bool = False,
    has_header: bool = True,
):
    """
    Run regex extraction and save to CSV.
    """
    import time

    import pandas as pd

    import SUDRegex.helper as hlp

    hlp.PRINT = debug

    # -- load the checklist object --
    if isinstance(checklist, str):
        checklist_obj = _import_python_object(checklist, "checklist")
    else:
        checklist_obj = checklist

    # -- build the combined terms list --
    terms_list = []
    if termslist and terms_active:
        groups = [g.strip() for g in terms_active.split(",")] if isinstance(terms_active, str) else list(terms_active)
        if isinstance(termslist, dict):
            for grp in groups:
                if grp not in termslist:
                    raise ValueError(f"Term group {grp} not found")
                terms_list.extend(termslist[grp])
        else:
            mod = _import_python_object(termslist, None)
            for grp in groups:
                if not hasattr(mod, grp):
                    raise ValueError(f"Term group {grp} not found in {termslist}")
                terms_list.extend(getattr(mod, grp))
        if terms:
            terms_list.extend(list(terms))
    elif terms:
        terms_list = list(terms)
    else:
        raise ValueError("You must supply either termslist & terms_active, or terms")

    hlp.set_terms(terms_list)

    # -- parse optional numeric args --
    nrows = int(nrows) if nrows is not None else None
    chunk_size = int(chunk_size) if chunk_size is not None else int(1e6)

    # -- optional parallel init --
    use_parallel = False
    if parallel:
        try:
            from pandarallel import pandarallel

            pandarallel.initialize(progress_bar=False)
            use_parallel = True
        except ImportError:
            pass

    # ---- build read_csv kwargs based on header presence ----
    read_kwargs = dict(
        sep=separator,
        engine="python",
        nrows=nrows,
        chunksize=chunk_size,
    )
    if not has_header:
        names = [c for c in (grid_column, id_column, note_column) if c]
        read_kwargs.update(header=None, names=names)

    # Force text dtypes for key/text columns to avoid merge dtype issues
    dtype_map = {}
    if grid_column:
        dtype_map[grid_column] = "string"
    if id_column:
        dtype_map[id_column] = "string"
    if note_column:
        dtype_map[note_column] = "string"
    if dtype_map:
        read_kwargs.update(dtype=dtype_map)

    start = time.time()
    part = 0

    for chunk in pd.read_csv(in_file, **read_kwargs):
        # ensure required columns
        req = {note_column, id_column}
        if grid_column:
            req.add(grid_column)
        missing = req - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing required cols: {missing}")

        # normalize dtypes (string) to be safe
        for col in req:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("string")

        # subset and drop NAs (note_id and note_text must exist)
        keep = list(req)
        if keep_columns:
            keep += [c for c in keep_columns if c not in keep]
        chunk = chunk[keep]
        chunk.dropna(subset=[note_column, id_column], inplace=True)

        # save crosswalk
        crosswalk_cols = [c for c in (grid_column, id_column) if c in chunk.columns]
        crosswalk = chunk[crosswalk_cols].copy()
        for c in crosswalk_cols:
            crosswalk[c] = crosswalk[c].astype("string")

        # line-break removal
        if remove_linebreaks:
            if use_parallel:
                chunk[note_column] = chunk[note_column].parallel_apply(hlp.remove_line_break)
            else:
                chunk[note_column] = chunk[note_column].apply(hlp.remove_line_break)

        # group texts by note_id
        # ensure id is string before grouping
        chunk[id_column] = chunk[id_column].astype("string")
        grouped = chunk.groupby([id_column])[note_column].apply(" ".join).reset_index()
        grouped[id_column] = grouped[id_column].astype("string")

        # bring extras back
        if keep_columns:
            extras = chunk.groupby([id_column])[keep_columns].first().reset_index()
            extras[id_column] = extras[id_column].astype("string")
            grouped = grouped.merge(extras, on=id_column, how="left")

        EXPECTED = grouped.shape[0]
        meta = grouped[[id_column]].copy()
        meta[id_column] = meta[id_column].astype("string")

        # chunked filename
        out_fname = out_file if chunk_size == 1 else out_file.replace(".csv", f"_part_{part}.csv")

        # do the regex extraction
        result = hlp.regex_extract(
            checklist=checklist_obj,
            df_to_analyze=grouped,
            metadata=meta,
            preview_count=0,
            expected_row_count=EXPECTED,
        )

        # ensure consistent dtype on merge key
        if id_column in result.columns:
            result[id_column] = result[id_column].astype("string")

        # re-merge grid/note_id crosswalk
        if id_column in result.columns:
            result = result.merge(
                crosswalk.drop_duplicates(id_column),
                on=id_column,
                how="left",
            )

        # drop the raw note text if the user didn’t ask for it
        if not include_note_text and note_column in result.columns:
            result.drop(columns=[note_column], inplace=True)

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
    include_note_text: bool = False,
):
    import pandas as pd

    import SUDRegex.helper as hlp

    hlp.PRINT = debug

    checklist_obj = _import_python_object(checklist, "checklist") if isinstance(checklist, str) else checklist
    if keys:
        checklist_obj = {k: checklist_obj[k] for k in keys if k in checklist_obj}

    # --- terms (unchanged) ---
    terms_list = []
    if termslist and terms_active:
        groups = [g.strip() for g in terms_active.split(",")] if isinstance(terms_active, str) else list(terms_active)
        if isinstance(termslist, dict):
            for grp in groups:
                if grp not in termslist:
                    raise ValueError(f"Term group {grp} not found")
                terms_list.extend(termslist[grp])
        else:
            mod = _import_python_object(termslist, None)
            for grp in groups:
                if not hasattr(mod, grp):
                    raise ValueError(f"Term group {grp} not found in {termslist}")
                terms_list.extend(getattr(mod, grp))
        if terms:
            terms_list.extend(list(terms))
    elif terms:
        terms_list = list(terms)
    else:
        raise ValueError("You must supply either termslist & terms_active, or terms")

    hlp.set_terms(terms_list)

    # --- normalize & validate ---
    req = {note_column}
    if id_column:
        req.add(id_column)
    if grid_column:
        req.add(grid_column)
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required cols: {missing}")

    work = df[list(req)].copy()
    for c in req:
        work[c] = work[c].astype("string")
    work.dropna(subset=[note_column] + ([id_column] if id_column else []), inplace=True)

    # --- crosswalk BEFORE extraction (key step) ---
    crosswalk = None
    if grid_column:
        crosswalk = work[[id_column, grid_column]].drop_duplicates(id_column).copy()

    # --- process text ALWAYS ---
    if remove_linebreaks:
        try:
            if parallel:
                from pandarallel import pandarallel

                pandarallel.initialize(progress_bar=False)
                work[note_column] = work[note_column].parallel_apply(hlp.remove_line_break)
            else:
                work[note_column] = work[note_column].apply(hlp.remove_line_break)
        except ImportError:
            work[note_column] = work[note_column].apply(hlp.remove_line_break)

    # --- group by note_id like extract() ---
    if id_column:
        grouped = work.groupby([id_column])[note_column].apply(" ".join).reset_index()
    else:
        grouped = work.copy()

    EXPECTED = grouped.shape[0]
    meta = (
        grouped[[id_column]].copy() if id_column and id_column in grouped.columns else pd.DataFrame(index=grouped.index)
    )
    if id_column in meta.columns:
        meta[id_column] = meta[id_column].astype("string")

    # --- single pass extraction ---
    res = hlp.regex_extract(
        checklist=checklist_obj,
        df_to_analyze=grouped,
        metadata=meta,
        preview_count=0,
        expected_row_count=EXPECTED,
    )

    # --- reattach GRID AFTER extraction (fix) ---
    if crosswalk is not None and id_column in res.columns:
        res[id_column] = res[id_column].astype("string")
        res = res.merge(crosswalk, on=id_column, how="left")

    # --- drop note text only for RETURN ---
    if not include_note_text and note_column in res.columns:
        res.drop(columns=[note_column], inplace=True)

    return res
