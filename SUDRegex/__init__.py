# SUDRegex/__init__.py

__version__ = "0.1.0"

import importlib.util
import os
import time

import pandas as pd

from .checklist import checklist as checklist_abc
from .helper import (check_common_false_positives, check_for_substance,
                     check_negation, discharge_instructions,
                     preview_string_matches, regex_extract, remove_line_break,
                     remove_tobacco_mentions, set_terms)
from .termslist import termslist as default_termslist

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
]


def _import_python_object(file_path, varname):
    """Dynamically import a variable (e.g., checklist) from a Python file."""
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, varname) if varname else module


def extract(
    in_file,
    out_file,
    checklist,
    separator=",",
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
):
    """
    Run regex extraction and save to CSV.
    """
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
        groups = (
            [g.strip() for g in terms_active.split(",")]
            if isinstance(terms_active, str)
            else list(terms_active)
        )
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

    start = time.time()
    part = 0

    for chunk in pd.read_csv(
        in_file,
        sep=separator,
        engine="python",
        nrows=nrows,
        chunksize=chunk_size,
    ):
        # ensure required columns
        req = {note_column, id_column}
        if grid_column:
            req.add(grid_column)
        missing = req - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing required cols: {missing}")

        # subset and drop NAs
        keep = list(req)
        if keep_columns:
            keep += [c for c in keep_columns if c not in keep]
        chunk = chunk[keep]
        chunk.dropna(subset=[note_column, id_column], inplace=True)

        # save crosswalk
        crosswalk = chunk[[c for c in (grid_column, id_column) if c in chunk.columns]]

        # line‑break removal
        if remove_linebreaks:
            if use_parallel:
                chunk[note_column] = chunk[note_column].parallel_apply(
                    hlp.remove_line_break
                )
            else:
                chunk[note_column] = chunk[note_column].apply(hlp.remove_line_break)

        # group texts by note_id
        grouped = chunk.groupby([id_column])[note_column].apply(" ".join).reset_index()

        # bring extras back
        if keep_columns:
            extras = chunk.groupby([id_column])[keep_columns].first().reset_index()
            grouped = grouped.merge(extras, on=id_column, how="left")

        EXPECTED = grouped.shape[0]
        meta = grouped[[id_column]]

        # chunked filename
        out_fname = (
            out_file
            if chunk_size == 1
            else out_file.replace(".csv", f"_part_{part}.csv")
        )

        # do the regex extraction
        result = hlp.regex_extract(
            checklist=checklist_obj,
            df_to_analyze=grouped,
            metadata=meta,
            preview_count=0,
            expected_row_count=EXPECTED,
        )

        # re‑merge grid/note_id crosswalk
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
):
    """
    Run regex extraction on a DataFrame and return a new DataFrame with added columns.
    """
    import SUDRegex.helper as hlp

    hlp.PRINT = debug

    # load the checklist
    if isinstance(checklist, str):
        checklist_obj = _import_python_object(checklist, "checklist")
    else:
        checklist_obj = checklist

    # build the terms list
    if terms:
        terms_list = list(terms)
    elif termslist and terms_active:
        groups = (
            [g.strip() for g in terms_active.split(",")]
            if isinstance(terms_active, str)
            else list(terms_active)
        )
        terms_list = []
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
    else:
        raise ValueError(
            "You must supply either `terms` or (`termslist` & `terms_active`)"
        )

    hlp.set_terms(terms_list)

    # optionally strip linebreaks into a temp column
    proc_col = note_column
    if remove_linebreaks:
        df["_note_proc"] = df[note_column].apply(hlp.remove_line_break)
        proc_col = "_note_proc"

    out = df.copy()
    keys_to_run = keys or list(checklist_obj.keys())

    for k in keys_to_run:
        single = {k: checklist_obj[k]}
        meta = (
            out[["note_id"]].copy()
            if "note_id" in out.columns
            else pd.DataFrame(index=out.index)
        )
        res = hlp.regex_extract(
            checklist=single,
            df_to_analyze=out.assign(**{proc_col: out[proc_col]}),
            metadata=meta,
            preview_count=0,
            expected_row_count=out.shape[0],
        )
        coln = checklist_obj[k].get("col_name", k)
        out[coln] = res[coln]

    if remove_linebreaks:
        out.drop(columns=["_note_proc"], inplace=True)

    return out
