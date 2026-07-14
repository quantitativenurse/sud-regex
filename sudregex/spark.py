# sudregex/spark.py
"""
Optional PySpark / Azure Databricks execution support for sudregex.

This module is intentionally import-safe without PySpark installed: importing
``sudregex`` (and therefore this module, if it is imported lazily) must never
require ``pyspark`` for ordinary local users. PySpark is only imported inside
the functions that actually need it.

The chosen execution pattern is ``DataFrame.mapInPandas`` (see the SUDS-853
design note). The reasons, in brief:

- ``sudregex.extract_df`` already takes a pandas DataFrame in and returns a
  multi-column pandas DataFrame out, which is exactly the shape ``mapInPandas``
  works with.
- ``mapInPandas`` hands each Spark partition to the Python function as a normal
  pandas DataFrame, so the existing extraction code runs unchanged on each
  worker.
- A ``mapInPandas`` batch may return a different number of rows than it
  received, which tolerates ``sudregex``'s group-by-``note_id`` behavior.

Two correctness rules are enforced here:

1. ``parallel=False`` is forced inside every worker so that Spark's
   distribution is not nested on top of ``pandarallel``/``loky``
   multiprocessing inside each executor.
2. The distributed input is reduced to exactly one row per ``note_id`` before
   ``mapInPandas`` runs, so a single note is never split across two Arrow
   batches (which would double-count or under-count that note).

The Spark return schema is generated from the selected pattern library rather
than hard-coded, so a custom pattern library automatically produces the right
output columns.
"""
from typing import List, Optional

SUPPORTED_ENVIRONMENTS = {"local", "databricks"}


def _resolve_library(pattern_library, keys=None):
    """
    Resolve a pattern library from a Python object or a file path, optionally
    filtering to ``keys``. Mirrors the resolution used by the public API so the
    Spark schema and the worker extraction agree on the same library.
    """
    from .validation import import_python_object

    obj = (
        import_python_object(pattern_library, "pattern_library")
        if isinstance(pattern_library, str)
        else pattern_library
    )
    if keys:
        obj = {k: obj[k] for k in keys if k in obj}
    return obj


def expected_count_columns(pattern_library_obj) -> List[str]:
    """
    Return the ordered list of match-count output columns produced by
    ``regex_extract`` for the given (already-resolved) pattern library.

    This replicates exactly the ``merge_cols`` logic in ``helper.regex_extract``
    so the generated Spark schema matches the local pandas output column for
    column:

    - every pattern emits a raw-count column named ``col_name``;
    - patterns with substance/opioid matching add ``<col>_SUBSTANCE_MATCHED``;
    - those that also negate add ``<col>_SUBSTANCE_MATCHED_NEG``;
    - negation-only patterns (no substance) add ``<col>_NEG``.
    """
    cols: List[str] = []
    for key in pattern_library_obj:
        cfg = pattern_library_obj[key]
        col_name = cfg["col_name"]
        has_substance = bool(cfg.get("substance") or cfg.get("opioid"))
        has_negation = bool(cfg.get("negation"))

        cols.append(col_name)
        if has_substance:
            cols.append(f"{col_name}_SUBSTANCE_MATCHED")
        if has_negation and has_substance:
            cols.append(f"{col_name}_SUBSTANCE_MATCHED_NEG")
        elif has_negation and not has_substance:
            cols.append(f"{col_name}_NEG")
    return cols


def output_columns(
    pattern_library_obj,
    *,
    id_column: str = "note_id",
    person_column: Optional[str] = None,
    extra_id_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Full ordered output column list (identifier columns first, then count
    columns), matching the ordering produced by ``_finalize_result`` in the
    local path.
    """
    front = [c for c in [person_column, id_column, *(extra_id_columns or [])] if c]
    return front + expected_count_columns(pattern_library_obj)


def build_spark_schema(
    pattern_library_obj,
    *,
    id_column: str = "note_id",
    person_column: Optional[str] = None,
    extra_id_columns: Optional[List[str]] = None,
    keys=None,
):
    """
    Build a PySpark ``StructType`` describing the Databricks output, generated
    from the pattern library. Identifier columns are ``StringType``; match-count
    columns are ``LongType`` (64-bit, matching pandas int64).

    ``pattern_library_obj`` may be a resolved dict, a module, or a file path;
    ``keys`` optionally restricts to a subset of pattern entries.
    """
    from pyspark.sql.types import LongType, StringType, StructField, StructType

    resolved = _resolve_library(pattern_library_obj, keys=keys)

    fields = []
    for c in [person_column, id_column, *(extra_id_columns or [])]:
        if c:
            fields.append(StructField(c, StringType(), True))
    for c in expected_count_columns(resolved):
        fields.append(StructField(c, LongType(), True))
    return StructType(fields)


def run_databricks(
    notes,
    pattern_library,
    spark,
    *,
    note_column: str = "note_text",
    id_column: str = "note_id",
    person_column: Optional[str] = None,
    extra_id_columns: Optional[List[str]] = None,
    keys=None,
    terms=None,
    termslist=None,
    terms_active=None,
    remove_linebreaks: bool = True,
    exclude_discharge_mentions: bool = True,
    negation_scope: str = "left",
    aggregate_notes: bool = True,
    num_partitions: Optional[int] = None,
    **_ignored,
):
    """
    Run sudregex extraction distributed across a Spark cluster using
    ``mapInPandas`` and return the resulting **Spark DataFrame**.

    Parameters mirror the relevant ``extract_df`` arguments. ``spark`` must be
    an active ``SparkSession`` (a clear error is raised when it is ``None``).

    ``notes`` may be a pandas DataFrame or a Spark DataFrame. A pandas DataFrame
    is converted with the supplied session. The result is lazy; the caller must
    trigger an action (``.count()``, ``.collect()``, write, ...) to materialize.

    Note: ``include_note_text``, ``preview_*``, ``parallel`` and similar
    local-only arguments are intentionally ignored on the Spark path. Previews
    are excluded from the primary distributed pass by design (``mapInPandas``
    declares a single output table); generate previews separately from a
    smaller sample if needed.
    """
    if spark is None:
        raise ValueError(
            "environment='databricks' requires an active SparkSession passed as " "spark=...; received spark=None."
        )

    try:
        from pyspark.sql import DataFrame as SparkDataFrame
        from pyspark.sql import functions as F
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PySpark is required for environment='databricks'. Install it on the "
            "Databricks cluster (e.g. it is preinstalled in the Databricks Runtime) "
            "or `pip install pyspark` locally."
        ) from exc

    import pandas as pd

    pattern_library_obj = _resolve_library(pattern_library, keys=keys)

    # --- Normalize input to a Spark DataFrame -------------------------------
    if isinstance(notes, pd.DataFrame):
        sdf = spark.createDataFrame(notes)
    elif isinstance(notes, SparkDataFrame):
        sdf = notes
    else:
        raise TypeError(
            "Databricks input must be a pandas DataFrame or a Spark DataFrame; " f"got {type(notes).__name__}."
        )

    # --- Validate required columns ------------------------------------------
    required = [note_column, id_column]
    if person_column:
        required.append(person_column)
    required += list(extra_id_columns or [])
    missing = [c for c in required if c and c not in sdf.columns]
    if missing:
        raise ValueError(f"Missing required columns on Spark input: {missing}")

    # --- Cast identifier columns to string ----------------------------------
    sdf = sdf.withColumn(id_column, F.col(id_column).cast("string"))
    for c in [person_column, *(extra_id_columns or [])]:
        if c:
            sdf = sdf.withColumn(c, F.col(c).cast("string"))

    # --- Enforce one row per note_id ----------------------------------------
    # extract_df joins rows that share a note_id, but under mapInPandas the
    # rows for one note could land in different Arrow batches. Pre-aggregating
    # to a single row per note_id guarantees each note is processed wholly
    # within one batch.
    if aggregate_notes:
        agg_exprs = [F.concat_ws(" ", F.collect_list(F.col(note_column).cast("string"))).alias(note_column)]
        for c in [person_column, *(extra_id_columns or [])]:
            if c:
                agg_exprs.append(F.first(F.col(c).cast("string"), ignorenulls=True).alias(c))
        sdf = sdf.groupBy(id_column).agg(*agg_exprs)
    else:
        # Defensive check: refuse to silently double-count split notes.
        dupes = sdf.groupBy(id_column).count().filter(F.col("count") > 1).limit(1).count()
        if dupes:
            raise ValueError(
                f"Input contains multiple rows for some {id_column!r} values. "
                "Pass aggregate_notes=True (default) or pre-aggregate to one row "
                "per note before distributed execution."
            )

    if num_partitions:
        sdf = sdf.repartition(int(num_partitions), F.col(id_column))

    out_cols = output_columns(
        pattern_library_obj,
        id_column=id_column,
        person_column=person_column,
        extra_id_columns=extra_id_columns,
    )
    schema = build_spark_schema(
        pattern_library_obj,
        id_column=id_column,
        person_column=person_column,
        extra_id_columns=extra_id_columns,
    )
    count_cols = expected_count_columns(pattern_library_obj)

    # Capture plain Python objects only; nothing Spark-bound is closed over.
    _note_column = note_column
    _id_column = id_column
    _person_column = person_column
    _extra = list(extra_id_columns or [])
    _terms = terms
    _termslist = termslist
    _terms_active = terms_active
    _remove_linebreaks = remove_linebreaks
    _exclude_discharge = exclude_discharge_mentions
    _negation_scope = negation_scope
    _library = pattern_library_obj
    _out_cols = out_cols
    _count_cols = count_cols

    def _map_partitions(iterator):
        # Imported on the worker; sudregex must be installed on the cluster.
        import pandas as pd  # noqa: F811

        import sudregex

        empty = pd.DataFrame({c: pd.Series([], dtype="object") for c in _out_cols})

        for pdf in iterator:
            if pdf is None or pdf.empty:
                yield empty
                continue

            res = sudregex.extract_df(
                pdf,
                _library,
                note_column=_note_column,
                id_column=_id_column,
                person_column=_person_column,
                extra_id_columns=_extra or None,
                keys=None,
                parallel=False,  # Spark already distributes the work.
                terms=_terms,
                termslist=_termslist,
                terms_active=_terms_active,
                remove_linebreaks=_remove_linebreaks,
                exclude_discharge_mentions=_exclude_discharge,
                negation_scope=_negation_scope,
                include_note_text=False,
            )

            # Align to the declared schema: add any missing columns as 0,
            # order columns, and coerce dtypes for a clean Arrow conversion.
            for c in _out_cols:
                if c not in res.columns:
                    res[c] = 0
            res = res[_out_cols].copy()
            for c in _count_cols:
                res[c] = pd.to_numeric(res[c], errors="coerce").fillna(0).astype("int64")
            for c in _out_cols:
                if c not in _count_cols:
                    res[c] = res[c].astype("object").where(res[c].notna(), None)
            yield res

    return sdf.mapInPandas(_map_partitions, schema=schema)
