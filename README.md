[![CI](https://github.com/quantitativenurse/sud-regex/actions/workflows/lint.yml/badge.svg)](https://github.com/quantitativenurse/sud-regex/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/sudregex?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/sudregex)


# sudregex

> **Version:** 0.1.8

A lightweight, high-throughput pipeline for regex-driven extraction with negation and false-positive pruning. Developed for Substance Use Disorder (SUD) research, but the core extraction workflow is flexible enough for broader clinical text mining use cases.

---

## ✨ Features

- **Unified gating utilities** for substance context, negation, common false-positive pruning, and discharge-context filtering
- **Configurable negation scope** with `left` (default), `right`, or `both`
- **Substance-context gating** to require matches near a user-supplied vocabulary
- **Actual match counts** in output columns — not binary flags
- **Deterministic, gated previews** that only show rows passing all configured gates
- **Notebook-friendly preview output** via `previews_df`
- **Line-break normalization** with whitespace cleanup
- **Packaged defaults** including an ABC pattern library and grouped term lists
- **CLI and Python APIs** for shell workflows and notebook use
- **Multiple parallel backends** with support for `pandarallel` and `loky`
- **Distributed Databricks/Spark execution** via `run_sudregex(..., environment="databricks")`, using the same extraction logic as the local pandas path
- **Python 3.9–3.13 compatible**

---

## 📦 Installation

### From PyPI

```bash
pip install sudregex
```

To upgrade an existing installation:

```bash
pip install --upgrade sudregex
```

For Databricks or Spark execution, install the optional `spark` extra:

```bash
pip install "sudregex[spark]"
```

This installs `pyspark>=3.4` and `pyarrow>=15.0`. PySpark is only imported when the Databricks execution path is actually used — local pandas execution (`extract()`, `extract_df()`, and `run_sudregex(..., environment="local")`, the default) never requires PySpark, Java, or a Databricks environment.

PySpark comes pre-installed on Databricks clusters. If you want to test the Databricks path locally, install the `spark` extra above (or a compatible PySpark version directly) plus a working Java 17+ runtime.

### From source

```bash
git clone https://github.com/quantitativenurse/sud-regex.git
cd sud-regex
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

This installs `sudregex` along with development tools: `black`, `isort`, `flake8`, and `pytest`.

For development and testing including the Spark path:

```bash
pip install -e ".[dev,spark]"
```

### Windows setup

```powershell
git clone https://github.com/quantitativenurse/sud-regex.git
cd sud-regex
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .[dev]
```

---

## Identifier columns

Your input data does not need to follow OMOP naming conventions. Map your own identifiers with:

- `--person-column`
- `--note-id-column`

Extra identifier columns can be passed through the pipeline when needed.

---

## Usage

For an interactive walkthrough, see the tutorial notebook:

`sudregex_tutorial_notebook.ipynb`

---

## Quick Start (CLI)

Show help:

```bash
sudregex --help
```

### Run extraction on a comma-delimited file

#### macOS / Linux

```bash
sudregex --extract \
  --in_file path/to/notes.csv \
  --out_file path/to/results.csv \
  --pattern-library path/to/my_pattern_library.py \
  --termslist path/to/termslist.py \
  --terms_active opioid_terms \
  --separator , \
  --parallel \
  --parallel-backend loky \
  --n-workers 4 \
  --person-column patient_id \
  --note-id-column note_id \
  --negation-scope left \
  --exclude-discharge-mentions
```

#### Windows PowerShell

```powershell
sudregex --extract `
  --in_file path/to/notes.csv `
  --out_file path/to/results.csv `
  --pattern-library path/to/my_pattern_library.py `
  --termslist path/to/termslist.py `
  --terms_active opioid_terms `
  --separator , `
  --parallel `
  --parallel-backend loky `
  --n-workers 4 `
  --person-column patient_id `
  --note-id-column note_id `
  --negation-scope left `
  --exclude-discharge-mentions
```

### Validate a pattern library against labeled examples

```bash
sudregex --validate \
  --pattern-library path/to/my_pattern_library.py \
  --examples path/to/validation_examples.txt \
  --val_out validation_detailed.csv \
  --val_by_item validation_by_item.csv \
  --print_mismatches
```

### Parallel backends

`sudregex` supports two parallel backends:

- `pandarallel` — multiprocessing via pandas `.parallel_apply()`
- `loky` — joblib-based, works on all platforms including Windows

```bash
# Loky (recommended — cross-platform)
sudregex --extract ... --parallel --parallel-backend loky --n-workers 4

# Pandarallel
sudregex --extract ... --parallel --parallel-backend pandarallel --n-workers 4
```

### Files without headers

If your input file has no header row, use `--no-header` and specify column names in file order:

#### macOS / Linux

```bash
sudregex --extract \
  --in_file path/to/notes.txt \
  --out_file path/to/results.csv \
  --pattern-library path/to/my_pattern_library.py \
  --termslist path/to/termslist.py \
  --terms_active opioid_terms \
  --separator $'\t!\\^!\t' \
  --no-header \
  --columns patient_id,note_id,note_text
```

#### Windows PowerShell

```powershell
sudregex --extract `
  --in_file path/to/notes.txt `
  --out_file path/to/results.csv `
  --pattern-library path/to/my_pattern_library.py `
  --termslist path/to/termslist.py `
  --terms_active opioid_terms `
  --separator '\t!\^!\t' `
  --no-header `
  --columns patient_id,note_id,note_text
```

---

## Discharge-instruction pruning

By default, `sudregex` excludes matches found in discharge-instruction contexts.

```bash
# Default — exclude discharge mentions (recommended)
sudregex --extract ... --exclude-discharge-mentions

# Include discharge-context hits
sudregex --extract ... --include-discharge-mentions
```

---

## Custom separators

Clinical notes often contain commas and tabs as part of normal text. A custom delimiter prevents parsing ambiguity.

For tab-delimited custom markers such as `\t!^!\t`:

**macOS / Linux:**
```bash
--separator $'\t!\\^!\t'
```

**Windows PowerShell:**
```powershell
--separator '\t!\^!\t'
```

> `pandas.read_csv(..., engine="python")` treats `sep` as a regular expression.
> Escape regex-special characters in your separator accordingly.

---

## Quick Start (Python API)

```python
import sudregex as sud

# Packaged defaults
pattern_library = sud.pattern_library_abc   # ABC OUD checklist (20 items)
termslist = sud.default_termslist           # grouped vocab: opioid_terms, alcohol_terms, chronic_pain_terms

# In-memory DataFrame API
result_df, previews_df = sud.extract_df(
    df=my_notes_df,
    pattern_library=pattern_library,        # dict or path to pattern_library.py
    termslist=termslist,                    # dict, module, or path to termslist.py
    terms_active="opioid_terms",            # which term group to use for substance gating
    person_column="patient_id",             # optional — reattached to output
    id_column="note_id",
    include_note_text=True,
    remove_linebreaks=True,
    exclude_discharge_mentions=True,
    preview_count=5,
    preview_span=120,
    negation_scope="left",
    parallel=False,
    debug=False,
    return_previews_df=True,
)

print("Results shape:", result_df.shape)
print("Previews shape:", previews_df.shape)
```

### Output columns

Each pattern library item produces up to three output columns:

| Column | Description |
|--------|-------------|
| `col_name` | Raw match count |
| `col_name_SUBSTANCE_MATCHED` | Matches that also had a substance term nearby |
| `col_name_SUBSTANCE_MATCHED_NEG` | Matches that survived negation (final signal) |

Column values are **match counts**, not binary flags. A value of `2` means the pattern matched twice in that note.

### Preview columns

```python
# previews_df columns:
# item_key, note_id, span_start, span_end, snippet, snippet_marked

# Filter previews for a specific item
previews_df.query("item_key == '1a'")[["note_id", "snippet_marked"]].head(10)
```

### File-based API

```python
import sudregex as sud

sud.extract(
    in_file="notes.csv",
    out_file="results.csv",
    pattern_library="path/to/my_pattern_library.py",
    separator=",",
    termslist="path/to/termslist.py",
    terms_active="opioid_terms",
    remove_linebreaks=True,
    exclude_discharge_mentions=True,
    preview_count=5,
    preview_file="note_previews.txt",
    preview_csv="previews.csv",
    negation_scope="left",
    parallel=True,
    parallel_backend="loky",
    n_workers=4,
)
```

### Validation API

```python
from sudregex.validation import validate_pattern_library

detailed, by_item = validate_pattern_library(
    pattern_library=sud.pattern_library_abc,
    examples=val_df,                        # DataFrame with item_key | expected | note_text
    substance_terms=sud.default_termslist["opioid_terms"],
)

# by_item columns: item_key, n, tp, fp, fn, precision, recall, f1
print(by_item.to_string(index=False))

# detailed columns: item_key, expected, actual_match, mismatch, failure_reason
# failure_reason values: negated | needs_substance | common_fp | no_raw_hit
print(detailed[["item_key", "expected", "actual_match", "mismatch", "failure_reason"]])
```

---

## Bringing your own pattern library

Each item in the pattern library is a dict with these keys:

| Key | Type | Purpose |
|-----|------|---------|
| `lab` | str | Human-readable label |
| `pat` | str or compiled regex | The regex pattern |
| `col_name` | str | Output column name |
| `substance` | bool | Require a substance term nearby? |
| `negation` | bool | Apply the negation gate? |
| `preview` | bool | Emit preview snippets? |
| `common_fp` | list (optional) | Terms that indicate a false positive |

Save your pattern library as a `.py` file defining a variable named `pattern_library`:

```python
# my_pattern_library.py
import re

pattern_library = {
    "item_A": {
        "lab": "Active opioid use — self-report",
        "pat": re.compile(r"(patient|pt).{0,60}(report|admit|endors).{0,60}(opioid|heroin|fentanyl)", re.IGNORECASE),
        "col_name": "opioid_active_use",
        "substance": True,
        "negation": True,
        "preview": True,
    },
}
```

> **Backward compatibility:** The `checklist=` parameter is a deprecated alias for `pattern_library=`.
> Files that define a `checklist` variable still work. Both will be supported through the next major version.

---

## Termslist structure

The termslist is a dict of named term groups:

```python
termslist = {
    "opioid_terms": ["heroin", "fentanyl", "oxycodone", ...],
    "alcohol_terms": ["alcohol", "etoh", "ethanol", ...],
    "chronic_pain_terms": ["chronic pain", "fibromyalgia", ...],
}
```

Pass a specific group via `terms_active=` or directly via `terms=`:

```python
# Option 1 — via termslist + terms_active (recommended for file-based workflows)
sud.extract_df(..., termslist=termslist, terms_active="opioid_terms")

# Option 2 — pass the list directly (convenient for notebooks)
sud.extract_df(..., terms=termslist["opioid_terms"])
```

---

## Packaged defaults

```python
import sudregex as sud

pattern_library = sud.pattern_library_abc   # ABC OUD checklist
termslist = sud.default_termslist           # grouped term vocabulary
```

---

## Output naming behavior

When using `extract()` with chunked input:

- If exactly one result batch is produced → output is written to `out_file`
- If multiple batches are produced → numbered part files are written:

```
results_part_0.csv
results_part_1.csv
results_part_2.csv
```

---

## Databricks / Spark execution

Version 0.1.8 adds a distributed execution path for Databricks and Apache Spark through a new public entry point, `run_sudregex()`. It runs the exact same pandas-based extraction logic (`extract_df()` internally) across Spark partitions via `mapInPandas`, rather than maintaining a second, separate matching engine — so local and distributed results stay consistent by construction.

### New public API

```python
run_sudregex(
    notes,
    pattern_library,
    environment="local",
    spark=None,
    **kwargs,
)
```

- `notes` — a pandas DataFrame (local execution), or a pandas or Spark DataFrame (Databricks execution)
- `pattern_library` — same format accepted by `extract_df()`
- `environment` — `"local"` (default) or `"databricks"`
- `spark` — the active `SparkSession`; required when `environment="databricks"`, ignored for local execution
- `**kwargs` — forwarded to the existing extraction behavior (term lists, identifier columns, text normalization, gating, negation settings)

An unsupported `environment` value raises a `ValueError`. Selecting `"databricks"` without passing a `SparkSession` also raises a clear `ValueError`.

### Local usage

Existing code calling `extract_df()` directly needs no changes — nothing about the local pandas path was modified in this release. The new wrapper simply defaults to that same path:

```python
import sudregex as sud

# Existing call — unchanged
result_df = sud.extract_df(
    my_notes_df,
    sud.pattern_library_abc,
    terms=sud.default_termslist["opioid_terms"],
)

# Equivalent via the new wrapper
result_df = sud.run_sudregex(
    my_notes_df,
    sud.pattern_library_abc,
    terms=sud.default_termslist["opioid_terms"],
)
```

`run_sudregex(notes, library, environment="local", **kwargs)` is designed to produce the same result as calling `extract_df(notes, library, **kwargs)` directly.

### Databricks usage

```python
from pyspark.sql import SparkSession
import sudregex

spark = SparkSession.builder.getOrCreate()

results = sudregex.run_sudregex(
    notes_sdf,
    sudregex.pattern_library_abc,
    environment="databricks",
    spark=spark,
)
```

`results` is a **Spark DataFrame** — it can be filtered, joined, written, or aggregated without collecting the full result to the driver first. A pandas DataFrame is also accepted as input on the Databricks path and will be converted internally.

### Distributed execution behavior

- Input is pre-aggregated to one row per `note_id` before distribution, so a single note is never split across two Arrow batches (`aggregate_notes=True` by default; pass `aggregate_notes=False` to instead raise a `ValueError` if duplicate `note_id`s are detected).
- Worker-level extraction forces `parallel=False` — Spark already provides the outer parallelism, so a second multiprocessing layer inside each partition is deliberately avoided.
- The output schema is generated dynamically from the active pattern library, not hard-coded:
  - Identifier columns use Spark `StringType`
  - Match-count columns use Spark `LongType`
  - Every column is a match **count**, matching local behavior — never a boolean flag
- For the packaged ABC pattern library (27 entries), this produces 69 output count columns: 27 raw-match columns, 22 substance-gated columns (of which 17 also carry the negation gate), and 3 negation-only columns.

### Preview behavior

Preview generation (`preview_count`, `preview_file`, `preview_csv`) is **not** produced on the distributed Spark path — preview collection is driver-oriented and can be expensive to pull off a large cluster job. Generate previews from the local path on a bounded sample instead.

### Backward compatibility

- `extract()` is unchanged.
- `extract_df()` is unchanged.
- `run_sudregex()` defaults to local execution and is equivalent to calling `extract_df()` directly.
- Existing local users do not need PySpark, Java, or a Databricks environment — PySpark is only imported inside the functions that need it.
- Existing pattern libraries, term lists, gating options, negation behavior, and match-count output semantics are all preserved.

### Testing

Spark support is covered by `unittests/test_spark.py` (17 tests):

- Output column-naming logic (`expected_count_columns`, `output_columns`) verified against hand-built libraries and cross-checked against the real ABC `pattern_library`.
- Spark schema generation (`build_spark_schema`), including loading a pattern library from a file path.
- Error handling: missing `SparkSession`, wrong input type, missing required columns, duplicate `note_id`s when `aggregate_notes=False`.
- A real end-to-end integration test against a local `SparkSession`, comparing distributed `run_databricks()` output to local `extract_df()` output row-for-row.
- Output schema matches the declared Spark schema.
- Note aggregation across split rows sharing one `note_id`.
- Person and extra identifier column handling, verified through the full distributed path with per-row correctness checks (not just column presence).

Tests requiring PySpark are skipped automatically (`pytest.importorskip`) when PySpark isn't installed. Tests requiring an actual Spark session will error, rather than skip, if PySpark is installed but no working JVM/`JAVA_HOME` is available — a Java 17+ runtime is required to run the integration tests locally.

Run the full suite with:

```bash
pytest
```

Run just the Spark tests with:

```bash
pytest unittests/test_spark.py -v
```

---

## Changelog

### 0.1.8

- **Added `run_sudregex(...)`** — a unified entry point with `environment="local"|"databricks"` for selecting local pandas or distributed Databricks/Spark execution.
- **Added native Databricks/Apache Spark support** via `mapInPandas`, reusing the existing pandas extraction logic rather than a separate matching engine.
- **Added the optional `spark` dependency extra** (`pip install "sudregex[spark]"`, installs `pyspark>=3.4` and `pyarrow>=15.0`).
- **Fixed: `extra_id_columns` are now correctly reattached to `extract_df()` output.** Previously, only `person_column` was carried through the identifier crosswalk; `extra_id_columns` were computed but silently dropped. This affected both the local and Databricks paths identically (the Databricks path additionally crashed rather than silently dropping the column, since the missing identifier was backfilled with an integer that Arrow couldn't serialize as a string). Fixed by extending `_build_crosswalk()` to carry `extra_id_columns` alongside `person_column`.
- **No other changes to `extract()` or `extract_df()`** — both otherwise remain behaviorally identical to 0.1.7; the local path of `run_sudregex()` is a direct passthrough to `extract_df()`.
- **Added `unittests/test_spark.py`** covering schema generation, error handling, and real Spark-session integration tests.
- **Added `run_sudregex()` dispatcher tests to `unittests/test_init.py`** covering environment selection, error handling, and local-path parity with `extract_df()`.

### 0.1.7

- **Renamed `checklist` → `pattern_library`** throughout the API, CLI, and internal modules. The old `checklist=` parameter and `--checklist` flag remain as deprecated aliases for backward compatibility.
- **More descriptive output column names** — e.g. `illicit_drug_use` instead of `illicit_drugs`, `nonadherence_prn` instead of `prn`.
- **Match counts instead of binary flags** — `_SUBSTANCE_MATCHED` and `_NEG` columns now return the number of matches that passed each gate, not 0/1.
- **Termslist restructured** into named groups (`opioid_terms`, `alcohol_terms`, `chronic_pain_terms`) for selective activation via `terms_active=`.
- **Fixed `ZeroDivisionError`** in `validate_pattern_library()` when a pattern library item has no positive examples in the validation set. Precision/recall/F1 now return `NaN` instead of crashing.
- **Python 3.13 compatibility** verified — all 69 unit tests pass on Python 3.13.2.
- **Expanded helper utilities** for gating, parallel backends, and preview generation.

### 0.1.6

- Added support for multiple parallel backends
- Added `loky` backend for cross-platform parallel execution
- Preserved identical output across serial, Pandarallel, and Loky workflows
- Improved input handling for headerless files and custom separators

### 0.1.5

- Unified gating utilities for substance, negation, common false positives, and discharge filtering
- Added `negation_scope` with `left`, `right`, and `both`
- Added in-memory preview support with `extract_df(..., return_previews_df=True)`
- Added highlighted preview output via `snippet_marked`
- Improved dtype normalization and error handling

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## 📣 Citation / Acknowledgements

If `sudregex` is useful in your work, please cite:

> Quantitative Nurse Lab. (2025). *sudregex* (Version 0.1.8). GitHub. https://github.com/quantitativenurse/sud-regex

### Acknowledgements

This work was supported, in part, by the National Institute on Drug Abuse under award number DP1DA056667. The content is solely the responsibility of the authors and does not necessarily represent the official views of the U.S. Government or the National Institutes of Health.

Thanks to all contributors and collaborators for feedback and testing.
