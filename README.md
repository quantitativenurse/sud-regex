[![CI](https://github.com/quantitativenurse/sud-regex/actions/workflows/lint.yml/badge.svg)](https://github.com/quantitativenurse/sud-regex/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/sudregex?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/sudregex)


# sudregex

> **Version:** 0.2.0

A lightweight, high-throughput pipeline for regex-driven extraction with negation and false-positive pruning. Developed for Substance Use Disorder (SUD) research, but the core extraction workflow is flexible enough for broader clinical text mining use cases.

---

## ✨ Features

- **Unified gating utilities** for substance context, negation, common false-positive pruning, and discharge-context filtering
- **Configurable negation scope** with `left` (default), `right`, or `both`, set globally or per pattern library item
- **Substance-context gating** to require matches near a user-supplied vocabulary
- **Actual match counts** in output columns — not binary flags
- **Deterministic, gated previews** that only show matches passing all configured gates
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
| `col_name` | Raw match count (for items with neither a substance nor a negation gate, this column already has discharge-context and false-positive pruning applied) |
| `col_name_SUBSTANCE_MATCHED` | Matches that also had a substance term nearby |
| `col_name_SUBSTANCE_MATCHED_NEG` | Matches that passed every gate: substance, negation, discharge context, and common false positives (final signal) |
| `col_name_NEG` | Same as above for items with a negation gate but no substance gate |

Column values are **match counts**, not binary flags. A value of `2` means the pattern matched twice in that note. A match is counted in the final column only if it passes every gate configured for the item. `sudregex.helper.expected_count_columns(pattern_library)` returns the full ordered list of count columns a library produces; the Spark schema is built from the same function.

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
| `negation_scope` | `"left"`, `"right"`, `"both"`, or `False` (optional) | Negation direction for this item only. When omitted, the global `negation_scope` argument is used. See [Per-item negation direction](#per-item-negation-direction). |
| `negation_window` | positive int (optional) | Characters scanned on each side for negation cues for this item only. Default 65. See [Per-item negation window](#per-item-negation-window). |
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

### Negation cues

The negation gate rejects a match when a cue occurs in the window scanned next to it. Cues are compiled, case-insensitive, word-bounded patterns, grouped in `sudregex.helper.NEGATION_CUE_GROUPS`:

| Group | Cues |
|-------|------|
| direct | no, not, never, without |
| denial | deny, denies, denied, denying, denial |
| negative result | neg, neg., negative, negative for, ruled out |
| contraction | didn't, doesn't, don't, isn't, aren't, wasn't, weren't, hasn't, haven't, hadn't, each with a straight apostrophe, a curly apostrophe, or none (`didnt`) |
| absence | absent, absence of, free of |

Word boundaries keep `neg` from matching "neglect" and `no` from matching "casino". Capitalization (`NO`, `Negative`) and trailing punctuation (`neg.`, `denies:`) do not matter. Multiword phrases such as "no evidence of" or "no history of" are not listed because `no` already covers them inside the window.

Some language is deliberately **not** a cue, because sudregex reports match counts and has no state between counted and rejected:

- **Uncertainty**: rule out, r/o, unlikely, doubt, low suspicion for, possible. "Rule out pulmonary embolism" keeps the match counted; "pulmonary embolism was ruled out" rejects it (with a right or both scope). These are collected in `helper.UNCERTAINTY_CUES` for callers who want to build an uncertainty state.
- **Historical or resolved**: resolved, history of, hx of, in remission, former, prior. "AKI resolved" keeps the match counted; "No AKI" rejects it. Collected in `helper.HISTORICAL_CUES`. Note that "no longer" rejects a match through `no`.
- **Inability**: cannot, can't, unable to. In substance-use notes "can't cut down" and "unable to stop using" describe the disorder rather than negate it.

Releases through 0.1.8 used literal substrings (`"neg"` matched "neglect", `"denie"` missed "deny", `"doubt"` counted as negation). That list is kept as `helper.LEGACY_NEGATION_GATE_CUES`; to reproduce earlier output set `sudregex.helper.NEGATION_GATE_CUES = list(sudregex.helper.LEGACY_NEGATION_GATE_CUES)` before extracting.

### Per-item negation direction

The negation gate looks for cues in a window around each match. By default one global setting, `negation_scope="left"` (also `"right"` or `"both"`), controls which side of the match is scanned for every item. An item can override that with an optional `negation_scope` key:

| Item value | Behavior |
|------------|----------|
| `"left"` | Scan only the text before the match |
| `"right"` | Scan only the text after the match |
| `"both"` | Scan both sides |
| `False` (or `"false"` / `"off"`) | Keep the item's `_NEG` columns but apply no negation cues |
| omitted or `None` | Use the global `negation_scope` argument (default `"left"`) |

The key only matters when the item has `"negation": True`; it is ignored otherwise. Items that do not set it behave exactly as before, so existing pattern libraries, including the packaged ABC library, produce identical output. An invalid value raises a `ValueError` naming the item.

**When to prefer `left` or `right`.** Pasted lab results put several substances next to each other, so a cue that belongs to one substance can sit next to another. In `amphetamines negative, cocaine positive`, the word "negative" belongs to amphetamines, but it is immediately left of "cocaine". A left or both scan on the cocaine item treats the cocaine match as negated and drops a true positive. Setting the cocaine item to `"right"` keeps it, and setting the amphetamine item to `"right"` still honors its negative result. Use `"left"` for narrative text such as "denies cocaine use", where the cue comes first. Use `"right"` when results are written after the substance name, as in lab panels ("cocaine negative"). Use `"both"` when either order is common and neighbouring terms are not a concern.

### Per-item negation window

Each side is scanned 65 characters by default. An item can set `negation_window` (a positive integer) to scan fewer or more characters. A short right-only window reads one result in a drug screen and stops before the next one:

```python
# "amphetamines - negative. cocaine - positive. oxycodone - negative."
pattern_library = {
    "coc": {"pat": r"cocaine", "col_name": "cocaine", "negation": True,
            "negation_scope": "right", "negation_window": 12, ...},
}
```

With the default 65-character window, a right scan on `cocaine` still reaches the oxycodone result 20 characters later and wrongly rejects the match; with `negation_window: 12` it reads only `- positive.` and keeps it. The same settings on an amphetamine or oxycodone item still reject their negative results. Items that do not set the key are unchanged.

### Scope terminators (opt-in)

By default the window does not stop at a sentence boundary or a contrast word, so `No fever, but patient has chest pain.` negates "chest pain" and `No pain today. Uses cocaine.` negates "cocaine". To cut the window at the nearest terminator, enable the clinical set before extracting:

```python
import sudregex
from sudregex import helper

helper.NEGATION_SCOPE_TERMINATORS = list(helper.CLINICAL_SCOPE_TERMINATORS)  # but, however, although, except, nevertheless, . ;
result = sudregex.extract_df(df, pattern_library, terms=[...])
```

The left window then keeps only the text after its last terminator and the right window only the text before its first, so a cue in another clause or sentence no longer reaches the match. "Pt. denies cocaine" still rejects (the cue is after the period) and "Flu: neg." still rejects a right-scanned "flu" (the period is after the cue). This is off by default (`NEGATION_SCOPE_TERMINATORS = []`) so existing output is unchanged. The validator honours the same setting.

### Known limitations of window-based gating

Every gate scans a fixed number of characters around each match: 100 for substance terms, 65 for negation cues (or the item's `negation_window`) and common false-positive terms, and 250 for discharge context. Two consequences are worth knowing:

- **Windows do not stop at sentence boundaries or contrast words unless scope terminators are enabled.** A cue in the previous sentence, or before "but", still counts.
- **Lab-result text can drop a true positive.** In `amphetamines negative cocaine positive` the word "negative" belongs to amphetamines but sits directly before "cocaine", so a left or both scan on a cocaine item rejects the match. The mitigations are a right scope, a short `negation_window`, or scope terminators.

Each match is judged on its own windows. A match rejected by one gate is never re-counted by another, and a note with several matches of the same item can have some counted and some rejected.

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

### 0.2.0

- **Added per-item negation direction.** Pattern library items accept an optional `negation_scope` key (`"left"`, `"right"`, `"both"`, or `False`). When an item does not set it, the global `negation_scope` argument is used, so existing pattern libraries and the packaged ABC library produce identical output.
- **Documented the lab-result case** (`amphetamines negative, cocaine positive`) and when to prefer `left` or `right`.
- **Added `unittests/test_negation_scope.py`** covering each setting, the global fallback, validation, the lab-result case, and a regression check of the packaged library.
- **Fixed: final match counts now apply every gate per match.** Previously the substance, negation, discharge and common-false-positive checks each re-counted every match in the note, so a match that failed an earlier check could still be counted in the final `_NEG` / `_SUBSTANCE_MATCHED_NEG` column. Each match is now checked against all configured gates once, and counted only if it passes all of them. Notes with a single match are unaffected; on notes with several matches the final count can be lower than in earlier versions. In a 100,000-note sample this changed 193 notes (about 15% fewer counted matches overall), and the number of notes flagged changed only for `opioid_mention`.
- **The final-column counts can therefore differ from results produced with 0.1.7 and 0.1.8.** The raw count and `_SUBSTANCE_MATCHED` columns are unchanged.
- **Faster extraction.** The pattern is scanned once per note and item instead of up to five times.
- **Added `unittests/test_per_match_gating.py`** covering matches that fail different gates, discharge on and off, common false positives, and serial vs. parallel agreement.
- **Reviewed the per-match fix.** Matches are judged in place during a single `finditer` pass, so repeated identical text, adjacent matches, matches at the start or end of a note, and items that match several times under a per-item `negation_scope` are each handled per match. Added `unittests/test_gating_review.py`.
- **Fixed: zero-row input now yields int64 count columns.** Earlier releases produced a string-typed raw column and the per-match fix produced object-typed columns on an empty input.
- **Added a differential parity test** (`unittests/test_parity_legacy.py`). The 0.1.8 gate chain is kept verbatim as a test oracle (`unittests/legacy_gating.py`) and run against the new scorer on synthetic notes under every negation scope and discharge setting. Column names, suffixes, order and dtypes are identical. Every divergence is a decrease on a note with two or more matches, caused by the old chain re-counting a match an earlier gate had rejected; a divergence report is written to `unittests/artifacts/parity_divergences.csv`.
- **Refactored `regex_extract`** into settings resolution, scoring and column assembly. `helper.item_count_columns` and `helper.expected_count_columns` are now the single source of output column names, and `sudregex.spark` builds its schema from them. The duplicate private `_apply_series` was removed from the package root.
- **Previews show only matches that passed the gates.** A preview row for a note that was counted never shows a match the final column rejected. `write_previews_for_item` gained an optional `match_filter` argument; without it the function behaves as before.
- **Documented the known limitations** of window-based gating (sentence boundaries, lab-result text).
- **Scores computed with earlier versions may be inflated for multi-match notes.** ABC item scores and any models or analyses built on sudregex 0.1.x output should be regenerated with 0.2.0 before being compared with new results. Single-match notes and the raw and `_SUBSTANCE_MATCHED` columns are unchanged.
- **Reworked the negation cues.** `helper.NEGATION_GATE_CUES` is now a list of compiled, case-insensitive, word-bounded patterns built from `helper.NEGATION_CUE_GROUPS` (direct, denial, negative result, contraction, absence). Compared with the 0.1.8 substrings: "neg" no longer matches inside "neglect" or "negligible", "no" no longer matches inside "casino", "deny" and "denying" are recognized, "No." and "not." are recognized without a trailing space, every common auxiliary contraction is recognized with a straight apostrophe, a curly apostrophe, or none, "ruled out" (but never "rule out") rejects a match, and "absent", "absence of" and "free of" are cues. "doubt" is no longer a negation cue: it is uncertainty. The old list is kept as `helper.LEGACY_NEGATION_GATE_CUES`. Previews and the validator use the same cues.
- **Uncertainty and historical language are documented as not-negation.** `helper.UNCERTAINTY_CUES` (rule out, r/o, unlikely, doubt, low suspicion for, possible) and `helper.HISTORICAL_CUES` (resolved, history of, in remission, former, prior) are available but not applied, because the package has no state between counted and rejected.
- **Added per-item `negation_window`** (characters scanned on each side, default 65), so a right-only scan of a drug-screen result can stop before the next result.
- **Added opt-in scope terminators** (`helper.NEGATION_SCOPE_TERMINATORS`, with `helper.CLINICAL_SCOPE_TERMINATORS` providing but, however, although, except, nevertheless, period and semicolon). Off by default.
- **The validator applies the same negation rules as extraction**: the package cues, per-item `negation_scope` and `negation_window`, a new `negation_scope` argument, and the terminators. Its left window no longer includes the match text itself.
- **Added `unittests/test_clinical_negation.py`** covering direct negation, denial, contractions and transcription variants, negative-result language with and without punctuation, scope and contrast, uncertainty versus negation, historical versus negated, capitalization, strings that must not be cues, the legacy-versus-current cue manifest, `negation_window`, terminators, and validator agreement with extraction.

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

> Quantitative Nurse Lab. (2025). *sudregex* (Version 0.2.0). GitHub. https://github.com/quantitativenurse/sud-regex

### Acknowledgements

This work was supported, in part, by the National Institute on Drug Abuse under award number DP1DA056667. The content is solely the responsibility of the authors and does not necessarily represent the official views of the U.S. Government or the National Institutes of Health.

Thanks to all contributors and collaborators for feedback and testing.
