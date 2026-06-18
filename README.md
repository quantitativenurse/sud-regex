[![CI](https://github.com/quantitativenurse/sud-regex/actions/workflows/lint.yml/badge.svg)](https://github.com/quantitativenurse/sud-regex/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/sudregex?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/sudregex)


# sudregex

> **Version:** 0.1.6

A lightweight, high-throughput pipeline for regex-driven extraction with negation and false-positive pruning. It was developed for Substance Use Disorder (SUD) research, but the core extraction workflow is flexible enough for broader clinical text mining use cases.

---

## ✨ Features

- **Unified gating utilities** for substance context, negation, common false-positive pruning, and discharge-context filtering
- **Configurable negation scope** with `left` (default), `right`, or `both`
- **Substance-context gating** to require matches near a user-supplied vocabulary
- **Deterministic, gated previews** that only show rows passing all configured gates
- **Notebook-friendly preview output** via `previews_df`
- **Line-break normalization** with whitespace cleanup
- **Packaged defaults** including an ABC checklist and default term lists
- **CLI and Python APIs** for shell workflows and notebook use
- **Multiple parallel backends** with support for `pandarallel` and `loky`

---

## 📦 Installation

### From PyPI

```bash
pip install sud-regex
```

### From source

```bash
git clone https://github.com/quantitativenurse/sud-regex.git
cd sud-regex
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

This installs `sudregex` along with development tools such as `black`, `isort`, `flake8`, and `pytest`.

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

Your input data does not need to follow OMOP naming conventions. You can map your own identifiers with:

- `--person-column`
- `--note-id-column`

You can also pass extra identifier columns through the pipeline when needed.

---

## Usage

For interactive notebook usage, see the tutorial:

`sudregex_tutorial_notebook.ipynb`

---

## Quick Start (CLI)

Show help:

```bash
sudregex --help
```

Run extraction on a comma-delimited file:

### macOS / Linux

```bash
sudregex --extract \
  --in_file path/to/notes.csv \
  --out_file path/to/results.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active alcohol_terms,opioid_terms \
  --separator , \
  --parallel \
  --n-workers 2 \
  --negation-scope left \
  --exclude-discharge-mentions
```

### Windows PowerShell

```powershell
sudregex --extract `
  --in_file path/to/notes.csv `
  --out_file path/to/results.csv `
  --checklist path/to/checklist.py `
  --termslist path/to/termslist.py `
  --terms_active alcohol_terms,opioid_terms `
  --separator , `
  --parallel `
  --n-workers 2 `
  --negation-scope left `
  --exclude-discharge-mentions
```

### Parallel backends

`sudregex` supports two parallel backends:

- `pandarallel`
- `loky`

Example with Pandarallel:

```bash
sudregex --extract \
  --in_file path/to/notes.csv \
  --out_file path/to/results.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active opioid_terms \
  --parallel \
  --parallel-backend pandarallel \
  --n-workers 4
```

Example with Loky:

```bash
sudregex --extract \
  --in_file path/to/notes.csv \
  --out_file path/to/results.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active opioid_terms \
  --parallel \
  --parallel-backend loky \
  --n-workers 4
```

### Files without headers

If your input file does not contain a header row, use `--no-header` and provide column names in file order:

#### macOS / Linux

```bash
sudregex --extract \
  --in_file path/to/notes.txt \
  --out_file path/to/results.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active opioid_terms \
  --separator $'\t!\\^!\t' \
  --no-header \
  --columns patient_id,note_id,note_text
```

#### Windows PowerShell

Because PowerShell does not support Bash ANSI-C quoting, pass the escaped regex directly:

```powershell
sudregex --extract `
  --in_file path/to/notes.txt `
  --out_file path/to/results.csv `
  --checklist path/to/checklist.py `
  --termslist path/to/termslist.py `
  --terms_active opioid_terms `
  --separator '\t!\^!\t' `
  --no-header `
  --columns patient_id,note_id,note_text
```

---

## Discharge-instruction pruning

By default, `sudregex` excludes matches found in discharge-instruction contexts.

Use the default behavior explicitly:

```bash
sudregex --extract ... --exclude-discharge-mentions
```

To keep discharge-context hits:

```bash
sudregex --extract \
  --in_file path/to/notes.csv \
  --out_file path/to/results_raw.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active alcohol_terms \
  --include-discharge-mentions
```

---

## Custom separators

Clinical notes often contain commas, tabs, semicolons, and other punctuation as part of normal text. For text-based input files, using a custom delimiter can make parsing more reliable.

A custom separator is useful when:

- the note text contains commas or tabs
- standard delimiters create parsing ambiguity
- you want a delimiter that is unlikely to appear in clinical text

Example using a custom token:

```bash
sudregex --extract \
  --in_file path/to/notes.txt \
  --out_file path/to/results.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active opioid_terms \
  --separator '\\|\\^\\|'
```

If your separator contains regex-special characters, remember that `pandas.read_csv(..., engine="python")` treats `sep` as a regular expression. Escape it accordingly.

For tab-delimited custom markers such as `\t!^!\t`, use:

### macOS / Linux

```bash
--separator $'\t!\\^!\t'
```

### Windows PowerShell

```powershell
--separator '\t!\^!\t'
```

---

## Quick Start (Python API)

```python
import sudregex as sud

# Packaged defaults
checklist = sud.checklist_abc
terms = sud.default_termslist

# In-memory DataFrame API
result_df, previews_df = sud.extract_df(
    df=my_notes_df,                          # requires note text and note identifier columns
    checklist=checklist,                    # dict or path to checklist.py
    termslist=terms,                        # dict, module, or path to termslist.py
    terms_active="alcohol_terms,opioid_terms",
    person_column="patient_sid",            # optional person identifier
    id_column="doc_oid",                    # optional note/document identifier
    include_note_text=True,
    exclude_discharge_mentions=True,
    preview_count=10,
    preview_span=120,
    negation_scope="left",
    parallel=True,
    parallel_backend="loky",
    n_workers=4,
    debug=False,
    return_previews_df=True,
)

# Preview columns:
# item_key, note_id, span_start, span_end, snippet, snippet_marked
print(previews_df.head())
```

Example of filtering previews for a single checklist item:

```python
previews_df.query("item_key == 'cocaine_mention'")[["note_id", "snippet_marked"]].head(10)
```

Example of joining one preview per note back to the main result:

```python
one_preview = (
    previews_df.groupby("note_id").first().reset_index()[["note_id", "snippet_marked"]]
)
result_with_preview = result_df.merge(one_preview, on="note_id", how="left")
```

### File-based API

```python
import sudregex as sud

sud.extract(
    in_file="notes.csv",
    out_file="results.csv",
    checklist="path/to/checklist.py",
    separator=",",
    termslist="path/to/termslist.py",
    terms_active="opioid_terms",
    include_note_text=False,
    exclude_discharge_mentions=False,
    preview_count=10,
    preview_file="note_previews.txt",
    preview_csv="previews.csv",
    negation_scope="both",
    parallel=True,
    parallel_backend="pandarallel",
    n_workers=4,
)
```

---

## Packaged defaults

The package includes a default checklist and default term lists:

```python
import sudregex as sud

checklist = sud.checklist_abc
termslist = sud.default_termslist
```

---

## Output naming behavior

When using `extract()` with chunked input:

- if exactly one result batch is produced, output is written to the requested `out_file`
- if multiple batches are produced, output is written as numbered part files such as:

```text
results_part_0.csv
results_part_1.csv
results_part_2.csv
```

---

## Changelog highlights

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

> Quantitative Nurse Lab. (2025). *sudregex* (Version 0.1.6). GitHub. https://github.com/quantitativenurse/sud-regex

### Acknowledgements

This work was supported, in part, by the National Institute on Drug Abuse under award number DP1DA056667. The content is solely the responsibility of the authors and does not necessarily represent the official views of the U.S. Government or the National Institutes of Health.

Thanks to all contributors and collaborators for feedback and testing.
