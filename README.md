[![CI](https://github.com/quantitativenurse/sud-regex/actions/workflows/lint.yml/badge.svg)](https://github.com/quantitativenurse/sud-regex/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# sudregex

> **Version:** 0.2.0

A lightweight, high-throughput pipeline for regex-driven extraction with negation and false-positive pruning—built for Substance Use Disorder (SUD) research, but flexible enough for general clinical text mining.

---

## ✨ Features

- **Unified gating utilities** – Substance context (“opioid/alcohol/etc.”), negation, common false-positive pruning, and discharge-context filtering now share consistent windowing + logic for easier debugging.
- **Negation scope control** – Choose where to scan for cues: `left` (default), `right`, or `both`.
- **Substance context window** – Require that matches occur near a user-supplied vocabulary.
- **Deterministic, gated previews** – Previews only show rows that **pass all gates** (substance/negation/FP/discharge) with fixed-seed sampling.
- **Notebook-friendly previews** – Return previews as a **DataFrame** (`previews_df`) for interactive review.
- **Line-break normalization** – Remove literal markers (default `"$+$"`) and collapse whitespace.
- **Batteries included** – A ready-to-use “ABC” checklist + default terms list.
- **CLI & Python API** – Use from shell scripts or notebooks
---


## 📦 Installation

```bash
# From PyPI
pip install sud-regex


# From source (dev)
git clone https://github.com/quantitativenurse/sud-regex.git
cd sud-regex
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[dev]   # installs sudregex + black, isort, flake8, pytest, etc.
---
```

## Usage
- For interactive usage on notebooks refer to our tutorial <https://github.com/quantitativenurse/sud-regex/blob/main/notebook_tutorial.ipynb>

### Quick Start (CLI)

```bash
sudregex --help
Run extraction (CSV with commas) using the default pruning behavior:



sudregex --extract \
  --in_file path/to/notes.csv \
  --out_file path/to/results.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active alcohol_terms,opioid_terms \
  --separator , \
  --parallel --n-workers 2 \
  --negation-scope left \
  --exclude-discharge-mentions

```
### Discharge-instruction pruning

By default, sudregex **excludes** matches that occur in discharge-instruction contexts.

- **Default:** no flag needed, or explicit:
```bash
  sudregex --extract ... --exclude-discharge-mentions

To keep discharge-context hits:

sudregex --extract \
  --in_file path/to/notes.csv \
  --out_file path/to/results_raw.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active alcohol_terms \
  --include-discharge-mention
```

### Use a custom separator (example: a unique token unlikely to appear in notes):

Clinical notes often contain commas, semicolons, tabs and other common punctuation marks as part of natural language. Using these as delimiters can lead to unintended splits and parsing errors, especially when extracting structured information from note text fields.
In our work, we use the custom marker |^| because:

  It is highly unlikely to appear naturally in clinical documentation.
  It provides a clear, unambiguous boundary between segments.
  It avoids conflicts with commonly used punctuation, improving extraction accuracy.
  It simplifies line-break normalization and downstream processing.

This choice ensures that our pipeline remains robust across diverse note formats.
```bash
sudregex --extract \
  --in_file path/to/notes.txt \
  --out_file path/to/results.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active opioid_terms \
  --separator $'|^|'    # or any safe custom delimiter
```
---

### Quickstart (Python API)
```bash
import sudregex as sud

# Use packaged defaults if desired
checklist = sud.checklist_abc
terms = sud.default_termslist

# ---- In-memory DataFrame API ----
result_df, previews_df = sud.extract_df(
    df=my_notes_df,                           # requires columns: note_id, note_text
    checklist=checklist,                      # dict or path to checklist.py (must define `checklist`)
    termslist=terms,                          # dict or path to termslist.py (must define groups)
    terms_active="alcohol_terms,opioid_terms",
    include_note_text=True,                   # keep text in result_df if you want to eyeball later
    exclude_discharge_mentions=True,          # default True
    preview_count=10,                         # gated previews (pass all checks)
    preview_span=120,                         # chars on each side
    negation_scope="left",                    # try "both" for stricter negation
    debug=False,
    return_previews_df=True,                  # <<< NEW: get previews as a DataFrame
)

# previews_df columns: item_key, note_id, span_start, span_end, snippet, snippet_marked
previews_df.head()

# Example: show marked previews for a single checklist item
previews_df.query("item_key == 'cocaine_mention'")[["note_id","snippet_marked"]].head(10)

# Optional: join a single preview back to the results per note (for convenience)
one_preview = (previews_df.groupby("note_id").first()
               .reset_index()[["note_id","snippet_marked"]])
result_with_preview = result_df.merge(one_preview, on="note_id", how="left")


# ---- File API ----
sud.extract(
    in_file="notes.csv",
    out_file="results.csv",
    checklist="path/to/checklist.py",
    separator=",",
    termslist="path/to/termslist.py",
    terms_active="opioid_terms",
    include_note_text=False,
    exclude_discharge_mentions=False,         # keep raw matches even in discharge contexts
    preview_count=10,
    preview_file="note_previews.txt",
    preview_csv="previews.csv",
    negation_scope="both",
)


```
---

The default checklist and termslist are available using the below method. 

checklist = sud.checklist_abc

checklist

termslist = sud.default_termslist

termslist 

---

## Changelog (highlights)

0.2.0

Major refactor: unified gating utilities for substance/negation/common-FP/discharge.

New negation_scope (left/right/both) for CLI and Python API.

In-memory previews: extract_df(..., return_previews_df=True) now returns (result_df, previews_df).

Previews are gated and include highlighted variants (snippet_marked).

Early file checks, dtype normalization, clearer errors.

## License 
MIT – see LICENSE for details.

## 📣 Citation / Acknowledgements

If **sudregex** is useful in your work, please cite:

> Quantitative Nurse Lab. (2025). *sudregex* (Version 0.1.2). GitHub. https://github.com/quantitativenurse/sud-regex

**Acknowledgements**
This was work was supported, in part, by the National Institute on Drug Abuse under award number DP1DA056667. The content is solely the responsibility of the authors and does not necessarily represent the official views of the US government or the National Institutes of Health.


- Thanks to all contributors and collaborators for feedback and testing.
---
