[![CI](https://github.com/quantitativenurse/sud-regex/actions/workflows/lint.yml/badge.svg)](https://github.com/quantitativenurse/sud-regex/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# sudregex

> **Version:** 0.1.0

A lightweight, high-throughput pipeline for regex-driven extraction with negation and false-positive pruning—built for Substance Use Disorder (SUD) research, but flexible enough for general clinical text mining.

---

## ✨ Features

- **Negation detection** – Filter matches when preceded by cues (e.g., “no”, “denies”, “not”).  
- **False-positive ** – Drop matches in noisy contexts (e.g., **discharge instructions**, **family history**).  
- **Substance context window** – Confirm that matches occur near a user-supplied vocabulary (e.g., opioid, alcohol terms).  
- **Line-break normalization** – Remove literal markers (default `"$+$"`) and collapse whitespace.  
- **Batteries included** – A ready-to-use “ABC” checklist for common SUD signals.  
- **CLI & Python API** – Use from shell scripts or notebooks.  
- **Deterministic previews** – Sampling uses a fixed seed for reproducible tests.

---


## 📦 Installation

```bash
# From PyPI (enable after publish)
pip install sud-regex


# From source (dev)
git clone https://github.com/quantitativenurse/sud-regex.git
cd sud-regex
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[dev]   # installs sudregex + black, isort, flake8, pytest, etc.

## Usage
- For interactive usage on notebooks refer to our tutorial <link>


#Quick Start (CLI)
sudregex --help
Run extraction (CSV with commas) using the default pruning behavior:

sudregex --extract \
  --in_file path/to/notes.csv \
  --out_file path/to/results.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active alcohol_terms,opioid_terms \
  --separator , \
  --parallel --n-workers 2

### Discharge-instruction pruning

By default, sudregex **excludes** matches that occur in discharge-instruction contexts.

- **Default:** no flag needed, or explicit:
  ```bash
  sudregex --extract ... --exclude-discharge-mentions

Turn pruning OFF (keep discharge-context hits):

sudregex --extract \
  --in_file path/to/notes.csv \
  --out_file path/to/results_raw.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active alcohol_terms \
  --no-exclude-discharge-mentions

### Use a custom separator (example: a unique token unlikely to appear in notes):

Clinical notes often contain commas, semicolons, tabs and other common punctuation marks as part of natural language. Using these as delimiters can lead to unintended splits and parsing errors, especially when extracting structured information from note text fields.
In our work, we use the custom marker |^| because:

  It is highly unlikely to appear naturally in clinical documentation.
  It provides a clear, unambiguous boundary between segments.
  It avoids conflicts with commonly used punctuation, improving extraction accuracy.
  It simplifies line-break normalization and downstream processing.

This choice ensures that our pipeline remains robust across diverse note formats.

sudregex --extract \
  --in_file path/to/notes.txt \
  --out_file path/to/results.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active opioid_terms \
  --separator $'|^|'    # or any safe custom delimiter


#Quickstart (Python API)

import sudregex as sud

# Use the packaged defaults if desired
checklist = sud.checklist_abc
terms = sud.default_termslist

# DataFrame API
df_results = sud.extract_df(
    df=my_notes_df,                  # columns: note_id, note_text (and optional grid)
    checklist=checklist,
    termslist=terms,
    terms_active="alcohol_terms,opioid_terms",
    parallel=True,                   # <— enable parallel apply (if pandarallel is installed)
    n_workers=2,                     
    include_note_text=False,
    exclude_discharge_mentions=True, # default True; set False to disable pruning
)

# File API (CSV/TSV/…)
result = sud.extract(
    in_file="notes.csv",
    out_file="results.csv",
    checklist="path/to/checklist.py",
    separator=",",
    termslist="path/to/termslist.py",
    terms_active="opioid_terms",
    parallel=True,
    n_workers=2,                      
    include_note_text=False,
    exclude_discharge_mentions=False, # keep raw matches even in discharge contexts
)



#Checklist usage 
After installing the package, the default checklist and termslist are available using the below method. 

checklist = sud.checklist_abc
checklist

termslist = sud.default_termslist
termslist 

## License 
MIT – see LICENSE for details.

## 📣 Citation / Acknowledgements

If **sudregex** is useful in your work, please cite:

Quantitative Nurse Lab. (2025). *sudregex* (Version 0.1.0). GitHub. https://github.com/quantitativenurse/sud-regex

**Acknowledgements:**  
Thanks to all contributors and collaborators for feedback and testing.
