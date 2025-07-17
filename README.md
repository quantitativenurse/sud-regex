[![CI](https://github.com/quantitativenurse/sud-regex/actions/workflows/lint.yml/badge.svg)](https://github.com/quantitativenurse/sud-regex/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


# SUDRegex

> **Version:** 0.2.0

A lightweight, high‑throughput pipeline for regex‑driven extraction and negation/false‑positive filtering of clinical text—optimized for Substance Use Disorder research.

---

## Features

- **Pattern counts**: count occurrences of arbitrary regex patterns per note.  
- **Negation detection**: flag matches preceded by negation cues.  
- **False‑positive filters**: drop matches in common “family history” or “discharge instructions” contexts.  
- **Substance context**: detect user‑supplied vocabulary around a match (e.g. alcohol, opioids).  
- **Line‑break normalization**: remove literal markers (default `"$+$"`) and collapse whitespace.  
- **Built‑in ABC checklist**: ready‑to‑use patterns for illicit drugs, alcohol, opioids, etc.  
- **Customizable**: load your own checklist & term lists.  
- **CLI & Python API**: choose your workflow—script, notebook, or shell.  
- **CI‑ready**: includes a GitHub Actions workflow for `isort`/`black`/`flake8`.

---

## Installation

```bash
# From PyPI (pending release):
pip install SUDRegex

# Or from your local clone:
git clone git@github.com:<your‑org>/SUDRegex.git
cd SUDRegex
pip install -e .[dev]  # installs flake8, black, isort, pytest etc.

import pandas as pd
import SUDRegex as sud

# 1) Load your notes (must have columns: note_id, note_text)
df = pd.read_csv("notes.csv")

# 2) (Optional) normalize line breaks
df["note_text"] = df["note_text"].apply(sud.remove_line_break)

# 3) Extract with built‑in checklist & terms
from SUDRegex import checklist_abc, default_termslist

# Extract returns (out_df, )—for notebook friendliness we unpack only the df
out_df, = sud.extract_df(
    df,
    checklist=checklist_abc,
    termslist=default_termslist,
    terms_active="alcohol_terms,opioid_terms",
    note_column="note_text",
    parallel=False,
    debug=False
)

print(out_df.head())

# Core Helper function 
>>> sud.__all__
[
  "__version__", "extract", "extract_df", "remove_line_break",
  "remove_tobacco_mentions", "set_terms", "regex_extract",
  "check_for_substance", "check_negation",
  "check_common_false_positives", "discharge_instructions",
  "preview_string_matches", "checklist_abc", "default_termslist",
]

# CLI Usage
python -m SUDRegex.cli \
  --extract \
  --in_file notes.csv \
  --out_file results.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active alcohol_terms,opioid_terms \
  --separator , \
  --include_note_text \
  --parallel \
  --debug

# Providing your own checklist 
# Your checklist is a dict[str,dict] with keys:

{
  "my_item": {
    "lab":       "Human‑readable label",
    "pat":       r"your_regex_pattern",
    "col_name":  "your_column_name",
    "negation":  True|False,
    "substance": False,           # omit or False if no context check
    "preview":   False,           # omit or False to skip text preview
    "common_fp": ["family",...]   # optional false‑positive terms
  },
  ...
}

