[![CI](https://github.com/quantitativenurse/sud-regex/actions/workflows/lint.yml/badge.svg)](https://github.com/quantitativenurse/sud-regex/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# SUDRegex

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
pip install -e .[dev]   # installs SUDRegex + black, isort, flake8, pytest, etc.


import pandas as pd
import SUDRegex as sud
from SUDRegex import checklist_abc, default_termslist

# 1) Your DataFrame needs at least: note_id, note_text
df = pd.read_csv("notes.csv", dtype=str)

# 2) Run extraction using built-in checklist and term groups
out_df = sud.extract_df(
    df=df,
    checklist=checklist_abc,
    termslist=default_termslist,
    terms_active="opioid_terms",  # which term groups to activate
    note_column="note_text",
    id_column="note_id",
    grid_column=None,            # include if you have a stratification column
    remove_linebreaks=True,     # already normalized above
    parallel=False,
    debug=False,
    include_note_text=False
)

print(out_df.head())


## CLI 

# Help
python -m SUDRegex.cli --help

python -m SUDRegex.cli \
  --extract \
  --in_file path/to/notes.csv \
  --out_file path/to/results.csv \
  --checklist path/to/checklist.py \
  --termslist path/to/termslist.py \
  --terms_active alcohol_terms,opioid_terms \
  --separator , \
  --include_note_text \
  --parallel

## Validate checklist 

source_dir = "/path"

notes_input format --> item_key | expected match| actual match | free text
detailed, by_item = sud.validation(
    checklist=f"{source_dir}/snapshotCheclist.py",
    examples=f"{source_dir}/notes_input.txt",
    out_csv=f"{source_dir}/package_validation_result.csv",
    by_item_csv=f"{source_dir}/package_checklist_validation_by_item.csv",
)

detailed


## Checklist format

checklist = {
  "my_item": {
    "lab":       "Human-readable label",
    "pat":       r"(your|regex|here)",   # str or compiled re.Pattern
    "col_name":  "my_item",
    "negation":  True,                   # run negation filtering
    "substance": False,                  # enable context vocabulary window
    "preview":   False,                  # print text previews (for debugging)
    "common_fp": ["family", "history"]   # optional false-positive contexts
  },
  # ...
}

## License 
MIT – see LICENSE for details.

##📣 Citation / Acknowledgements

If SUDRegex is useful in your work, please cite the repository and the version used.
Thanks to the team for all contributions. 