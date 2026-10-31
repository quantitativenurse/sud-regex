"""
Differential parity test: 0.1.8 gate chain versus the 0.2.0 per-match scorer.

The legacy oracle (``unittests/legacy_gating.py``) is the old ``regex_extract`` loop driving
the gate functions that still ship in ``sudregex.helper``. Both paths run through the public
``extract_df`` API on the same synthetic notes, so columns, names, suffixes, dtypes and
ordering are compared at the level the lab's downstream code sees.

Every divergent cell must be a decrease with a mechanical explanation: the legacy chain's
last gate re-counted a match that an earlier gate had rejected. An independent per-match
reference evaluator in this file reproduces both formulas, so each divergence is classified
by the gate(s) the re-counted matches failed. Any increase, any divergence on a single-match
note, and any divergence the reference cannot explain is reported as a bug.

A CSV of every divergence is written to ``unittests/artifacts/parity_divergences.csv``.
"""

import contextlib
import importlib.util
import itertools
import re
from pathlib import Path

import pandas as pd
import pytest

import sudregex
from sudregex import helper


def _load_legacy():
    path = Path(__file__).with_name("legacy_gating.py")
    spec = importlib.util.spec_from_file_location("legacy_gating", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.legacy_regex_extract


legacy_regex_extract = _load_legacy()

ARTIFACTS = Path(__file__).with_name("artifacts")

LONG = " " + "The weather was mild and the clinic schedule stayed routine all week. " * 4
SHORT = " "

# Fragments for one "cocaine" mention. Each one is designed to pass or fail specific gates.
FRAGMENTS = {
    "clean": "Reports cocaine use.",
    "clean_sub": "Reports cocaine use, on oxycodone.",
    "neg_left": "Denies cocaine use.",
    "neg_left_sub": "Denies cocaine use, on oxycodone.",
    "neg_right": "Cocaine use denied.",
    "discharge": "Discharge instructions: avoid cocaine.",
    "fp": "Cocaine history noted.",
}

# One item for each column-naming branch, all with a false-positive list.
MINI_LIBRARY = {
    "sub_neg": {"pat": r"cocaine", "col_name": "sn", "opioid": True, "negation": True, "common_fp": ["history"]},
    "neg_only": {"pat": r"cocaine", "col_name": "n", "opioid": False, "negation": True, "common_fp": ["history"]},
    "sub_only": {"pat": r"cocaine", "col_name": "s", "opioid": True, "negation": False, "common_fp": ["history"]},
    "plain_fp": {"pat": r"cocaine", "col_name": "p", "opioid": False, "negation": False, "common_fp": ["history"]},
    "plain": {"pat": r"cocaine", "col_name": "q", "opioid": False, "negation": False},
}
for _cfg in MINI_LIBRARY.values():
    _cfg.update(lab="parity", preview=False)

# ABC-library sentences, each exercising one or two published items.
ABC_SENTENCES = [
    "Patient reports using heroin and is on oxycodone.",
    "No heroin use. Takes oxycodone daily.",
    "Patient denies heroin use while taking oxycodone.",
    "Chart also lists marijuana as a topic.",
    "Discharge instructions: avoid heroin while on oxycodone.",
    "Urine drug screen: amphetamines negative, cocaine positive. Currently on oxycodone.",
    "Counseled about problem drinking; alcohol abuse in remission.",
    "Father had alcoholism; pt reports problem drinking since college.",
    "Arrested for DUI in 2015.",
    "Patient has been stockpiling oxycodone at home.",
    "Patient was sedated and confused after taking oxycodone.",
    "Percocet 25 mg daily.",
    "Please discuss pain medication with your surgeon.",
    "Patient received opioids from multiple providers and multiple pharmacies.",
    "Wife is concerned about his use of hydrocodone.",
    "Running out early of oxycodone again, not out of bed.",
]


@contextlib.contextmanager
def _legacy_pipeline():
    original = helper.regex_extract
    helper.regex_extract = legacy_regex_extract
    try:
        yield
    finally:
        helper.regex_extract = original


def _both(df, lib, **kwargs):
    new = sudregex.extract_df(df, lib, **kwargs)
    with _legacy_pipeline():
        old = sudregex.extract_df(df, lib, **kwargs)
    return old, new


def _mini_corpus():
    rows = []
    names = list(FRAGMENTS)
    for sep_name, sep in (("long", LONG), ("short", SHORT)):
        for k in (1, 2, 3):
            for n, combo in enumerate(itertools.product(names, repeat=k)):
                # Every pair, and a deterministic third of the triples, keeps the run short.
                if k == 3 and n % 3:
                    continue
                rows.append((f"{sep_name}:{'+'.join(combo)}", sep.join(FRAGMENTS[c] for c in combo)))
    return pd.DataFrame(rows, columns=["note_id", "note_text"])


def _abc_corpus():
    rows = [(f"abc{i}", s) for i, s in enumerate(ABC_SENTENCES)]
    for sep_name, sep in (("long", LONG), ("short", SHORT)):
        for i, j in itertools.product(range(len(ABC_SENTENCES)), repeat=2):
            rows.append((f"abc:{sep_name}:{i}+{j}", ABC_SENTENCES[i] + sep + ABC_SENTENCES[j]))
    return pd.DataFrame(rows, columns=["note_id", "note_text"])


# ---------- independent per-match reference ----------


def _contains(terms, text):
    return any(re.search(re.escape(t), text, flags=re.IGNORECASE) for t in terms)


def _match_verdicts(text, cfg, *, terms, side, exclude_discharge):
    """Per-match gate outcomes as a list of dicts with True = passed."""
    pat = cfg["pat"]
    if not isinstance(pat, re.Pattern):
        pat = re.compile(pat, re.IGNORECASE | re.MULTILINE)
    has_sub = bool(cfg.get("substance") or cfg.get("opioid"))
    has_neg = bool(cfg.get("negation"))
    prune = has_neg or not has_sub
    fp = cfg.get("common_fp") or []
    out = []
    for m in pat.finditer(text):
        s, e = m.span()
        v = {}
        if has_sub:
            v["substance"] = _contains(terms, text[max(0, s - helper.WIN_SUBSTANCE) : e + helper.WIN_SUBSTANCE])
        if has_neg:
            left = text[max(0, s - helper.WIN_NEGATION) : s] if side in ("left", "both") else ""
            right = text[e : e + helper.WIN_NEGATION] if side in ("right", "both") else ""
            cues = helper.NEGATION_GATE_CUES
            v["negation"] = not (_contains(cues, left) or _contains(cues, right))
        if prune and exclude_discharge:
            v["discharge"] = not _contains(
                helper.DISCHARGE_TERMS, text[max(0, s - helper.WIN_DISCHARGE) : e + helper.WIN_DISCHARGE]
            )
        if prune and fp:
            v["fp"] = not _contains(fp, text[max(0, s - helper.WIN_CFP) : e + helper.WIN_CFP])
        out.append(v)
    return out


def _legacy_columns(verdicts, cfg):
    """Column values the 0.1.8 chain produces: each gate re-counts every match."""
    has_sub = bool(cfg.get("substance") or cfg.get("opioid"))
    has_neg = bool(cfg.get("negation"))
    col = cfg["col_name"]
    raw = len(verdicts)
    vals = {col: raw}
    active = raw
    if has_sub:
        active = sum(v["substance"] for v in verdicts) if raw else 0
        vals[f"{col}_SUBSTANCE_MATCHED"] = active
    if has_neg:
        active = sum(v["negation"] for v in verdicts) if active else 0
    for gate in ("discharge", "fp"):
        if active and verdicts and gate in verdicts[0]:
            active = sum(v[gate] for v in verdicts)
    if has_neg:
        vals[f"{col}_SUBSTANCE_MATCHED_NEG" if has_sub else f"{col}_NEG"] = active
    elif not has_sub:
        vals[col] = active
    return vals


def _per_match_columns(verdicts, cfg):
    """Column values the 0.2.0 scorer produces: a match counts only if it passes every gate."""
    has_sub = bool(cfg.get("substance") or cfg.get("opioid"))
    has_neg = bool(cfg.get("negation"))
    col = cfg["col_name"]
    vals = {col: len(verdicts)}
    if has_sub:
        vals[f"{col}_SUBSTANCE_MATCHED"] = sum(v["substance"] for v in verdicts)
    final = sum(all(v.values()) for v in verdicts)
    if has_neg:
        vals[f"{col}_SUBSTANCE_MATCHED_NEG" if has_sub else f"{col}_NEG"] = final
    elif not has_sub:
        vals[col] = final
    return vals


def _explain(verdicts):
    """Describe the matches the legacy chain re-counted after an earlier gate rejected them."""
    gates = list(verdicts[0]) if verdicts else []
    if not gates:
        return None
    last = gates[-1]
    recounted = [v for v in verdicts if v[last] and not all(v.values())]
    failed = sorted({g for v in recounted for g in gates if not v[g]})
    return len(recounted), failed


# ---------- the parity check ----------


def _check_parity(df, lib, terms, *, side, exclude_discharge, label):
    kwargs = dict(terms=terms, negation_scope=side, exclude_discharge_mentions=exclude_discharge)
    old, new = _both(df, lib, **kwargs)

    # Structural contract: identical columns, order, dtypes and row order.
    assert list(old.columns) == list(new.columns), label
    assert old.dtypes.to_dict() == new.dtypes.to_dict(), label
    assert old["note_id"].tolist() == new["note_id"].tolist(), label

    cols_by_item = {}
    for key, cfg in lib.items():
        for c in _cols(cfg):
            cols_by_item[c] = key

    text_by_id = dict(zip(df["note_id"], df["note_text"]))
    divergences = []
    for idx, nid in enumerate(old["note_id"]):
        text = text_by_id[nid]
        verdicts_by_item = {
            key: _match_verdicts(text, cfg, terms=terms, side=side, exclude_discharge=exclude_discharge)
            for key, cfg in lib.items()
        }
        for col, key in cols_by_item.items():
            o, n = int(old.at[idx, col]), int(new.at[idx, col])
            cfg = lib[key]
            verdicts = verdicts_by_item[key]
            ref_old = _legacy_columns(verdicts, cfg)[col]
            ref_new = _per_match_columns(verdicts, cfg)[col]
            assert o == ref_old, f"{label}: legacy oracle disagrees with reference on {nid}/{col}: {o} vs {ref_old}"
            assert n == ref_new, f"{label}: new scorer disagrees with reference on {nid}/{col}: {n} vs {ref_new}"
            if o == n:
                continue
            assert n < o, f"BUG {label}: increase on {nid}/{col}: {o} -> {n}"
            assert len(verdicts) >= 2, f"BUG {label}: single-match note diverged on {nid}/{col}: {o} -> {n}"
            explanation = _explain(verdicts)
            assert explanation is not None, f"BUG {label}: unexplained divergence on {nid}/{col}"
            n_recounted, failed = explanation
            assert o - n == n_recounted, f"BUG {label}: unexplained size of decrease on {nid}/{col}: {o} -> {n}"
            divergences.append(
                {
                    "run": label,
                    "note_id": nid,
                    "column": col,
                    "legacy": o,
                    "per_match": n,
                    "raw_matches": len(verdicts),
                    "recounted_matches": n_recounted,
                    "gates_failed_by_recounted": "+".join(failed),
                    "cause": "legacy chain re-counted matches rejected by an earlier gate",
                }
            )

    # Everything that did not diverge must be identical cell for cell.
    mask = pd.DataFrame(False, index=old.index, columns=old.columns)
    for d in divergences:
        mask.loc[old["note_id"] == d["note_id"], d["column"]] = True
    pd.testing.assert_frame_equal(old.mask(mask), new.mask(mask))
    return divergences


def _cols(cfg):
    has_sub = bool(cfg.get("substance") or cfg.get("opioid"))
    has_neg = bool(cfg.get("negation"))
    col = cfg["col_name"]
    out = [col]
    if has_sub:
        out.append(f"{col}_SUBSTANCE_MATCHED")
    if has_neg:
        out.append(f"{col}_SUBSTANCE_MATCHED_NEG" if has_sub else f"{col}_NEG")
    return out


RUNS = [(side, dis) for side in ("left", "right", "both") for dis in (True, False)]


@pytest.fixture(scope="module")
def divergence_log():
    log = []
    yield log
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(ARTIFACTS / "parity_divergences.csv", index=False)


@pytest.mark.parametrize("side,exclude_discharge", RUNS)
def test_mini_library_parity(side, exclude_discharge, divergence_log):
    df = _mini_corpus()
    label = f"mini:{side}:discharge={exclude_discharge}"
    divs = _check_parity(df, MINI_LIBRARY, ["oxycodone"], side=side, exclude_discharge=exclude_discharge, label=label)
    divergence_log.extend(divs)
    # Single-fragment notes have at most one match and must never diverge.
    single = {d["note_id"] for d in divs if "+" not in d["note_id"]}
    assert not single
    # The corpus is built to trigger the old bug, so a run with no divergence means the check is not working.
    assert divs


@pytest.mark.parametrize("side,exclude_discharge", RUNS)
def test_abc_library_parity(side, exclude_discharge, divergence_log):
    df = _abc_corpus()
    terms = list(sudregex.default_termslist["opioid_terms"])
    label = f"abc:{side}:discharge={exclude_discharge}"
    divs = _check_parity(
        df, sudregex.pattern_library_abc, terms, side=side, exclude_discharge=exclude_discharge, label=label
    )
    divergence_log.extend(divs)
    assert all(d["per_match"] < d["legacy"] for d in divs)


def test_sub_only_items_never_diverge(divergence_log):
    # Substance-only items get no negation, discharge or false-positive pruning in either
    # release, so their columns are identical by construction.
    df = _mini_corpus()
    divs = _check_parity(df, MINI_LIBRARY, ["oxycodone"], side="left", exclude_discharge=True, label="sub_only")
    assert not [d for d in divs if d["column"].split("_")[0] == "s"]


def test_regression_cases_from_ticket():
    lib = {"1a": sudregex.pattern_library_abc["1a"]}
    terms = list(sudregex.default_termslist["opioid_terms"])
    df = pd.DataFrame(
        {
            "note_id": ["a", "b", "c"],
            "note_text": [
                "Patient reports using heroin and is on oxycodone.",
                "No heroin use. Takes oxycodone daily.",
                "Patient admits to heroin use.",
            ],
        }
    )
    old, new = _both(df, lib, terms=terms)
    assert new["illicit_drug_use_SUBSTANCE_MATCHED_NEG"].tolist() == [1, 0, 0]
    pd.testing.assert_frame_equal(old, new)
