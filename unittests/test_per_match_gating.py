"""
Tests that each match is checked against every configured gate (substance, negation,
discharge, common false positives) and counted only if it passes all of them.

Before this was fixed, each gate re-counted every match in the note, so a match that failed
an earlier gate could still be counted in the final column.
"""

import pandas as pd
import pytest

import sudregex

LIB = {"1a": sudregex.pattern_library_abc["1a"]}
FINAL = "illicit_drug_use_SUBSTANCE_MATCHED_NEG"
FILLER = "The weather was mild and the clinic schedule stayed routine all week. " * 8

# (note text, expected final count with discharge exclusion on, expected with it off)
CASES = {
    "one clean positive": ("Patient reports using heroin and is on oxycodone.", 1, 1),
    "one negated": ("No heroin use. Takes oxycodone daily.", 0, 0),
    "two clean positives": (
        "Uses cocaine on weekends, on oxycodone. " + FILLER + "Also smokes marijuana daily, takes oxycodone.",
        2,
        2,
    ),
    "two negated": (
        "Denies heroin use, takes oxycodone. " + FILLER + "Denies cocaine use as well, still on oxycodone.",
        0,
        0,
    ),
    "no opioid term nearby": ("Patient admits to heroin use.", 0, 0),
    # Each case below has matches that fail different gates, so no match passes all of them.
    "negated plus far from opioid term": (
        "Patient denies heroin use while taking oxycodone. " + FILLER + "Chart also lists marijuana as a topic.",
        0,
        0,
    ),
    "negated plus discharge context": (
        "Patient denies heroin use, on oxycodone. "
        + FILLER
        + "Discharge instructions: avoid heroin while on oxycodone.",
        0,
        1,
    ),
    "one positive plus one negated": (
        "Reports cocaine use, on oxycodone. " + FILLER + "Denies heroin use, takes oxycodone.",
        1,
        1,
    ),
    "negated, positive, far from opioid term": (
        "Denies heroin use, on oxycodone. "
        + FILLER
        + "Reports cocaine use, on oxycodone. "
        + FILLER
        + "Chart also lists marijuana as a topic.",
        1,
        1,
    ),
    "negated plus discharge without opioid term": (
        "Denies heroin use, on oxycodone. " + FILLER + "Discharge instructions: stop marijuana.",
        0,
        0,
    ),
}


def _run(text, discharge):
    df = pd.DataFrame({"note_id": ["1"], "note_text": [text]})
    return sudregex.extract_df(
        df,
        LIB,
        termslist=sudregex.default_termslist,
        terms_active="opioid_terms",
        exclude_discharge_mentions=discharge,
    ).iloc[0]


@pytest.mark.parametrize("name", list(CASES))
def test_final_count_only_includes_matches_passing_every_gate(name):
    text, expected_on, expected_off = CASES[name]
    assert _run(text, True)[FINAL] == expected_on
    assert _run(text, False)[FINAL] == expected_off


@pytest.mark.parametrize("name", list(CASES))
def test_counts_never_increase_down_the_gate_chain(name):
    text = CASES[name][0]
    row = _run(text, True)
    assert row[FINAL] <= row["illicit_drug_use_SUBSTANCE_MATCHED"] <= row["illicit_drug_use"]


def test_common_false_positive_applies_per_match():
    lib = {"16": sudregex.pattern_library_abc["16"]}
    # First mention sits next to "mg" (a listed false positive); second is clean.
    text = "Percocet 25 mg daily. " + FILLER + "Please discuss pain medication with your surgeon."
    df = pd.DataFrame({"note_id": ["1"], "note_text": [text]})
    row = sudregex.extract_df(df, lib, termslist=sudregex.default_termslist, terms_active="opioid_terms").iloc[0]
    assert row["opioid_mention"] == 2
    assert row["opioid_mention_SUBSTANCE_MATCHED_NEG"] == 1


def test_serial_and_parallel_agree():
    df = pd.DataFrame({"note_id": [str(i) for i in range(len(CASES))], "note_text": [v[0] for v in CASES.values()]})
    kwargs = dict(termslist=sudregex.default_termslist, terms_active="opioid_terms")
    serial = sudregex.extract_df(df, LIB, **kwargs)
    loky = sudregex.extract_df(df, LIB, parallel=True, parallel_backend="loky", n_workers=2, **kwargs)
    pd.testing.assert_frame_equal(serial.reset_index(drop=True), loky.reset_index(drop=True))
