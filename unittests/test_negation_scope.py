"""
Tests for the per-item `negation_scope` pattern library key.

Covers the four item-level settings (left / right / both / False), the fallback to
the global `negation_scope` argument, validation, the Geisinger-style lab-result
case, and a regression check that the published ABC pattern library is unchanged.
"""

import pandas as pd
import pytest

import sudregex
from sudregex import helper

FILLER = "The weather was mild and the clinic schedule stayed routine all week. " * 8


def _lib(scope="__missing__", pat=r"cocaine", col="coc"):
    cfg = {"lab": "test item", "pat": pat, "col_name": col, "opioid": False, "negation": True, "preview": False}
    if scope != "__missing__":
        cfg["negation_scope"] = scope
    return {"t": cfg}


def _run(lib, text, **kwargs):
    df = pd.DataFrame({"note_id": ["1"], "note_text": [text]})
    return sudregex.extract_df(df, lib, terms=["oxycodone"], **kwargs).iloc[0]


# A negation cue left of the match, right of the match, and on both sides.
LEFT_NOTE = "Patient denies cocaine."
RIGHT_NOTE = "Cocaine was denied by patient."
BOTH_NOTE = "Patient denies cocaine, which was denied again."


# ---------- item-level settings ----------
@pytest.mark.parametrize(
    "scope, left_kept, right_kept, both_kept",
    [
        ("left", 0, 1, 0),
        ("right", 1, 0, 0),
        ("both", 0, 0, 0),
        (False, 1, 1, 1),
    ],
)
def test_item_level_scope(scope, left_kept, right_kept, both_kept):
    lib = _lib(scope)
    assert _run(lib, LEFT_NOTE)["coc_NEG"] == left_kept
    assert _run(lib, RIGHT_NOTE)["coc_NEG"] == right_kept
    assert _run(lib, BOTH_NOTE)["coc_NEG"] == both_kept


@pytest.mark.parametrize("spelling", ["FALSE", "false", "off", " Right "])
def test_item_level_scope_string_spellings(spelling):
    out = _run(_lib(spelling), RIGHT_NOTE)["coc_NEG"]
    assert out == (0 if spelling.strip().lower() == "right" else 1)


def test_item_level_false_keeps_neg_column_but_applies_no_cues():
    out = _run(_lib(False), LEFT_NOTE)
    assert "coc_NEG" in out.index
    assert out["coc"] == 1 and out["coc_NEG"] == 1


# ---------- global fallback ----------
@pytest.mark.parametrize(
    "global_scope, left_kept, right_kept",
    [("left", 0, 1), ("right", 1, 0), ("both", 0, 0)],
)
def test_global_scope_used_when_item_has_no_value(global_scope, left_kept, right_kept):
    lib = _lib()
    assert _run(lib, LEFT_NOTE, negation_scope=global_scope)["coc_NEG"] == left_kept
    assert _run(lib, RIGHT_NOTE, negation_scope=global_scope)["coc_NEG"] == right_kept


def test_none_value_falls_back_to_global():
    assert _run(_lib(None), RIGHT_NOTE, negation_scope="right")["coc_NEG"] == 0
    assert _run(_lib(None), RIGHT_NOTE, negation_scope="left")["coc_NEG"] == 1


def test_item_value_overrides_global():
    assert _run(_lib("left"), RIGHT_NOTE, negation_scope="right")["coc_NEG"] == 1
    assert _run(_lib("right"), LEFT_NOTE, negation_scope="left")["coc_NEG"] == 1


def test_default_scope_is_left_when_nothing_is_set():
    assert _run(_lib(), LEFT_NOTE)["coc_NEG"] == 0
    assert _run(_lib(), RIGHT_NOTE)["coc_NEG"] == 1


def test_items_can_use_different_scopes_in_one_library():
    text = "Cocaine was denied. Patient denies amphetamine."

    def lib(coc_scope, amp_scope):
        return {
            "coc": _lib(coc_scope, pat=r"cocaine", col="coc")["t"],
            "amp": _lib(amp_scope, pat=r"amphetamine", col="amp")["t"],
        }

    both_suppressed = _run(lib("right", "left"), text)
    assert both_suppressed["coc_NEG"] == 0 and both_suppressed["amp_NEG"] == 0

    both_kept = _run(lib("left", "right"), text)
    assert both_kept["coc_NEG"] == 1 and both_kept["amp_NEG"] == 1


def test_invalid_item_scope_raises_with_item_key():
    with pytest.raises(ValueError, match="'t'"):
        _run(_lib("sideways"), LEFT_NOTE)
    with pytest.raises(ValueError):
        _run(_lib(True), LEFT_NOTE)


def test_negation_scope_ignored_when_negation_is_off():
    lib = _lib("both")
    lib["t"]["negation"] = False
    out = _run(lib, LEFT_NOTE)
    assert out["coc"] == 1 and "coc_NEG" not in out.index


# ---------- lab-result case ----------
LAB_NOTE = "Urine drug screen: amphetamines negative, cocaine positive."


def _lab_lib(cocaine_scope, amphetamine_scope):
    return {
        "amp": {**_lib(amphetamine_scope, pat=r"amphetamine", col="amp")["t"]},
        "coc": {**_lib(cocaine_scope, pat=r"cocaine", col="coc")["t"]},
    }


def test_lab_result_right_only_keeps_cocaine_while_both_suppresses_it():
    # "negative" belongs to amphetamines but sits immediately left of cocaine.
    assert _run(_lab_lib("right", "right"), LAB_NOTE)["coc_NEG"] == 1
    assert _run(_lab_lib("both", "both"), LAB_NOTE)["coc_NEG"] == 0
    assert _run(_lab_lib("left", "left"), LAB_NOTE)["coc_NEG"] == 0


def test_lab_result_right_only_still_honours_the_negative_amphetamine_result():
    # "negative" is right of amphetamines, so a right-only scan correctly suppresses it.
    assert _run(_lab_lib("right", "right"), LAB_NOTE)["amp_NEG"] == 0
    assert _run(_lab_lib("right", "left"), LAB_NOTE)["amp_NEG"] == 1


# ---------- published library is unchanged ----------
def test_published_library_has_no_item_level_scope():
    assert not any("negation_scope" in cfg for cfg in sudregex.pattern_library_abc.values())


def test_published_library_output_is_unchanged():
    """
    Golden output of the published ABC library on synthetic notes, captured from 0.1.8
    before per-item negation_scope was added. Per-item scope must not change it.

    The values intentionally reflect current behaviour, including the known gate
    stacking over-count on multi-match notes (G03, G04, G16). Regenerate this
    dictionary deliberately when that behaviour is changed.
    """
    df = pd.DataFrame({"note_id": list(NOTES), "note_text": list(NOTES.values())})
    out = sudregex.extract_df(
        df, sudregex.pattern_library_abc, termslist=sudregex.default_termslist, terms_active="opioid_terms"
    )
    assert out.shape[1] - 1 == 69
    for _, row in out.iterrows():
        actual = {c: int(v) for c, v in row.items() if c != "note_id" and int(v) != 0}
        assert actual == GOLDEN[row["note_id"]], row["note_id"]


NOTES = {
    "G01": "Patient reports using heroin and is on oxycodone.",
    "G02": "No heroin use. Takes oxycodone daily.",
    "G03": "Patient denies heroin use while taking oxycodone. " + FILLER + "Chart also lists marijuana as a topic.",
    "G04": "Patient denies heroin use, on oxycodone. "
    + FILLER
    + "Discharge instructions: avoid heroin while on oxycodone.",
    "G05": "Urine drug screen: amphetamines negative, cocaine positive. Currently on oxycodone.",
    "G06": "Counseled about problem drinking; alcohol abuse in remission.",
    "G07": "Arrested for DUI in 2015.",
    "G08": "Patient has been stockpiling oxycodone at home.",
    "G09": "Patient was sedated and confused after taking oxycodone.",
    "G10": "Documented nonadherence to oxycodone, taking extra tablets some days.",
    "G11": "Father had alcoholism; pt reports problem drinking since college.",
    "G12": "Percocet 1 tab po q8h prn pain, 25 mg. Take your pain medication as directed.",
    "G13": "Patient received opioids from multiple providers and multiple pharmacies.",
    "G14": "Routine follow-up visit, no complaints today.",
    "G15": "Cocaine, no longer used; on oxycodone.",
    "G16": "Reports cocaine use, on oxycodone. " + FILLER + "Denies heroin use, takes oxycodone.",
}

GOLDEN = {
    "G01": {
        "illicit_drug_use": 1,
        "illicit_drug_use_SUBSTANCE_MATCHED": 1,
        "illicit_drug_use_SUBSTANCE_MATCHED_NEG": 1,
        "opioid_mention": 1,
        "opioid_mention_SUBSTANCE_MATCHED": 1,
        "opioid_mention_SUBSTANCE_MATCHED_NEG": 1,
    },
    "G02": {
        "illicit_drug_use": 1,
        "illicit_drug_use_SUBSTANCE_MATCHED": 1,
        "opioid_mention": 1,
        "opioid_mention_SUBSTANCE_MATCHED": 1,
    },
    "G03": {
        "illicit_drug_use": 2,
        "illicit_drug_use_SUBSTANCE_MATCHED": 1,
        "illicit_drug_use_SUBSTANCE_MATCHED_NEG": 2,
        "opioid_mention": 1,
        "opioid_mention_SUBSTANCE_MATCHED": 1,
    },
    "G04": {
        "illicit_drug_use": 2,
        "illicit_drug_use_SUBSTANCE_MATCHED": 2,
        "illicit_drug_use_SUBSTANCE_MATCHED_NEG": 1,
        "opioid_mention": 2,
        "opioid_mention_SUBSTANCE_MATCHED": 2,
        "opioid_mention_SUBSTANCE_MATCHED_NEG": 2,
    },
    "G05": {
        "illicit_drug_use": 2,
        "illicit_drug_use_SUBSTANCE_MATCHED": 2,
        "illicit_drug_use_SUBSTANCE_MATCHED_NEG": 2,
        "opioid_mention": 1,
        "opioid_mention_SUBSTANCE_MATCHED": 1,
    },
    "G06": {
        "problematic_alcohol_use": 2,
        "problematic_alcohol_use_NEG": 2,
    },
    "G07": {
        "dui_history": 1,
        "dui_history_NEG": 1,
    },
    "G08": {
        "medication_hoarding": 1,
        "medication_hoarding_SUBSTANCE_MATCHED": 1,
        "medication_hoarding_SUBSTANCE_MATCHED_NEG": 1,
        "opioid_mention": 1,
        "opioid_mention_SUBSTANCE_MATCHED": 1,
        "opioid_mention_SUBSTANCE_MATCHED_NEG": 1,
    },
    "G09": {
        "sedated_or_confused": 2,
        "sedated_or_confused_SUBSTANCE_MATCHED": 2,
        "sedated_or_confused_SUBSTANCE_MATCHED_NEG": 2,
        "opioid_mention": 1,
        "opioid_mention_SUBSTANCE_MATCHED": 1,
        "opioid_mention_SUBSTANCE_MATCHED_NEG": 1,
    },
    "G10": {
        "excess_narcotic_use": 1,
        "excess_narcotic_use_SUBSTANCE_MATCHED": 1,
        "excess_narcotic_use_SUBSTANCE_MATCHED_NEG": 1,
        "nonadherence_prn": 1,
        "nonadherence_prn_SUBSTANCE_MATCHED": 1,
        "opioid_mention": 1,
        "opioid_mention_SUBSTANCE_MATCHED": 1,
        "opioid_mention_SUBSTANCE_MATCHED_NEG": 1,
    },
    "G11": {
        "problematic_alcohol_use": 2,
    },
    "G12": {
        "opioid_mention": 2,
        "opioid_mention_SUBSTANCE_MATCHED": 2,
    },
    "G13": {
        "multiple_providers_primary": 2,
        "multiple_providers_primary_SUBSTANCE_MATCHED": 2,
        "multiple_providers_primary_SUBSTANCE_MATCHED_NEG": 2,
        "multiple_providers_alt": 2,
        "multiple_providers_alt_SUBSTANCE_MATCHED": 2,
        "multiple_providers_alt_SUBSTANCE_MATCHED_NEG": 2,
        "opioid_mention": 1,
        "opioid_mention_SUBSTANCE_MATCHED": 1,
        "opioid_mention_SUBSTANCE_MATCHED_NEG": 1,
    },
    "G14": {},
    "G15": {
        "illicit_drug_use": 1,
        "illicit_drug_use_SUBSTANCE_MATCHED": 1,
        "illicit_drug_use_SUBSTANCE_MATCHED_NEG": 1,
        "opioid_mention": 1,
        "opioid_mention_SUBSTANCE_MATCHED": 1,
    },
    "G16": {
        "illicit_drug_use": 2,
        "illicit_drug_use_SUBSTANCE_MATCHED": 2,
        "illicit_drug_use_SUBSTANCE_MATCHED_NEG": 2,
        "opioid_mention": 2,
        "opioid_mention_SUBSTANCE_MATCHED": 2,
        "opioid_mention_SUBSTANCE_MATCHED_NEG": 2,
    },
}
