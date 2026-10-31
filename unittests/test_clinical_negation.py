"""
Clinical negation cues: what negates, what must not, and how the cues behave inside the
window-based scope.

The gate has no assertion states. A match is either counted or rejected, so uncertainty
("rule out", "unlikely") and historical or resolved language ("resolved") are deliberately NOT
negation cues: those matches stay counted. The lists `helper.UNCERTAINTY_CUES` and
`helper.HISTORICAL_CUES` exist so the distinction is explicit and testable.
"""

import re

import pandas as pd
import pytest

import sudregex
from sudregex import helper


def _item(pat, **extra):
    cfg = {"lab": "t", "pat": pat, "col_name": "c", "opioid": False, "negation": True, "preview": False}
    cfg.update(extra)
    return {"t": cfg}


def _kept(text, pat, side="left", **extra):
    """Final count for one note and one negation-only item (1 = counted, 0 = negated)."""
    df = pd.DataFrame({"note_id": ["1"], "note_text": [text]})
    row = sudregex.extract_df(df, _item(pat, **extra), terms=["x"], negation_scope=side, remove_linebreaks=False)
    assert row.at[0, "c"] == 1, f"pattern {pat!r} should match once in {text!r}"
    return int(row.at[0, "c_NEG"])


def _cue_spans(text):
    """(start, end, matched text) for every gate cue found in `text`."""
    out = []
    for p in helper.NEGATION_GATE_CUES:
        out.extend((m.start(), m.end(), m.group()) for m in p.finditer(text))
    return sorted(out)


def _has_cue(text):
    return bool(_cue_spans(text))


# ---------- A. direct negation ----------


@pytest.mark.parametrize(
    "text, pat",
    [
        ("No fever.", "fever"),
        ("Patient does not have fever.", "fever"),
        ("Patient never had seizures.", "seizures"),
        ("Without evidence of infection.", "infection"),
    ],
)
def test_direct_negation(text, pat):
    assert _kept(text, pat) == 0


# ---------- B. denial ----------


@pytest.mark.parametrize(
    "text",
    [
        "Patient denies chest pain.",
        "Patient denied chest pain.",
        "Patient is denying chest pain.",
        "Patient will deny chest pain.",
        "Denial of chest pain.",
    ],
)
def test_denial(text):
    assert _kept(text, "chest pain") == 0


# ---------- C. contractions and transcription variants ----------


@pytest.mark.parametrize(
    "text, pat",
    [
        ("Patient doesn't smoke.", "smoke"),
        ("Patient doesn\u2019t smoke.", "smoke"),
        ("Patient doesnt smoke.", "smoke"),
        ("He wasn't febrile.", "febrile"),
        ("He wasnt febrile.", "febrile"),
        ("Patient didn't use heroin.", "heroin"),
        ("Patient didn\u2019t use heroin.", "heroin"),
        ("Patient didnt use heroin.", "heroin"),
        ("They don't drink.", "drink"),
        ("Patient isn't using.", "using"),
        ("Labs aren't back; hasn't used cocaine.", "cocaine"),
        ("Patient hadn't taken oxycodone.", "oxycodone"),
    ],
)
def test_contractions(text, pat):
    assert _kept(text, pat) == 0


@pytest.mark.parametrize("spelling", ["didn't", "didn\u2019t", "didnt", "DIDN'T", "Doesn't", "wasnt", "havent"])
def test_contraction_spellings_are_single_cues(spelling):
    assert [m[2].lower() for m in _cue_spans(f"he {spelling} use")] == [spelling.lower()]


# ---------- D. negative-result language ----------


def test_negative_result_language():
    assert _kept("Negative for influenza.", "influenza") == 0
    assert _kept("Influenza test was negative.", "influenza", side="right") == 0
    assert _kept("Influenza test was negative.", "influenza", side="left") == 1
    assert _kept("Flu: neg.", "flu", side="right") == 0
    assert _kept("Flu: neg", "flu", side="right") == 0
    assert _kept("Flu: NEG.", "flu", side="right") == 0


@pytest.mark.parametrize(
    "text, cue",
    [
        ("neg", "neg"),
        ("neg.", "neg"),
        ("negative", "negative"),
        ("negative.", "negative"),
        ("negative for", "negative"),
        ("Patient is neg.", "neg"),
        ("Patient is negative.", "negative"),
        ("Negative for influenza.", "Negative"),
        ("(neg)", "neg"),
        ("neg,", "neg"),
    ],
)
def test_neg_with_and_without_punctuation(text, cue):
    # The cue is the word alone; trailing punctuation is not part of the match. Presence is
    # what the gate needs, and previews highlight the word without its punctuation.
    spans = _cue_spans(text)
    assert [m[2] for m in spans] == [cue]


# ---------- E. scope and contrast ----------


def test_contrast_is_not_a_scope_boundary_by_default():
    # The window scan does not stop at "but": "No" is within 65 characters left of "chest
    # pain", so the match is negated. Documented limitation (see README).
    text = "No fever, but patient has chest pain."
    assert _kept(text, "fever") == 0
    assert _kept(text, "chest pain") == 0


# ---------- F. uncertainty is not negation ----------


@pytest.mark.parametrize(
    "text",
    [
        "Rule out pulmonary embolism.",
        "R/O pulmonary embolism.",
        "Pulmonary embolism is unlikely.",
        "Low suspicion for pulmonary embolism.",
        "Doubt pulmonary embolism.",
        "Possible pulmonary embolism.",
    ],
)
def test_uncertain_statements_stay_counted(text):
    assert _kept(text, "pulmonary embolism", side="both") == 1
    assert any(p.search(text) for p in helper.UNCERTAINTY_CUES)


def test_rule_out_versus_ruled_out():
    assert _kept("Rule out pulmonary embolism.", "pulmonary embolism", side="both") == 1
    assert _kept("Pulmonary embolism was ruled out.", "pulmonary embolism", side="right") == 0
    assert _kept("Pulmonary embolism was ruled out.", "pulmonary embolism", side="left") == 1
    assert not _has_cue("rule out")
    assert not _has_cue("ruling out")
    assert _has_cue("ruled out")
    assert _has_cue("ruled  out")


def test_uncertainty_cues_are_not_gate_cues():
    uncertain = ["rule out", "r/o", "unlikely", "doubt", "doubts", "doubted", "low suspicion for", "possible"]
    for phrase in uncertain:
        assert any(p.search(phrase) for p in helper.UNCERTAINTY_CUES), phrase
        assert not _has_cue(phrase), phrase


# ---------- G. historical and resolved are not negation ----------


def test_resolved_is_not_no():
    assert _kept("AKI resolved.", "AKI", side="both") == 1
    assert _kept("No AKI.", "AKI") == 0
    assert any(p.search("AKI resolved") for p in helper.HISTORICAL_CUES)
    assert not _has_cue("resolved")


def test_historical_cues_are_not_gate_cues():
    for phrase in ["resolved", "history of", "hx of", "h/o", "in remission", "former", "prior"]:
        assert not _has_cue(phrase), phrase


# ---------- real-world text ----------


@pytest.mark.parametrize("text", ["NO cocaine", "No cocaine", "no cocaine", "NEGATIVE cocaine", "Negative: cocaine"])
def test_capitalization(text):
    assert _kept(text, "cocaine") == 0


def test_punctuation_and_whitespace_around_cues():
    assert _kept("Denies: cocaine", "cocaine") == 0
    assert _kept("Denies, cocaine", "cocaine") == 0
    assert _kept("Cocaine? No.", "cocaine", side="right") == 0
    assert _kept("absence   of cocaine", "cocaine") == 0
    assert _kept("free\nof cocaine", "cocaine") == 0


@pytest.mark.parametrize(
    "text",
    [
        "neglect",
        "negligible",
        "negotiated",
        "casino",
        "nothing",
        "knot",
        "denim",
        "dent",
        "denial-free",  # "denial" is a cue; this checks "free" alone is not
        "cannot",
        "can't",
        "unable to",
        "won't",
    ],
)
def test_strings_that_must_not_be_cues(text):
    if text == "denial-free":
        assert [m[2] for m in _cue_spans(text)] == ["denial"]
    else:
        assert not _has_cue(text), text


def test_inability_is_not_negation_in_substance_use_text():
    # DSM-style criteria language must stay counted.
    assert _kept("Patient cannot cut down on cocaine use.", "cocaine") == 1
    assert _kept("Patient is unable to stop using cocaine.", "cocaine") == 1
    assert _kept("Patient can't control her cocaine use.", "cocaine") == 1


def test_no_multiword_triggers_are_needed_for_no_phrases():
    # "no" alone covers these, because the scan window contains the whole phrase.
    for phrase in ["no evidence of", "no signs of", "no history of", "no known history of", "no hx of"]:
        assert _kept(f"{phrase} cocaine use", "cocaine") == 0


# ---------- legacy versus current cues: the documented behavior change ----------

LEGACY = helper._compile_gate_terms(helper.LEGACY_NEGATION_GATE_CUES)
CURRENT = helper._compile_gate_terms(helper.NEGATION_GATE_CUES)


def _found(pats, text):
    return any(p.search(text) for p in pats)


@pytest.mark.parametrize(
    "text, legacy, current, why",
    [
        ("neglect", True, False, "substring 'neg' matched inside a word"),
        ("negligible", True, False, "substring 'neg' matched inside a word"),
        ("casino ", True, False, "substring 'no ' matched inside a word"),
        ("doubt", True, False, "uncertainty is no longer treated as negation"),
        ("deny", False, True, "'denie' missed the base form"),
        ("denying", False, True, "'denie' missed the -ing form"),
        ("No.", False, True, "'no ' required a trailing space"),
        ("not.", False, True, "'not ' required a trailing space"),
        ("didn\u2019t", False, True, "curly apostrophe"),
        ("didnt", False, True, "omitted apostrophe"),
        ("doesn't", False, True, "other auxiliaries were missing"),
        ("ruled out", False, True, "excluded diagnosis"),
        ("absence of", False, True, "new absence cue"),
        ("free of", False, True, "new absence cue"),
        ("no ", True, True, "unchanged"),
        ("denies", True, True, "unchanged"),
        ("negative for", True, True, "unchanged"),
        ("without", True, True, "unchanged"),
        ("never", True, True, "unchanged"),
    ],
)
def test_legacy_versus_current_cue_manifest(text, legacy, current, why):
    assert _found(LEGACY, text) is legacy, why
    assert _found(CURRENT, text) is current, why


def test_legacy_cues_can_be_restored(monkeypatch):
    monkeypatch.setattr(helper, "NEGATION_GATE_CUES", list(helper.LEGACY_NEGATION_GATE_CUES))
    assert _kept("Doubt cocaine use.", "cocaine") == 0
    assert _kept("Patient denying cocaine use.", "cocaine") == 1


def test_cue_groups_compose_the_gate_list():
    flat = [p for group in helper.NEGATION_CUE_GROUPS.values() for p in group]
    assert [p.pattern for p in helper.NEGATION_GATE_CUES] == flat
    assert helper.NEGATION_CUES is helper.NEGATION_GATE_CUES
    assert all(isinstance(p, re.Pattern) for p in helper.NEGATION_GATE_CUES)


def test_preview_highlights_the_cue_word_only(tmp_path):
    lib = _item("cocaine", preview=True)
    # The cue sits outside the 65-character negation window but inside the 120-character preview.
    filler = "Routine visit, vitals stable, patient comfortable and chatting in clinic today. " + "x" * 5 + " "
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["Flu: neg. " + filler + "Reports cocaine use."]})
    _, previews = sudregex.extract_df(
        df, lib, terms=["x"], preview_count=1, return_previews_df=True, preview_file=str(tmp_path / "p.txt")
    )
    assert "((neg))." in previews.at[0, "snippet_marked"]


# ---------- per-item negation_window ----------

UDS = "amphetamines - negative. cocaine - positive. oxycodone - negative."


def test_negation_window_lets_a_right_scan_stop_before_the_next_result():
    # Right scope alone still sees the oxycodone result 20 characters away.
    assert _kept(UDS, "cocaine", negation_scope="right") == 0
    # A short right window reads only the cocaine result.
    assert _kept(UDS, "cocaine", negation_scope="right", negation_window=12) == 1
    # The other two results are still honoured with the same settings.
    assert _kept(UDS, "amphetamines", negation_scope="right", negation_window=12) == 0
    assert _kept(UDS, "oxycodone", negation_scope="right", negation_window=12) == 0


def test_negation_window_applies_to_left_and_both():
    text = "No fever today. Cocaine positive."
    assert _kept(text, "cocaine") == 0
    assert _kept(text, "cocaine", negation_window=10) == 1
    assert _kept("Cocaine use was never confirmed", "cocaine", negation_scope="both", negation_window=12) == 1
    assert _kept("Cocaine use was never confirmed", "cocaine", negation_scope="both", negation_window=30) == 0


def test_negation_window_default_is_package_window():
    assert helper._resolve_negation_window("t", {}) == helper.WIN_NEGATION == 65
    assert helper._resolve_negation_window("t", {"negation_window": None}) == 65


@pytest.mark.parametrize("bad", [0, -5, 2.5, "65", True])
def test_negation_window_rejects_bad_values(bad):
    with pytest.raises(ValueError, match="'t'"):
        _kept("no cocaine", "cocaine", negation_window=bad)


def test_negation_window_ignored_when_negation_is_off():
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["no cocaine"]})
    lib = _item("cocaine", negation=False, negation_window=5)
    out = sudregex.extract_df(df, lib, terms=["x"])
    assert out.at[0, "c"] == 1 and "c_NEG" not in out.columns


def test_published_library_sets_no_window():
    assert not any("negation_window" in cfg for cfg in sudregex.pattern_library_abc.values())


# ---------- opt-in scope terminators ----------


@pytest.fixture
def clinical_terminators(monkeypatch):
    monkeypatch.setattr(helper, "NEGATION_SCOPE_TERMINATORS", list(helper.CLINICAL_SCOPE_TERMINATORS))


def test_terminators_are_off_by_default():
    assert helper.NEGATION_SCOPE_TERMINATORS == []
    assert helper._resolve_item_settings("t", _item("x")["t"], "left", True).gates.terminators == ()


def test_contrast_stops_the_scope_when_terminators_are_on(clinical_terminators):
    text = "No fever, but patient has chest pain."
    assert _kept(text, "fever") == 0
    assert _kept(text, "chest pain") == 1


@pytest.mark.parametrize(
    "text",
    [
        "No fever however chest pain is present.",
        "No fever although chest pain is present.",
        "No complaints except chest pain.",
        "No fever; nevertheless chest pain is present.",
        "No fever. Chest pain is present.",
    ],
)
def test_each_terminator_protects_the_concept_after_it(clinical_terminators, text):
    assert _kept(text, "chest pain") == 1


def test_terminators_stop_the_right_scan_too(clinical_terminators):
    assert _kept("Cocaine use denied.", "cocaine", side="right") == 0
    assert _kept("Cocaine use noted; denied heroin.", "cocaine", side="right") == 1
    assert _kept("Cocaine use noted but denied heroin.", "cocaine", side="right") == 1


def test_terminators_solve_the_lab_panel_without_a_window(clinical_terminators):
    assert _kept(UDS, "cocaine", negation_scope="right") == 1
    assert _kept(UDS, "cocaine", negation_scope="both") == 1
    assert _kept(UDS, "amphetamines", negation_scope="right") == 0
    assert _kept(UDS, "oxycodone", negation_scope="right") == 0


def test_terminators_keep_cues_in_the_same_clause(clinical_terminators):
    assert _kept("Pt. denies cocaine use.", "cocaine") == 0
    assert _kept("Flu: neg.", "flu", side="right") == 0
    assert _kept("No cocaine, no heroin.", "heroin") == 0


def test_terminator_helpers():
    t = helper.CLINICAL_SCOPE_TERMINATORS
    assert helper._after_last_terminator("no a. no b; c", t) == " c"
    assert helper._after_last_terminator("no cue here", t) == "no cue here"
    assert helper._before_first_terminator("a but b. c", t) == "a "
    assert helper._before_first_terminator("plain", t) == "plain"


# ---------- the validator applies the same rules ----------

from sudregex.validation import _helper_negation, validate_pattern_library  # noqa: E402


def _validate(lib, text, expected=1, **kwargs):
    examples = pd.DataFrame({"item_key": ["t"], "expected": [expected], "note_text": [text]})
    detailed, _ = validate_pattern_library(lib, examples, **kwargs)
    return int(detailed.at[0, "actual_match"])


def test_validator_uses_the_package_cues():
    assert _validate(_item("cocaine"), "Patient denying cocaine use.") == 0
    assert _validate(_item("cocaine"), "Doubt cocaine use.") == 1
    assert _validate(_item("cocaine"), "Patient didnt use cocaine.") == 0
    assert _validate(_item("cocaine"), "neglect noted; cocaine use.") == 1


def test_validator_honours_item_scope_and_window():
    assert _validate(_item("cocaine", negation_scope="right"), "Cocaine use denied.") == 0
    assert _validate(_item("cocaine", negation_scope="right"), "Denies cocaine use.") == 1
    assert _validate(_item("cocaine", negation_scope=False), "Denies cocaine use.") == 1
    assert _validate(_item("cocaine", negation_scope="right", negation_window=12), UDS) == 1
    assert _validate(_item("cocaine", negation_scope="right"), UDS) == 0


def test_validator_global_scope_and_legacy_window_key():
    assert _validate(_item("cocaine"), "Cocaine use denied.", negation_scope="right") == 0
    assert _validate(_item("cocaine"), "Cocaine use denied.") == 1
    # The validator-only `neg_left_span` key still works when `negation_window` is absent.
    assert _validate(_item("cocaine", neg_left_span=10), "No fever today. Cocaine positive.") == 1
    assert _validate(_item("cocaine"), "No fever today. Cocaine positive.") == 0


def test_validator_agrees_with_extract_df_on_every_section():
    cases = [
        ("No fever.", "fever", {}),
        ("Patient denies chest pain.", "chest pain", {}),
        ("Patient doesn\u2019t smoke.", "smoke", {}),
        ("Influenza test was negative.", "influenza", {"negation_scope": "right"}),
        ("Rule out pulmonary embolism.", "pulmonary embolism", {"negation_scope": "both"}),
        ("Pulmonary embolism was ruled out.", "pulmonary embolism", {"negation_scope": "right"}),
        ("AKI resolved.", "AKI", {"negation_scope": "both"}),
        (UDS, "cocaine", {"negation_scope": "right", "negation_window": 12}),
    ]
    for text, pat, extra in cases:
        assert _validate(_item(pat, **extra), text) == _kept(text, pat, **extra), (text, pat)


def test_helper_negation_is_left_only_with_package_cues():
    assert _helper_negation("denies cocaine", (7, 14)) is True
    assert _helper_negation("cocaine denied", (0, 7)) is False
    assert _helper_negation("doubt cocaine", (6, 13)) is False
