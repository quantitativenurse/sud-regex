"""
Review tests for per-match gating (0.2.0).

These pin down the properties a reviewer asked about: matches are judged by position,
not by their text; adjacent matches get their own windows; windows clamp at the note
boundaries; windows ignore sentence boundaries (a documented limitation); and a per-item
``negation_scope`` is applied to each match of that item separately.
"""

import re

import pandas as pd
import pytest

import sudregex
from sudregex import helper

FILLER = "The weather was mild and the clinic schedule stayed routine all week. " * 8


def _item(pat=r"cocaine", col="coc", *, opioid=False, negation=True, **extra):
    cfg = {"lab": "test", "pat": pat, "col_name": col, "opioid": opioid, "negation": negation, "preview": False}
    cfg.update(extra)
    return {"t": cfg}


def _run(lib, text, **kwargs):
    df = pd.DataFrame({"note_id": ["1"], "note_text": [text]})
    return sudregex.extract_df(df, lib, terms=["oxycodone"], **kwargs).iloc[0]


# ---------- identity of a match: position, not text ----------


def test_identical_matched_text_is_judged_per_occurrence():
    # Both matches are the string "cocaine"; only the first is negated. Keying matches by
    # their text would collapse them to one verdict (0 or 2). The answer is 1.
    row = _run(_item(), "Denies cocaine. " + FILLER + "Uses cocaine.")
    assert row["coc"] == 2
    assert row["coc_NEG"] == 1


def test_identical_text_failing_different_gates_counts_zero():
    # Same matched text twice: the first is negated, the second has no opioid term nearby.
    row = _run(_item(opioid=True), "Denies cocaine, on oxycodone. " + FILLER + "Chart mentions cocaine.")
    assert row["coc"] == 2
    assert row["coc_SUBSTANCE_MATCHED"] == 1
    assert row["coc_SUBSTANCE_MATCHED_NEG"] == 0


# ---------- adjacent and overlapping matches ----------


def test_adjacent_matches_use_their_own_windows():
    # The cue sits between two matches: it is right of the first and left of the second.
    row = _run(_item(), "cocaine no cocaine")
    assert row["coc"] == 2
    assert row["coc_NEG"] == 1
    assert _run(_item(), "cocaine no cocaine", negation_scope="right")["coc_NEG"] == 1
    assert _run(_item(), "cocaine no cocaine", negation_scope="both")["coc_NEG"] == 0


@pytest.mark.parametrize("pat", [r"coca|cocaine", r"(?=cocaine)", r"c\w+"])
def test_raw_count_equals_findall_for_alternation_and_zero_width_patterns(pat):
    text = "cocaine cocaine cocaine"
    row = _run(_item(pat=pat, negation=False), text)
    assert row["coc"] == len(re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE))


# ---------- note boundaries ----------


def test_match_at_note_start_has_empty_left_window():
    assert _run(_item(), "cocaine use reported")["coc_NEG"] == 1
    assert _run(_item(), "cocaine use reported", negation_scope="both")["coc_NEG"] == 1
    assert _run(_item(), "no cocaine")["coc_NEG"] == 0


def test_match_at_note_end_has_empty_right_window():
    assert _run(_item(), "patient takes cocaine", negation_scope="right")["coc_NEG"] == 1
    assert _run(_item(), "cocaine denied", negation_scope="right")["coc_NEG"] == 0
    assert _run(_item(), "cocaine", negation_scope="both")["coc_NEG"] == 1


def test_windows_clamp_at_note_bounds_for_every_gate():
    # Substance window and discharge window both extend past the note on one side.
    assert _run(_item(opioid=True), "oxycodone cocaine")["coc_SUBSTANCE_MATCHED_NEG"] == 1
    assert _run(_item(opioid=True), "cocaine oxycodone")["coc_SUBSTANCE_MATCHED_NEG"] == 1
    assert _run(_item(), "Discharge instructions: cocaine")["coc_NEG"] == 0
    assert _run(_item(), "cocaine. Discharge instructions follow.")["coc_NEG"] == 0


def test_empty_note_text_scores_zero():
    row = _run(_item(opioid=True), "")
    assert row["coc"] == 0 and row["coc_SUBSTANCE_MATCHED"] == 0 and row["coc_SUBSTANCE_MATCHED_NEG"] == 0


# ---------- sentence boundaries (documented limitation, not a bug) ----------


def test_windows_cross_sentence_boundaries():
    # Character windows do not stop at a period, so a cue in the previous sentence negates
    # the match. This is unchanged from earlier releases and is documented in the README.
    assert _run(_item(), "No pain today. Uses cocaine.")["coc_NEG"] == 0
    # A right-only scope on this item avoids the cue that belongs to the other result.
    assert _run(_item(negation_scope="right"), "Amphetamines negative, cocaine positive.")["coc_NEG"] == 1


# ---------- negation_scope with several matches of the same item ----------


def test_item_scope_is_applied_to_each_match():
    lib_right = _item(negation_scope="right")
    text = "Cocaine was denied. " + FILLER + "Patient uses cocaine."
    assert _run(lib_right, text)["coc_NEG"] == 1
    assert _run(_item(negation_scope="left"), text)["coc_NEG"] == 2
    assert _run(_item(negation_scope="both"), text)["coc_NEG"] == 1
    assert _run(_item(negation_scope=False), text)["coc_NEG"] == 2


def test_item_scope_with_three_matches_negated_on_different_sides():
    text = "Denies cocaine. " + FILLER + "Cocaine was denied. " + FILLER + "Uses cocaine."
    assert _run(_item(negation_scope="left"), text)["coc_NEG"] == 2
    assert _run(_item(negation_scope="right"), text)["coc_NEG"] == 2
    assert _run(_item(negation_scope="both"), text)["coc_NEG"] == 1


# ---------- output contract on degenerate input ----------


def test_zero_row_input_keeps_int64_count_columns():
    df = pd.DataFrame({"note_id": pd.Series([], dtype="string"), "note_text": pd.Series([], dtype="string")})
    out = sudregex.extract_df(df, _item(opioid=True), terms=["oxycodone"])
    assert list(out.columns) == ["note_id", "coc", "coc_SUBSTANCE_MATCHED", "coc_SUBSTANCE_MATCHED_NEG"]
    assert len(out) == 0
    for c in out.columns[1:]:
        assert out[c].dtype == "int64", (c, out[c].dtype)


def test_compiled_and_string_patterns_score_the_same():
    text = "Denies COCAINE. " + FILLER + "Uses Cocaine, on oxycodone."
    as_str = _run(_item(opioid=True), text)
    as_re = _run(_item(opioid=True, pat=re.compile(r"cocaine", re.IGNORECASE)), text)
    pd.testing.assert_series_equal(as_str, as_re)


def test_score_row_counts_are_monotone():
    gates = helper._GateConfig(
        pat=re.compile(r"cocaine", re.IGNORECASE),
        substance=helper._compile_gate_terms(["oxycodone"]),
        neg_pats=helper._compile_gate_terms(helper.NEGATION_GATE_CUES),
        side="left",
        dis_pats=helper._compile_gate_terms(helper.DISCHARGE_TERMS),
        fp_pats=[],
    )
    text = "Denies cocaine, on oxycodone. " + FILLER + "cocaine."
    assert helper._score_row(text, gates) == (2, 1, 0)
    verdicts = list(helper._iter_match_verdicts(text, gates))
    assert [(v[2], v[3]) for v in verdicts] == [(True, False), (False, False)]
    assert helper._passing_spans(text, gates) == []


def test_shared_column_naming_matches_spark_and_extract_output():
    from sudregex import spark as sudregex_spark

    lib = sudregex.pattern_library_abc
    assert sudregex_spark.expected_count_columns(lib) == helper.expected_count_columns(lib)
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["Routine visit."]})
    out = sudregex.extract_df(df, lib, termslist=sudregex.default_termslist, terms_active="opioid_terms")
    assert list(out.columns) == ["note_id", *helper.expected_count_columns(lib)]


# ---------- previews only show matches that passed the gates ----------


def test_preview_shows_a_match_that_passed_every_gate(tmp_path):
    lib = _item(opioid=True, preview=True)
    first = "Denies cocaine use, on oxycodone. "
    text = first + FILLER + "Reports cocaine use, on oxycodone."
    df = pd.DataFrame({"note_id": ["1"], "note_text": [text]})
    out, previews = sudregex.extract_df(
        df,
        lib,
        terms=["oxycodone"],
        preview_count=5,
        return_previews_df=True,
        preview_file=str(tmp_path / "previews.txt"),
    )
    assert out.at[0, "coc_SUBSTANCE_MATCHED_NEG"] == 1
    assert len(previews) == 1
    assert previews.at[0, "span_start"] >= len(first)
    assert "Reports [[cocaine]]" in previews.at[0, "snippet_marked"]
    assert "Denies [[cocaine]]" not in previews.at[0, "snippet_marked"]


def test_write_previews_without_filter_still_shows_first_match(tmp_path):
    text = "Denies cocaine use. " + FILLER + "Reports cocaine use."
    df = pd.DataFrame({"note_id": ["1"], "note_text": [text], "mask": [1]})
    rows = helper.write_previews_for_item(df, "t", r"cocaine", "mask", n_notes=None)
    assert len(rows) == 1 and rows[0]["span_start"] == len("Denies ")
    rows = helper.write_previews_for_item(df, "t", r"cocaine", "mask", n_notes=None, match_filter=lambda t: [])
    assert rows == []


def test_package_root_has_no_private_apply_series():
    assert not hasattr(sudregex, "_apply_series")
    df = pd.DataFrame({"note_id": ["1"], "note_text": ["line one$+$line two cocaine"]})
    assert sudregex.extract_df(df, _item(), terms=["x"]).at[0, "coc"] == 1
