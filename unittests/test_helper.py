import re

import pandas as pd

from sudregex import helper


def test_remove_line_break_simple():
    txt = "Hello$+$\nWorld"
    out = helper.remove_line_break(txt)
    assert out == "Hello World"


def test_remove_line_break_multiple_markers():
    txt = "A+++B and C$$$D"
    out = helper.remove_line_break(txt, break_markers=["+++", r"\$\$\$"])
    assert out == "A B and C D"


def test_remove_tobacco_mentions():
    txt = "Patient denies Tobacco use: None today."
    masked = helper.remove_tobacco_mentions(txt)
    assert "Tobacco: [Redacted]" in masked


def test_set_terms_and_regex_search_file_basic(tmp_path):
    # prepare small df
    df = pd.DataFrame({"note_id": [1, 2], "note_text": ["foo bar baz", "no match here"]})
    metadata = df[["note_id"]].copy()
    # compile a simple pat
    pat = re.compile("foo")
    helper.set_terms(["bar"])  # for substance detection later
    df_searched = helper.regex_search_file(pat, "foo", df.copy(), metadata, preview=False)
    # note 1 has 1 match, note 2 has 0
    assert df_searched.loc[df_searched.note_id == 1, "foo"].iloc[0] == 1
    assert df_searched.loc[df_searched.note_id == 2, "foo"].iloc[0] == 0


def test_check_for_substance_detects_term():
    # build df_searched with an initial 'foo' count >0
    df = pd.DataFrame({"note_id": [1], "note_text": ["abc foo bar xyz"], "foo": [1]})
    helper.set_terms(["bar"])
    out = helper.check_for_substance(re.compile("foo"), "foo", "foo_SUB", df.copy())
    # should find 'bar' near 'foo'
    assert out.loc[out.note_id == 1, "foo_SUB"].iloc[0] == 1


def test_check_negation_flags_negation():
    df = pd.DataFrame({"note_id": [1], "note_text": ["patient has not foo now"], "foo": [1]})
    out = helper.check_negation(re.compile("foo"), "foo", "foo_NEG", df.copy())
    # 'not' before foo → marked as negated → 0
    assert out.loc[out.note_id == 1, "foo_NEG"].iloc[0] == 0


def test_check_common_false_positives_filters_family_history():
    df = pd.DataFrame({"note_id": [1], "note_text": ["family history of asthma"], "hist": [1]})
    out = helper.check_common_false_positives(re.compile("history"), df.copy(), "hist", ["family"])
    # 'family history' should be filtered, so hist → 0
    assert out.loc[out.note_id == 1, "hist"].iloc[0] == 0


def test_discharge_instructions_removes_discharge_context():
    df = pd.DataFrame(
        {
            "note_id": [1, 2],
            "note_text": ["discharge instructions foo", "no discharge here foo"],
            "foo": [1, 1],
        }
    )
    out = helper.discharge_instructions(re.compile("foo"), df.copy(), "foo")
    # first row in discharge context → 0, second remains 1
    assert out.loc[out.note_id == 1, "foo"].iloc[0] == 0
    assert out.loc[out.note_id == 2, "foo"].iloc[0] == 1


# ---------- Helpers ----------
def _df(rows):
    return pd.DataFrame(rows)


# ---------- Highlighting utilities ----------
def test_first_span_picks_first_match():
    pats = [re.compile(r"no"), re.compile(r"never")]
    span = helper._first_span(pats, "xx no foo never")
    assert span == (3, 5)  # "no"


def test_apply_style_brackets_and_html_no_crash():
    s = "abc def ghi"
    styled = helper._apply_style(s, (4, 7), style="brackets", kind="hit")
    assert "[[" in styled and "]]" in styled
    styled_html = helper._apply_style(s, (4, 7), style="html", kind="neg")
    assert "<mark" in styled_html


def test_highlight_snippet_marks_hit_and_optional_terms():
    snippet = "… patient foo not present …"
    # compute spans robustly
    hit_m = re.search(r"\bfoo\b", snippet)
    neg_m = re.search(r"\bnot\b", snippet)
    assert hit_m and neg_m
    rel_hit = hit_m.span()
    neg_span = neg_m.span()

    out = helper._highlight_snippet(snippet, rel_hit, None, neg_span, style="brackets")
    # expect hit [[foo]] and neg ((not))
    assert "[[" in out and "]]" in out
    assert "((" in out and "))" in out


# ---------- gate_by_terms: require vs exclude ----------
def test_gate_by_terms_require_keeps_when_term_in_window():
    df = _df([{"note_id": "1", "note_text": "xx foo bar yy", "hit": 1}])
    out = helper.gate_by_terms(
        df,
        pat=re.compile("foo"),
        in_col="hit",
        out_col="out",
        terms=["bar"],
        left_chars=10,
        right_chars=10,
        policy="require",
    )
    assert out.loc[out.note_id == "1", "out"].iloc[0] == 1


def test_gate_by_terms_exclude_drops_when_term_in_window():
    df = _df([{"note_id": "1", "note_text": "xx foo bar yy", "hit": 1}])
    out = helper.gate_by_terms(
        df,
        pat=re.compile("foo"),
        in_col="hit",
        out_col="out",
        terms=["bar"],
        left_chars=10,
        right_chars=10,
        policy="exclude",
    )
    assert out.loc[out.note_id == "1", "out"].iloc[0] == 0


def test_gate_by_terms_missing_in_col_yields_zero():
    df = _df([{"note_id": "1", "note_text": "foo bar"}])  # no 'hit' col
    out = helper.gate_by_terms(
        df,
        pat=re.compile("foo"),
        in_col="hit",
        out_col="out",
        terms=["bar"],
        left_chars=10,
        right_chars=10,
        policy="require",
    )
    assert out.loc[out.note_id == "1", "out"].iloc[0] == 0


# ---------- Negation scope (left/right/both) ----------
def test_check_negation_left_only():
    df = _df([{"note_id": "1", "note_text": "not foo here", "foo": 1}])
    # left scope sees "not" → remove
    out = helper.check_negation(re.compile("foo"), "foo", "foo_NEG", df.copy(), span=50, side="left")
    assert out.loc[out.note_id == "1", "foo_NEG"].iloc[0] == 0


def test_check_negation_right_only():
    df = _df([{"note_id": "1", "note_text": "foo not used", "foo": 1}])
    # right scope sees "not" → remove
    out = helper.check_negation(re.compile("foo"), "foo", "foo_NEG", df.copy(), span=50, side="right")
    assert out.loc[out.note_id == "1", "foo_NEG"].iloc[0] == 0


def test_check_negation_both_sides():
    df = _df([{"note_id": "1", "note_text": "not foo not", "foo": 1}])
    out = helper.check_negation(re.compile("foo"), "foo", "foo_NEG", df.copy(), span=50, side="both")
    assert out.loc[out.note_id == "1", "foo_NEG"].iloc[0] == 0


# ---------- write_previews_for_item returns rows & honors mask ----------
def test_write_previews_respects_final_mask_and_returns_rows(tmp_path):
    helper.set_terms(["bar"])
    df = _df(
        [
            {"note_id": "A1", "note_text": "foo bar ok", "BASE": 1},
            {"note_id": "A2", "note_text": "foo not bar", "BASE": 1},
            {"note_id": "A3", "note_text": "no foo here", "BASE": 1},
        ]
    )
    # Gate to create FINAL mask = require bar + no negation
    df = helper.check_for_substance(re.compile("foo"), "BASE", "BASE_SUB", df, span=100)
    df = helper.check_negation(re.compile("foo"), "BASE_SUB", "FINAL", df, span=100, side="both")
    # A1 passes; A2 has "not" near foo; A3 no 'bar' → only A1 should preview
    rows = helper.write_previews_for_item(
        df_searched=df,
        item_key="test",
        pat=re.compile("foo"),
        mask_col="FINAL",
        n_notes=None,
        left_chars=50,
        right_chars=50,
        csv_path=None,
        outfile=None,
        highlight=True,
        highlight_style="brackets",
    )
    assert isinstance(rows, list) and len(rows) == 1
    assert rows[0]["note_id"] == "A1"
    assert "snippet" in rows[0] and "snippet_marked" in rows[0]


def test_write_previews_appends_csv_and_text(tmp_path):
    csv_path = tmp_path / "prev.csv"
    txt_path = tmp_path / "prev.txt"
    helper.set_terms(["bar"])
    df = _df([{"note_id": "A1", "note_text": "foo bar ok", "BASE": 1}])
    rows = helper.write_previews_for_item(
        df,
        "it",
        re.compile("foo"),
        "BASE",
        n_notes=1,
        left_chars=20,
        right_chars=20,
        csv_path=str(csv_path),
        outfile=str(txt_path),
        highlight=True,
    )
    assert csv_path.exists()
    assert txt_path.exists()
    # and still returns memory rows
    assert len(rows) == 1


# ---------- regex_search_file preview flag path ----------
def test_regex_search_file_preview_flag_controls_note_text():
    df = _df([{"note_id": "1", "note_text": "hello foo"}])
    meta = df[["note_id"]].copy()
    out_true = helper.regex_search_file(re.compile("foo"), "foo_col", df.copy(), meta, preview=True)
    out_false = helper.regex_search_file(re.compile("foo"), "foo_col", df.copy(), meta, preview=False)
    assert "note_text" in out_true.columns
    assert "note_text" not in out_false.columns
