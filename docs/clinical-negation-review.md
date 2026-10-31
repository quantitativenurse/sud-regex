# Clinical negation: review, changes, and open decisions

Branch: `feat/clinical-negation`, stacked on `fix/per-match-gating`. Every note quoted here
is synthetic. The numbered sections follow the review brief.

## What the package actually had

Three separate cue lists existed, and the one that mattered was the crudest:

| Where | Form | Used for |
|---|---|---|
| `helper.NEGATION_GATE_CUES` | 10 literal substrings: `"no "`, `"not "`, `"denie"`, `"denial"`, `"doubt"`, `"never"`, `"negative"`, `"without"`, `"neg"`, `"didn't"` | the gate that decides counts |
| `helper.NEGATION_CUES` | the 8 regexes quoted in the brief | preview highlighting only |
| `validation._HELPER_NEG_TERMS` | a copy of the 10 substrings | `validate_pattern_library` |

Consequences of the substring list: `"neg"` matched inside "neglect", "negligible" and
"negotiated"; `"no "` matched inside "casino " and missed "No." and "no,"; `"denie"` matched
"denies" and "denied" but missed "deny" and "denying"; `"didn't"` covered one spelling of one
auxiliary; `"doubt"` treated uncertainty as negation. The regexes in the brief were never used
for gating.

Scope is a character window on the side(s) the item or global `negation_scope` selects
(65 characters by default). There were no contrast or sentence terminators.

## 1. Concise, readable patterns

The gate now uses `helper.NEGATION_CUE_GROUPS`, five small groups of compiled
case-insensitive patterns, flattened into `helper.NEGATION_GATE_CUES`:

```python
"direct":          [r"\bno\b", r"\bnot\b", r"\bnever\b", r"\bwithout\b"]
"denial":          [r"\bden(?:y|ies|ied|ying|ial)\b"]
"negative_result": [r"\bneg(?:ative)?\b", r"\bruled\s+out\b"]
"contraction":     [r"\b(?:do|does|did|is|are|was|were|has|have|had)n['’]?t\b"]
"absence":         [r"\babsen(?:t|ce\s+of)\b", r"\bfree\s+of\b"]
```

Non-capturing groups throughout. The four direct cues stay as four patterns rather than one
alternation because each is read and tested on its own. The contraction group is one pattern
on purpose: ten auxiliaries times three apostrophe spellings would otherwise be thirty lines.

## 2. Coverage decisions

| Candidate | Decision | Reason |
|---|---|---|
| no, not, never, without | cue | definitive negation; word-bounded now |
| deny / denies / denied / denying | cue | `"denie"` missed two forms |
| denial | cue (kept) | shipped since 0.1.x; "in denial" is a known false-positive risk (decision 3) |
| negative, negative for, neg, neg. | cue | one pattern, `\bneg(?:ative)?\b` |
| ruled out | cue | excluded diagnosis; see section 3 |
| didn't, doesn't, don't, isn't, aren't, wasn't, weren't, hasn't, haven't, hadn't | cue | each with `'`, `’`, or no apostrophe |
| can't, cant, cannot, unable to | **not a cue** | inability, not absence. "can't cut down", "cannot control use", "unable to stop" are DSM-style criteria language in substance-use notes; a left scan would erase them |
| won't | not a cue | refusal or future, and not in the brief |
| absence of, absent, free of | cue | "absent" alone carries some risk (decision 4) |
| no evidence of, no signs of, no history of, no known history of, no diagnosis of, no hx of | not listed | `no` already matches inside the window; tested in `test_no_multiword_triggers_are_needed_for_no_phrases` |
| doubt, doubts, doubted | **removed from cues** | uncertainty; see section 4 |

## 3. "rule out" versus "ruled out"

Before: neither was a cue, so "Rule out PE" and "PE was ruled out" were both counted as
present. The second is wrong. Now `\bruled\s+out\b` is a cue and `\brule\s+out\b` is in
`UNCERTAINTY_CUES`. "ruled out" normally follows the concept, so it takes effect with a right
or both scope. Tests: `test_rule_out_versus_ruled_out`.

## 4. Uncertainty is not negation

The architecture is binary: a match is counted in the final column or it is not. There is no
uncertainty state, and I did not add one. "doubt" was a gate cue in 0.1.8 and is no longer,
because a doubted finding is not an absent one. The uncertainty vocabulary is collected in
`helper.UNCERTAINTY_CUES` and is tested to be absent from the gate.

What an uncertainty category would require, if wanted later: `_iter_match_verdicts` would yield
a state (present, negated, uncertain, historical) instead of a pass/fail, and
`_assemble_item_columns` would emit an extra column per item (for example
`<col>_UNCERTAIN`). That changes the output column set from 69 to up to 89 for the ABC library,
which the lab's downstream aggregate-and-score code keys on. It is a deliberate schema change,
so it is a decision, not something to slip into this branch.

## 5. Historical and resolved concepts

No mechanism exists. "resolved" is not a cue, "AKI resolved" stays counted, "No AKI" is
rejected. The vocabulary is in `helper.HISTORICAL_CUES`. One overlap to know about: "no longer
uses cocaine" is rejected through `no`, which for the ABC's "since last visit" framing is the
right outcome.

## 6. `neg.` punctuation

`\bneg(?:ative)?\.?\b` behaves as the brief suspected: on "neg. " the engine matches `neg`,
tries the period, fails the following `\b` (period then space is not a boundary), backtracks,
and matches without the period. It never consumed the period except before a word character
("neg.flu"). The new pattern is `\bneg(?:ative)?\b`. The gate only needs presence, and
previews highlight the word without its punctuation, so not consuming the period is the right
contract. Tested on all eight strings from the brief plus "(neg)" and "neg,".

## 7. Real-world text

- Apostrophes: `['’]?` accepts straight, curly, or omitted.
- Capitalization: `re.IGNORECASE`.
- Punctuation after a cue: word boundaries, so "denies:", "No.", "neg," all match.
- Extra whitespace and line breaks: `\s+` inside multiword cues; `remove_line_break` collapses
  whitespace upstream anyway.
- Nothing matches inside another word: "neglect", "negligible", "negotiated", "casino",
  "nothing", "knot", "denim" and "dent" are tested non-matches.

## 8. Scope

The window mechanism is unchanged and the new cues sit inside it. Because no terminators
existed, "No fever, but patient has chest pain" negates "chest pain" by default, and that is
pinned by `test_contrast_is_not_a_scope_boundary_by_default`.

Added, **opt-in**: `helper.NEGATION_SCOPE_TERMINATORS` (default `[]`). With
`helper.CLINICAL_SCOPE_TERMINATORS` (but, however, although, except, nevertheless, period,
semicolon) the left window keeps only the text after its last terminator and the right window
only the text before its first. That fixes the contrast case, the previous-sentence case, and
the drug-screen case without a per-item window. Abbreviation periods ("Pt. denies") are safe
because the cue follows the period. It is off by default so output is unchanged (decision 5).

## 9. False-positive control, per cue

| Cue | Uncertainty? | Historical? | Other concept nearby? | Inside another word? |
|---|---|---|---|---|
| no / not / never / without | no | "no longer" is historical-ish but correct for current use | yes, the window problem; mitigated by scope, window, terminators | no, word-bounded |
| deny family | no | no | yes, same window problem | "denim", "dent" excluded |
| denial | no | no | "in denial" is a psychiatric state, not a negation | no |
| neg / negative | no | no | yes: lab panels; mitigated by right scope and `negation_window` | "neglect" excluded |
| ruled out | no (that is the point) | no | low | no |
| contractions | no | "hasn't used in two years" is historical; counted as not current, consistent with "has not" | same window problem | "cant", "wont" never included |
| absent / absence of / free of | no | "drug free for a year" is resolution, not absence; "free of" alone is rarer than "free" | "absent from clinic" within 65 characters of a concept | no |

## 10. Tests

`unittests/test_clinical_negation.py`, 93 tests: sections A to G from the brief, capitalization
and punctuation, strings that must not be cues, inability stays counted, no multiword "no"
triggers needed, a legacy-versus-current manifest that names every behavior change, legacy cues
restorable, per-item `negation_window`, terminators, and validator agreement with extraction on
every section.

## 11. Regressions

Suite before: 163 passed, 9 skipped. After: 282 passed, 9 skipped. No existing test was
changed except `test_parity_legacy.py`, which now freezes the 0.1.8 cue list for both paths so
that it keeps measuring the per-match change alone. No existing test encoded clinically wrong
behavior; the one behavioral change an existing test could have caught ("doubt" as negation)
had no test.

Impact on the synthetic ABC corpus used by the parity test: zero cells differ between legacy
and current cues, because that corpus contains none of the changed words. The manifest test
lists each changed word explicitly. On real notes the direction of change is mixed: fewer
rejections from "neglect"/"negligible"/"casino"/"doubt", more rejections from "deny",
"denying", "No.", curly or omitted apostrophes, "ruled out" and the absence cues.

## Per-item direction and window

Per-item `negation_scope` was already on the base branch. This branch adds per-item
`negation_window` so the drug-screen case from the brief works as specified:

```
"amphetamines - negative. cocaine - positive. oxycodone - negative."
cocaine item: negation_scope="right", negation_window=12  ->  counted
cocaine item: negation_scope="right" (window 65)          ->  rejected (sees the oxycodone result)
```

The validator honours both keys and a new `negation_scope` argument, and the CLI passes
`--negation-scope` through to `--validate`.

## Open decisions

1. **New cues on by default.** Recommend yes: 0.2.0 already requires regenerating scores, so
   one regeneration covers both. The alternative is default legacy cues with the new set
   opt-in, which leaves "neglect" negating matches for everyone who does not read the docs.
2. **"doubt" removed.** Recommend yes per the brief's own framework. If the lab wants doubted
   mentions rejected, `helper.NEGATION_GATE_CUES += [UNCERTAINTY_CUES[3]]` restores it.
3. **"denial" kept.** "in denial about addiction" near an ABC item 10 match ("worried about
   addiction") would reject a true positive. Recommend keeping for continuity and revisiting
   with real-note counts.
4. **Bare "absent" kept.** Lower confidence than the rest. Recommend keeping; drop it if
   previews show "absent from" rejections.
5. **Terminators off by default.** Turning them on changes scores more than the cue change
   does (every cue in a previous sentence stops applying). Recommend on in the next minor
   after a run on real notes, with the period terminator reviewed against abbreviation-heavy
   text.
6. **Validator window excludes the match text** (was `text[left:end]`, now `text[left:start]`),
   matching extraction. Items whose pattern contains a cue (item 17 "not interested in
   rehab", item 18a "no relief") have `negation: False`, so nothing in the ABC library is
   affected.
7. **Version.** If this merges together with `fix/per-match-gating`, it is part of 0.2.0.
   If the base merges and releases first, this is 0.3.0: it changes scores again.
8. **Uncertainty and historical states as columns.** Not done; schema change (section 4).
9. **Legacy wrappers.** `check_negation` and `gate_by_cues` use the new cues but do not apply
   terminators; only the per-match scorer and the validator do. Fine if they are deprecated.
10. **Module-level switches** (`NEGATION_GATE_CUES`, `NEGATION_SCOPE_TERMINATORS`) follow the
    package's existing convention (`PRINT`, `TERMS_LIST`). Keyword arguments on `extract_df`
    would be cleaner for notebooks; recommend adding them when the deprecations land.
