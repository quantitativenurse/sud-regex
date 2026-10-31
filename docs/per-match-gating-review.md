# Per-match gating: review, parity results, and open decisions

Branch: `fix/per-match-gating`. Scope: review of the per-match gating fix (Task 1), the
behavior-preserving refactor (Task 2), Spark path review, version recommendation, and the
decisions that need a maintainer. Nothing in this document is patient data; every note quoted
is synthetic.

## 1. Verdict on the fix

The per-match implementation is correct and complete for the gates sudregex applies. I found
one defect (zero-row input produced object-dtype count columns) and fixed it. Everything else
checked out, and each check is now pinned by a test in `unittests/test_gating_review.py`.

| Question | Finding | Test |
|---|---|---|
| Span or text identity? | Neither key is used. Each match is judged in place during one `finditer` pass, so two matches with identical text get independent verdicts. | `test_identical_matched_text_is_judged_per_occurrence`, `test_identical_text_failing_different_gates_counts_zero` |
| Overlapping or adjacent matches | `finditer` yields the same non-overlapping matches `findall` counts, so the raw column is unchanged. Adjacent matches each get their own window; a cue between them negates whichever side the scope points at. | `test_adjacent_matches_use_their_own_windows`, `test_raw_count_equals_findall_for_alternation_and_zero_width_patterns` |
| Note start and end | Windows are plain slices, so they clamp at both ends for every gate. | `test_match_at_note_start_has_empty_left_window`, `test_match_at_note_end_has_empty_right_window`, `test_windows_clamp_at_note_bounds_for_every_gate` |
| Windows crossing sentence boundaries | They do, exactly as in 0.1.8. This is a documented limitation, not a regression. | `test_windows_cross_sentence_boundaries` |
| `negation_scope` with several matches of one item | Applied per match. Three matches negated on different sides give 2 / 2 / 1 under left / right / both. | `test_item_scope_is_applied_to_each_match`, `test_item_scope_with_three_matches_negated_on_different_sides` |
| Zero-row input | Was object dtype; now int64 for every count column. | `test_zero_row_input_keeps_int64_count_columns` |

Gate order does not affect the final count, because a match must pass every gate. The
substance count is taken before the other gates, as the `_SUBSTANCE_MATCHED` column requires.

## 2. Per-match gating versus span tracking

The Jira proposal was span tracking: the first gate yields surviving spans and each later gate
filters that list. The implemented approach evaluates each match against every gate in one pass.
They produce identical counts: both count exactly the matches that pass every gate.

I recommend keeping the per-match implementation.

- One scan of the note per item instead of up to five, and no intermediate lists.
- No shared state between gates, so there is nothing to get out of sync.
- `_iter_match_verdicts` already yields per-match verdicts, which is what span tracking would
  have produced. The gated previews use it, and an "explain this count" mode would be a few lines.

Span tracking would only be preferable if gates needed to see each other's output, for example
a gate whose window depends on which other matches survived. No such gate exists.

## 3. Parity results

`unittests/test_parity_legacy.py` runs the 0.1.8 orchestration (kept verbatim in
`unittests/legacy_gating.py`, driving the `check_*` functions that still ship) and the new
scorer through the public `extract_df` on the same synthetic notes, under every global
`negation_scope` and both discharge settings. A five-item library covers each column-naming
branch; the published ABC library runs with the packaged opioid vocabulary.

Structural parity holds in every run: identical column names, suffixes, order, dtypes and row
order. Every non-divergent cell is identical.

| Divergences | Count |
|---|---|
| Total divergent cells (12 runs) | 2,384 |
| Increases | 0 |
| On single-match notes | 0 |
| Unexplained | 0 |

Every divergence is a decrease equal to the number of matches the old chain re-counted after an
earlier gate rejected them. Breakdown by the gate(s) those matches failed:

| Gate(s) failed by re-counted matches | Cells |
|---|---|
| negation | 1,466 |
| discharge | 296 |
| negation + substance | 248 |
| substance | 196 |
| discharge + negation | 94 |
| discharge + substance | 50 |
| discharge + negation + substance | 34 |

In the ABC runs the divergent columns were `opioid_mention_SUBSTANCE_MATCHED_NEG` (540),
`illicit_drug_use_SUBSTANCE_MATCHED_NEG` (209), `problematic_alcohol_use_NEG` (16),
`multiple_providers_primary` and `_alt` (12 each) and `sedated_or_confused` (6). Item 16
dominates because the opioid vocabulary makes it match often. The full list is written to
`unittests/artifacts/parity_divergences.csv` on every run.

Two deviations from 0.1.8 are not covered by the parity test and are deliberate:

1. Zero-row input: count columns are int64. 0.1.8 returned a string-typed raw column and int64
   gate columns.
2. Direct callers of `helper.regex_extract` that pass a `metadata` frame with duplicate
   `note_id` values: 0.1.8 multiplied rows through a double merge; the refactor merges once.
   `extract` and `extract_df` always pass unique ids and are unaffected.

## 4. Spark path (read-only review)

The Databricks path calls `extract_df` per partition, so it inherits the fix and the per-item
`negation_scope` key automatically. Things I checked by reading:

- Schema: `spark.expected_count_columns` now re-exports `helper.expected_count_columns`, so the
  declared schema cannot drift from the local columns. Count columns are `LongType`, matching
  int64.
- Aggregation: `concat_ws(" ", collect_list(...))` then `remove_line_break` on the worker gives
  the same text as the local "normalize each row, then join with a space" because the
  normalizer collapses runs of whitespace. With `remove_linebreaks=False` both paths join with a
  single space.
- Empty partitions yield an all-object empty frame; Arrow converts that to the declared types.
- `parallel=False` is forced on workers. `set_terms` runs inside each worker call, so the
  vocabulary is set per batch and never shared across executors.

Parity risks worth a real test:

- `F.first(..., ignorenulls=True)` for person and extra id columns versus pandas
  `groupby().first()`: the same for ordinary data, but Spark's `first` is non-deterministic
  across partitions when a note's rows carry different values.
- Any item with `preview=True` makes `write_previews_for_item` open `note_previews.txt` in the
  worker's working directory even with `preview_count=0` (the file is opened before the empty
  sample is checked). Harmless on Databricks but untidy; the same happens locally.
- Divergences in the count columns between local and Spark would come only from text
  normalization or aggregation, not from gating, because the gating code is shared.

How to test it:

1. `unittests/test_spark.py::TestRunDatabricks::test_distributed_output_matches_local_extract_df`
   already asserts row-for-row equality with the local path. It needs `pip install pyspark
   pyarrow` and a JDK; it ran on nothing in this environment. A GitHub Actions job with
   `actions/setup-java` (Temurin 17) plus `pip install -e .[spark]` would run it on every PR.
2. Make the partition function testable without a JVM: move the body of `_map_partitions` to a
   module-level `_extract_partition(pdf, ...)` and feed it pandas frames in a test. That covers
   the schema alignment and dtype coercion, which is where most Spark-only bugs live.
3. On Databricks, run `run_sudregex(notes, lib, environment="databricks", spark=spark)` and
   `extract_df(notes.toPandas(), lib, ...)` on a synthetic sample, then
   `pd.testing.assert_frame_equal` after `.toPandas().sort_values("note_id")`.

## 5. Deprecation proposal (not implemented, awaiting your decision)

The old gate functions (`gate_by_terms`, `gate_by_cues`, `gate_by_cues_left`,
`check_for_substance`, `check_negation`, `check_common_false_positives`,
`discharge_instructions`, and `regex_search_file`) are no longer called by `regex_extract`.
Chaining them reproduces the over-count, so leaving them silently available is a trap for
external users who built their own pipelines on them. Proposal:

```python
import warnings

def _deprecated(name, replacement="regex_extract"):
    warnings.warn(
        f"sudregex.helper.{name} is deprecated since 0.2.0 and will be removed in 0.3.0. "
        f"Chaining the old gates re-counts matches rejected by earlier gates; use {replacement}.",
        DeprecationWarning,
        stacklevel=3,
    )
```

called at the top of each listed function, kept in `__all__`, and listed in the changelog. The
parity oracle in `unittests/legacy_gating.py` depends on them; the test can filter the warning,
and the oracle should be vendored into the test module before the functions are removed.

## 6. Version recommendation and changelog

Recommend **0.2.0** (already set on this branch). It is a minor bump under pre-1.0 semver
because the final-column counts change for multi-match notes and a new pattern-library key is
added. A patch release (0.1.9) would hide a score-changing fix from anyone scanning version
numbers; 1.0.0 would promise API stability the deprecations above do not yet deliver.

The changelog entry is in `README.md` under "0.2.0". The sentence that matters for the lab is
there: scores computed with earlier versions may be inflated for multi-match notes, and any
model or analysis built on 0.1.x output should be regenerated with 0.2.0 before comparison.

## 7. Open decisions

Each item has my recommendation; none is resolved in code unless marked.

1. **Deprecation warnings** for the old gate functions (section 5). Recommend yes, in 0.2.0.
2. **Zero-row dtype** is int64 everywhere (done). Confirm that is acceptable as the one
   dtype deviation from 0.1.8. Recommend keep.
3. **Gated previews are on by default** (done, column output unaffected). A preview row now shows
   a match that was counted rather than the first match in the note. Alternative: opt-in flag.
   Recommend keep on; the README already promised gated previews.
4. **Substance-only items receive no discharge or false-positive pruning.** Items 6, 14, 15,
   17 and 18a set `opioid: True, negation: False`, and `should_prune` is false for them, so the
   `common_fp` lists on items 6, 15 and 18a are never applied. Preserved for parity. Recommend a
   separate ticket; changing it alters scores.
5. **Item 15 `common_fp: [r"[0-9]"]`** is matched literally (every term is `re.escape`d), so it
   never fires. Preserved for parity. Same ticket as item 4.
6. **Items 7a and 7b are identical patterns** producing duplicate columns. Pre-existing; the
   downstream code expects 20 items, so leave as is.
7. **`validation.validate_pattern_library` ignores per-item `negation_scope`**: its default
   negation check is left-only, 65 characters, and includes the match text in the window.
   Validation results can disagree with `extract_df` for right or both items. Recommend a
   follow-up to route validation through `_iter_match_verdicts`.
8. **The README changelog states "100,000-note sample ... 193 notes ... about 15% fewer".** That
   number came with the fix commit and I could not verify it. Confirm it is from your own run
   and that you are comfortable publishing it.
9. **Parity test runtime** is about 30 seconds (12 full-library comparisons). Recommend keeping
   it in the default suite; alternatively mark it `slow` and run it in CI only.
10. **CI only lints.** No workflow runs `pytest`. Recommend adding a test job, and a Spark job
    as in section 4.
11. **`regex_extract` direct callers with duplicate `note_id` in `metadata`** no longer get
    multiplied rows (section 3). Recommend keep.
12. **`note_previews.txt` is created whenever an item has `preview=True`,** even with
    `preview_count=0`, locally and on Spark workers. Recommend a follow-up fix (open the file
    only when a sample is non-empty).
13. **Sentence-boundary and lab-result limitation** is documented, not solved. Recommend leaving
    it unless a sentence-aware window becomes a research need.

## 8. Commits on this branch

```
Add review tests and a legacy gate-chain oracle for per-match gating
Keep int64 count columns on zero-row input
Add differential parity test against the 0.1.8 gate chain
Split regex_extract into settings, scoring and column assembly
Build the Spark schema from helper.expected_count_columns
Use helper._apply_series for note normalization
Preview only matches that passed every gate
Document the review, parity results and open decisions
```
