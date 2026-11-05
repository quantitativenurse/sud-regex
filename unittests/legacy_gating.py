"""
Legacy (pre-0.2.0) gate orchestration, kept verbatim as a test oracle.

This is the ``regex_extract`` loop as released in sudregex 0.1.8, driving the gate
functions that still live in ``sudregex.helper`` (``regex_search_file``,
``check_for_substance``, ``check_negation``, ``discharge_instructions`` and
``check_common_false_positives``). Each gate re-scans every match in the note, which is
the over-counting behavior that 0.2.0 fixed. The differential parity test runs this
oracle against the current scorer and classifies every divergence.

Only the debug logging was removed. Do not "fix" anything here: its job is to be wrong
in exactly the way the old release was.
"""

import pandas as pd

from sudregex.helper import (
    WIN_CFP,
    WIN_DISCHARGE,
    WIN_NEGATION,
    WIN_SUBSTANCE,
    check_common_false_positives,
    check_for_substance,
    check_negation,
    discharge_instructions,
    regex_search_file,
    write_previews_for_item,
)


def legacy_regex_extract(
    pattern_library,
    df_to_analyze,
    metadata,
    preview_count,
    expected_row_count,
    exclude_discharge_mentions: bool = True,
    *,
    preview_span: int = 120,
    preview_csv: str | None = None,
    preview_file: str | None = None,
    negation_scope: str = "left",
    return_previews: bool = False,
    use_parallel: bool = False,
    parallel_backend: str | None = None,
    n_workers: int | None = None,
):
    """
    Apply the pattern library to the prepared input DataFrame.

    This function:
        pass
    - computes base regex matches
    - applies substance and negation gating when configured
    - applies discharge-context and false-positive pruning when configured
    - optionally emits previews for pattern library items marked with `preview=True`

    The input note text is assumed to have been normalized upstream.
    """
    negation_scope = (negation_scope or "left").lower()
    if negation_scope not in {"left", "right", "both"}:
        raise ValueError("negation_scope must be 'left', 'right', or 'both'")

    previews_acc: list[dict] = []

    for i in pattern_library:
        actual_rows = df_to_analyze.shape[0]
        assert actual_rows == expected_row_count, f"Row counts do not match ({actual_rows} != {expected_row_count})"

        pat = pattern_library[i]["pat"]
        col_name = pattern_library[i]["col_name"]

        has_substance = bool(pattern_library[i].get("substance") or pattern_library[i].get("opioid"))
        has_negation = bool(pattern_library[i].get("negation"))

        df_searched = regex_search_file(
            pat,
            col_name,
            df_to_analyze,
            metadata,
            preview=True,
            use_parallel=use_parallel,
            parallel_backend=parallel_backend,
            n_workers=n_workers,
        )
        base_sum = int(pd.to_numeric(df_searched[col_name], errors="coerce").fillna(0).sum())

        active_col = col_name

        if has_substance:
            if base_sum > 0:
                df_searched = check_for_substance(
                    pat,
                    col_name,
                    f"{col_name}_SUBSTANCE_MATCHED",
                    df_searched,
                    span=WIN_SUBSTANCE,
                    use_parallel=use_parallel,
                    parallel_backend=parallel_backend,
                    n_workers=n_workers,
                )
                active_col = f"{col_name}_SUBSTANCE_MATCHED"
                sub_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())

                if has_negation:
                    if sub_sum > 0:
                        df_searched = check_negation(
                            pat,
                            active_col,
                            f"{active_col}_NEG",
                            df_searched,
                            t=[],
                            neg=True,
                            span=WIN_NEGATION,
                            side=negation_scope,
                            use_parallel=use_parallel,
                            parallel_backend=parallel_backend,
                            n_workers=n_workers,
                        )
                        active_col = f"{active_col}_NEG"
                    else:
                        df_searched[f"{active_col}_NEG"] = 0
                        active_col = f"{active_col}_NEG"
            else:
                df_searched[f"{col_name}_SUBSTANCE_MATCHED"] = 0
                if has_negation:
                    df_searched[f"{col_name}_SUBSTANCE_MATCHED_NEG"] = 0
                    active_col = f"{col_name}_SUBSTANCE_MATCHED_NEG"
                else:
                    active_col = f"{col_name}_SUBSTANCE_MATCHED"

        elif has_negation:
            if base_sum > 0:
                df_searched = check_negation(
                    pat,
                    col_name,
                    f"{col_name}_NEG",
                    df_searched,
                    t=[],
                    neg=True,
                    span=WIN_NEGATION,
                    side=negation_scope,
                    use_parallel=use_parallel,
                    parallel_backend=parallel_backend,
                    n_workers=n_workers,
                )
                active_col = f"{col_name}_NEG"
            else:
                df_searched[f"{col_name}_NEG"] = 0
                active_col = f"{col_name}_NEG"

        else:
            pass

        should_prune = has_negation or not has_substance

        if active_col in df_searched.columns:
            pre_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())

            if should_prune and pre_sum > 0:
                if exclude_discharge_mentions:
                    df_searched = discharge_instructions(
                        pat,
                        df_searched,
                        active_col,
                        span=WIN_DISCHARGE,
                        use_parallel=use_parallel,
                        parallel_backend=parallel_backend,
                        n_workers=n_workers,
                    )

                common_fp = pattern_library[i].get("common_fp") or []
                if common_fp:
                    df_searched = check_common_false_positives(
                        pat,
                        df_searched,
                        active_col,
                        common_fp,
                        span=WIN_CFP,
                        use_parallel=use_parallel,
                        parallel_backend=parallel_backend,
                        n_workers=n_workers,
                    )
            else:
                pass
        else:
            pass

        if active_col not in df_searched.columns:
            df_searched[active_col] = 0

        if pattern_library[i].get("preview"):
            sub_win = WIN_SUBSTANCE if has_substance else 0
            neg_win = WIN_NEGATION if has_negation else 0
            fp_win = WIN_CFP if pattern_library[i].get("common_fp") else 0
            dis_win = WIN_DISCHARGE if exclude_discharge_mentions else 0
            left_req = max(preview_span, sub_win, neg_win, fp_win, dis_win)
            right_req = left_req

            rows = write_previews_for_item(
                df_searched=df_searched,
                item_key=i,
                pat=pat,
                mask_col=active_col,
                n_notes=preview_count,
                left_chars=left_req,
                right_chars=right_req,
                csv_path=preview_csv,
                outfile=preview_file,
                highlight=True,
                highlight_style="brackets",
            )
            if rows:
                previews_acc.extend(rows)

        merge_cols = ["note_id", col_name]
        if has_substance:
            merge_cols.append(f"{col_name}_SUBSTANCE_MATCHED")
        if has_negation and has_substance:
            merge_cols.append(f"{col_name}_SUBSTANCE_MATCHED_NEG")
        elif has_negation and not has_substance:
            merge_cols.append(f"{col_name}_NEG")
        if active_col not in merge_cols:
            merge_cols.append(active_col)

        for mc in merge_cols:
            if mc not in df_searched.columns:
                df_searched[mc] = 0

        metadata = metadata.merge(df_searched[merge_cols], on="note_id", how="left")

    metadata = metadata.merge(df_to_analyze[["note_id", "note_text"]], on="note_id", how="left")

    if return_previews:
        previews_df = (
            pd.DataFrame(previews_acc)
            if previews_acc
            else pd.DataFrame(columns=["item_key", "note_id", "span_start", "span_end", "snippet", "snippet_marked"])
        )
        return metadata, previews_df

    return metadata
