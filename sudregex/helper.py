"""
Helper file so that functions are stored separately from main execution file.
"""

import os
import re
import sys
from typing import List
from typing import Pattern as RePattern
from typing import Union

import pandas as pd

PRINT = False

# injected term storage (_terms by main)
TERMS_LIST: list[str] = []
TERMS_COMPILED: list[re.Pattern] = []


def _finditer(pat: RePattern | str, text: str):
    if isinstance(pat, re.Pattern):
        return pat.finditer(text)
    return re.finditer(pat, text, flags=re.IGNORECASE | re.MULTILINE)


def _search(term: RePattern | str, text: str):
    if isinstance(term, re.Pattern):
        return term.search(text) is not None
    return re.search(term, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def set_terms(terms: list[str]) -> None:
    """
    Call once at startup to tell this helper which vocabulary to use.
    """
    global TERMS_LIST, TERMS_COMPILED
    TERMS_LIST = terms
    TERMS_COMPILED = [re.compile(re.escape(t), re.IGNORECASE | re.MULTILINE) for t in TERMS_LIST]
    if PRINT:
        print("Using terms:", TERMS_LIST)


# Memory usage
try:
    total_memory, used_memory, free_memory = map(int, os.popen("free -t -m").readlines()[-1].split()[1:])
except Exception:
    total_memory, used_memory, free_memory = (0, 0, 0)

# note filters, for removing symbols etc.
# our helper (with both marker‑removal and whitespace‑collapse)


def remove_line_break(
    text: Union[str, bytes],
    break_markers: Union[str, List[str]] = r"\$\+\$",
    replacement: str = " ",
) -> str:
    s = text.decode() if isinstance(text, (bytes, bytearray)) else str(text)

    # Build regex pattern for markers
    if isinstance(break_markers, str):
        # treat string as a regex pattern directly
        pattern = break_markers
    else:
        parts = []
        for m in break_markers:
            # if the user‐supplied string starts with a backslash, assume it's already
            # a regex escape sequence; otherwise escape it literally
            if m and m[0] == "\\":
                parts.append(m)
            else:
                parts.append(re.escape(m))
        pattern = "|".join(parts)

    # 1) remove the markers
    s = re.sub(pattern, replacement, s)
    # 2) collapse any run of whitespace to a single space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def previews_batch(checklist, df_summarized, n_notes=2, span=300):
    """
    The following function previews the notes and writes them to a text file.
    """
    original_stdout = sys.stdout
    with open("note_previews.txt", "w") as f:
        sys.stdout = f
        columns = list(df_summarized)
        if "note_id" in columns:
            columns.remove("note_id")  # optional safety
        if "note_text" in columns:
            columns.remove("note_text")  # optional safety

        for i in checklist:
            col_name = checklist[i]["col_name"]
            if col_name not in columns:
                continue

            lab = checklist[i]["lab"]
            if PRINT:
                print("\n", lab, "\n")

            pat = checklist[i]["pat"]
            if PRINT:
                print("Pattern:", pat, "\n")

            if "substance" in checklist[i] and checklist[i]["substance"]:
                col_name_substance = col_name + "_SUBSTANCE_MATCHED"
                if checklist[i]["negation"]:
                    col_name_negated = col_name_substance + "_NEG"
            elif checklist[i]["negation"]:
                col_name_negated = col_name + "_NEG"

            if "substance" in checklist[i] and checklist[i]["substance"]:
                col_name_list = [col_name_substance]
                if checklist[i]["negation"]:
                    col_name_list = [col_name_negated]
            elif checklist[i]["negation"]:
                col_name_list = [col_name_negated]
            else:
                col_name_list = [col_name]

            if PRINT:
                print("Columns related to this checklist item:", col_name_list, "\n")

            for col_name in col_name_list:
                if PRINT:
                    print("Column Name:", col_name, "\n")

                total = len(df_summarized[df_summarized[col_name] > 0])
                k = min(n_notes, total)
                matches = df_summarized[df_summarized[col_name] > 0]
                if k == 0:
                    continue
                matches = matches.sample(k, random_state=123)

                for j in range(matches.shape[0]):  # minor: avoid reusing i
                    if PRINT:
                        print("~~~ " + str(matches["note_id"].iloc[j]) + " ~~~")

                    text = matches["note_text"].iloc[j]  # here is the change
                    for m in _finditer(pat, text):  # here is the change
                        if PRINT:
                            print(m, "\n")

                        start, stop = m.span()
                        s = max(0, start - span)  # here is the change
                        e = min(len(text), stop + span)  # here is the change
                        text_print = text[s:e]  # here is the change

                        if PRINT:
                            print(text_print)
                            print("\n")

                del matches

        sys.stdout = original_stdout


def regex_extract(checklist, df_to_analyze, metadata, preview_count, expected_row_count):
    """
    Applies the checklist of regex searches to the data frame, with optional substance and negation checks.
    """

    if PRINT:
        print(
            f"[DEBUG] Starting regex_extract: df_to_analyze.shape={df_to_analyze.shape}, "
            f"metadata.shape={metadata.shape}, expected_row_count={expected_row_count}"
        )

    for i in checklist:
        if PRINT:
            print(f"\n[DEBUG] Checklist item index: {i}")

        actual_rows = df_to_analyze.shape[0]
        if PRINT:
            print(f"[DEBUG]  → df_to_analyze has {actual_rows} rows")

        assert actual_rows == expected_row_count, f"Row counts do not match ({actual_rows} != {expected_row_count})"

        pat = checklist[i]["pat"]
        col_name = checklist[i]["col_name"]
        if PRINT:
            print(f"[DEBUG]  → pattern='{pat.pattern if hasattr(pat, 'pattern') else pat}', " f"col_name='{col_name}'")

        # Treat either 'substance' OR 'opioid' as the gating flag
        has_substance = bool(checklist[i].get("substance") or checklist[i].get("opioid"))
        has_negation = bool(checklist[i].get("negation"))

        if PRINT:
            print(f"[DEBUG]  → substance={has_substance}, negation={has_negation}")

        # Initial search
        if PRINT:
            print(f"[DEBUG]  → Calling regex_search_file for '{col_name}'")
        df_searched = regex_search_file(pat, col_name, df_to_analyze, metadata, preview=True)

        base_sum = int(pd.to_numeric(df_searched[col_name], errors="coerce").fillna(0).sum())
        if PRINT:
            print(f"[DEBUG]  → After regex_search_file: df_searched['{col_name}'].sum()={base_sum}")

        # Track which column is the "active" mask throughout downstream steps
        active_col = col_name

        # substance (+ optional negation) branch
        if has_substance:
            if PRINT:
                print(f"[DEBUG]  → Entering substance branch for '{col_name}'")

            if base_sum > 0:
                # Add *_SUBSTANCE_MATCHED
                df_searched = check_for_substance(pat, col_name, f"{col_name}_SUBSTANCE_MATCHED", df_searched)
                active_col = f"{col_name}_SUBSTANCE_MATCHED"
                sub_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
                if PRINT:
                    print(f"[DEBUG]    • After check_for_substance: {sub_sum} matches")

                # Optional negation on the substance-filtered hits
                if has_negation:
                    if PRINT:
                        print("[DEBUG]    • Entering negation checks")
                    if sub_sum > 0:
                        df_searched = check_negation(
                            pat,
                            active_col,  # input mask (*_SUBSTANCE_MATCHED)
                            f"{active_col}_NEG",  # output mask (*_SUBSTANCE_MATCHED_NEG)
                            df_searched,
                            t=[],
                            neg=True,
                            span=65,
                        )
                        active_col = f"{active_col}_NEG"  # now *_SUBSTANCE_MATCHED_NEG
                        neg_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
                        if PRINT:
                            print(f"[DEBUG]    • After check_negation: {neg_sum} kept")
                    else:
                        # Ensure column exists even if no substance matches
                        df_searched[f"{active_col}_NEG"] = 0
                        active_col = f"{active_col}_NEG"
                        if PRINT:
                            print(f"[DEBUG]    • No substance matches; set {active_col}=0")
            else:
                # No base matches — create downstream columns to avoid merge KeyErrors
                df_searched[f"{col_name}_SUBSTANCE_MATCHED"] = 0
                if has_negation:
                    df_searched[f"{col_name}_SUBSTANCE_MATCHED_NEG"] = 0
                    active_col = f"{col_name}_SUBSTANCE_MATCHED_NEG"
                else:
                    active_col = f"{col_name}_SUBSTANCE_MATCHED"
                if PRINT:
                    print(f"[DEBUG]    • No initial matches; zeroed {active_col}")

        # negation-only branch
        elif has_negation:
            if PRINT:
                print(f"[DEBUG]  → Entering negation-only branch for '{col_name}'")
            if base_sum > 0:
                df_searched = check_negation(pat, col_name, f"{col_name}_NEG", df_searched, t=[], neg=True, span=65)
                active_col = f"{col_name}_NEG"
                neg_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
                if PRINT:
                    print(f"[DEBUG]    • After check_negation: {neg_sum} negated")
            else:
                df_searched[f"{col_name}_NEG"] = 0
                active_col = f"{col_name}_NEG"
                if PRINT:
                    print(f"[DEBUG]    • No initial matches; set {active_col}=0")

        # base branch (neither substance nor negation)
        else:
            if PRINT:
                print(f"[DEBUG]  → No substance/negation flags for '{col_name}' (base branch)")

        # Pruning policy to match published behavior:
        # - prune if there is a negation branch OR there is no substance/opioid gate
        # - skip pruning for opioid-only (no negation) items
        should_prune = has_negation or not has_substance

        if active_col in df_searched.columns:
            pre_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
            if PRINT:
                print(f"[DEBUG]    • Before pruning on {active_col}: {pre_sum} kept")

            if should_prune and pre_sum > 0:
                # Discharge instruction pruning (wide window)
                df_searched = discharge_instructions(pat, df_searched, active_col, span=250)
                post_discharge_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
                if PRINT:
                    print(f"[DEBUG]    • After discharge_instructions on {active_col}: {post_discharge_sum} kept")

                # Common false positive pruning if provided in checklist
                common_fp = checklist[i].get("common_fp") or []
                if common_fp:
                    df_searched = check_common_false_positives(pat, df_searched, active_col, common_fp, span=65)
                    post_fp_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
                    if PRINT:
                        print(f"[DEBUG]    • After common FP pruning on {active_col}: {post_fp_sum} kept")
            else:
                if PRINT:
                    print(
                        f"[DEBUG]    • Skipping pruning for {active_col} "
                        f"(matches={pre_sum}, has_substance={has_substance}, has_negation={has_negation})"
                    )
        else:
            if PRINT:
                print(f"[DEBUG]    • Skipping pruning (missing column {active_col})")

        # Optional preview
        if checklist[i].get("preview"):
            if PRINT:
                print(f"[DEBUG]  → Previewing {preview_count} matches for '{col_name}'")
            preview_string_matches(pat, col_name, df_searched, n_notes=preview_count, span=100)

        # Build merge column set, ensuring active_col is present
        merge_cols = ["note_id", col_name]
        if has_substance:
            merge_cols.append(f"{col_name}_SUBSTANCE_MATCHED")
        if has_negation and has_substance:
            merge_cols.append(f"{col_name}_SUBSTANCE_MATCHED_NEG")
        elif has_negation and not has_substance:
            merge_cols.append(f"{col_name}_NEG")
        if active_col not in merge_cols:
            merge_cols.append(active_col)

        # Create missing columns with zeros before merge to avoid KeyErrors
        for mc in merge_cols:
            if mc not in df_searched.columns:
                df_searched[mc] = 0

        if PRINT:
            cur_sum = int(pd.to_numeric(df_searched[active_col], errors="coerce").fillna(0).sum())
            print(f"[DEBUG]  → SUMMARY '{col_name}': base={base_sum} | active({active_col})={cur_sum}")
            print(f"[DEBUG]  → Merging columns {merge_cols} into metadata (active_col={active_col})")

        metadata = metadata.merge(df_searched[merge_cols], on="note_id", how="left")

    # Final merge for note_text
    metadata = metadata.merge(df_to_analyze[["note_id", "note_text"]], on="note_id", how="left")
    if PRINT:
        print(f"[DEBUG] Finished regex_extract: metadata.shape={metadata.shape}")
    return metadata


def regex_search_file(pat, new_col_name, df_to_search, metadata, preview=True):
    """
    This function searches the note text in each of the rows of the file and
    returns the number of matches for the regex.
    It concatenates the matches to a summarized df for each note_id

    INPUTS: the pattern from the checklist, the name of the column for the
    summarized df from the checklist,
    the df to search for matches w/ notes, and preview argument.

    OUTPUTS: df that has been searched for matches with note_text and matches
    """
    from re import Pattern

    # print("Regex Search File: RAM memory % used:", round((used_memory/total_memory) * 100, 2))
    # create empty dataframe for storing counts
    # new_col_name: is col_name taken in new_column_and_search and specified in each ABC checklist
    # item
    counts = pd.DataFrame(columns=["note_id", "note_text", new_col_name])

    # search for pattern
    # findall: Return all non-overlapping matches of pattern in string,
    # as a list of strings or tuples. The string is scanned left-to-right,
    # and matches are returned in the order found. Empty matches are included in the result.

    # returns the number of matches in the note_text with ignoring case and the multiline flag
    def pat_search(text):
        if isinstance(pat, Pattern):
            # use the compiled regex’s own flags
            return len(pat.findall(text))
        else:
            # compile on the fly with IGNORECASE|MULTILINE
            return len(re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE))

    # apply the search to every note text
    # use parallel apply to process multiple notes at the same time
    # for df in dfs:

    # need to fix - not sure why parallel_apply isn't working with AUDIT
    #     df_to_search['note_text'] = df_to_search['note_text'].parallel_apply(remove_line_break)
    #     df_to_search[new_col_name] = df_to_search['note_text'].parallel_apply(pat_search)
    df_to_search["note_text"] = df_to_search["note_text"].apply(remove_line_break)
    df_to_search[new_col_name] = df_to_search["note_text"].apply(pat_search)

    # add resulting counts to the temporary dataframe
    # outer join of counts to input df that also sorts the df
    counts = pd.concat([counts, df_to_search], sort=True)

    # columns to keep for the final df
    keep_cols = ["note_id", new_col_name]
    # optional preview of note text
    if preview:
        keep_cols.append("note_text")

    # merge w/ left join to creat df on note id with the counts
    df_searched = metadata.merge(counts[keep_cols], how="left", on="note_id")

    # create dataframe of matches for providing descriptive statistics
    # print('Original pattern search yielded '
    # + str(for_counts[new_col_name].sum()) + ' total matches within '
    # + str(len(np.unique(for_counts['note_id']))) + ' notes.')

    # print("Regex Search File: RAM memory % used:", round((used_memory/total_memory) * 100, 2))

    return df_searched


# possible name change for this functino to mask_no_tobacco_mentions
def remove_tobacco_mentions(text):
    """Mask mentions of no tobacco use in the text."""
    # Extended regular expression to find various mentions of no tobacco use
    tobacco_pattern = r"(Tob(?:acco)?[ -]*(?:use)?[: -]*\b(None|Never|No\s+use|abstains|denies use,? never|ever)\b|Smoking[: -]*None|Smoker[: -]*(?:never|no))"
    return re.sub(tobacco_pattern, "Tobacco: [Redacted]", str(text), flags=re.IGNORECASE)


def check_for_substance(pat, col_name, col_name_substance, df_searched, span=100):
    """
    Searches if substance-related tag around match. Creates new column with results.

    Inputs: pattern to search, new column name, df searched for initial matches, and span argument that determines
    scope of search.

    Outputs: searched df with additional substance matches
    """
    import re

    import pandas as pd

    from SUDRegex.helper import TERMS_LIST

    # internal helper to iterate matches without mixing flags on compiled patterns
    def _iter_matches(pattern, text):
        if hasattr(pattern, "finditer"):
            return pattern.finditer(text)
        return re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE)

    yes_or_no = []
    matches = df_searched[df_searched[col_name] > 0]

    for i in range(matches.shape[0]):
        match_found = False
        text = matches["note_text"].iloc[i]
        for m in _iter_matches(pat, text):
            start, stop = m.span()
            # expand window
            start = max(0, start - span)
            stop = max(stop, stop + span)
            match_str = text[start:stop]

            # check each term
            for term in TERMS_LIST:
                if re.search(term, match_str, flags=re.IGNORECASE | re.MULTILINE):
                    match_found = True
                    break
            if match_found:
                break

        yes_or_no.append(1 if match_found else 0)

    # attach results
    matches[col_name_substance] = yes_or_no
    add_df = pd.DataFrame({col_name_substance: yes_or_no}, index=matches["note_id"]).reset_index()
    df_substance = df_searched.merge(add_df, on="note_id", how="left")
    return df_substance


def check_negation(
    pat,
    col_name,
    col_name_negated,
    df_searched,
    t=None,
    neg=True,
    span=65,
):
    """
    Searches for negation vocabulary around match. Creates new column with results.

    Inputs:
      pat                – regex or string to look for
      col_name           – initial-match count column
      col_name_negated   – name of the new negation column to add
      df_searched        – dataframe with at least [note_id, note_text, col_name]
      t                  – extra negation terms (list of strings/regex)
      neg                – whether to include the built-in negation list
      span               – how many chars before the match to look for negations
    Outputs: df with a new column `col_name_negated` (1 = not negated, 0 = negated)
    """
    # default extra terms
    t = t or []
    # build negation vocabulary
    if neg:
        vocab = [
            "no ",
            "not ",
            "denie",
            "denial",
            "doubt",
            "never",
            "negative",
            "without",
            "neg",
            "didn't",
        ]
        vocab.extend(t)
    else:
        vocab = t

    # rows that actually had at least one initial match
    matches = df_searched[df_searched[col_name] > 0].copy()

    yes_or_no = []
    for idx, row in matches.iterrows():
        text = row["note_text"]
        is_negated = False

        # for each match occurrence ...
        for m in _finditer(pat, text):
            start, stop = m.span()

            window = text[max(0, start - span) : stop]

            # if any negation term appears in that window, flag it
            for term in vocab:
                if _search(term, window):
                    is_negated = True
                    break
            if is_negated:
                break

        # append 0 if negated, else 1
        yes_or_no.append(0 if is_negated else 1)

    # attach new column back onto the original df
    matches[col_name_negated] = yes_or_no
    neg_df = matches[["note_id", col_name_negated]]
    result = df_searched.merge(neg_df, on="note_id", how="left")

    return result


def check_common_false_positives(pat, df_searched, col_name_fp, common_fp, span=20):
    """
    Remove matches that occur only in common false‑positive contexts.
    """
    # work on a copy
    df = df_searched.copy()
    # get only rows with an initial hit
    hits = df[df[col_name_fp] > 0]
    to_replace = []
    for idx, row in hits.iterrows():
        text = row["note_text"]
        is_fp = False
        for m in _finditer(pat, text):
            start, stop = m.span()
            window = text[max(0, start - span) : min(len(text), stop + span)]
            for term in common_fp:
                if _search(term, window):
                    is_fp = True
                    break
            if is_fp:
                break
        to_replace.append((idx, 0 if is_fp else 1))

    # overwrite the original column
    for idx, val in to_replace:
        df.at[idx, col_name_fp] = val

    return df


def discharge_instructions(pat, df_searched, col_name_discharge, span=350):
    """
    Remove matches that occur only in discharge instruction contexts.
    """
    discharge_terms = ["discharge instructions", "no results for"]
    df = df_searched.copy()
    hits = df[df[col_name_discharge] > 0]
    to_replace = []
    for idx, row in hits.iterrows():
        text = row["note_text"]
        is_dis = False
        for m in _finditer(pat, text):
            start, stop = m.span()
            window = text[max(0, start - span) : min(len(text), stop + span)]
            for term in discharge_terms:
                if _search(term, window):
                    is_dis = True
                    break
            if is_dis:
                break
        to_replace.append((idx, 0 if is_dis else 1))

    # overwrite the original column
    for idx, val in to_replace:
        df.at[idx, col_name_discharge] = val

    return df


def preview_string_matches(pat, col_name, df_searched, col_check=False, n_notes=10, span=100):
    """
    INPUTS: data frame with match data and note text
    OUTPUT: a preview of the string where the match occurs in the note printed out
    """
    if col_check and col_name not in df_searched.columns:  # optional safety
        raise KeyError(f"Column '{col_name}' not found in df_searched")

    hits = df_searched[df_searched[col_name] > 0]
    hit_count = len(hits)
    if hit_count == 0 or n_notes <= 0:  # ← minimal guard
        return

    # respect the caller's n_notes
    k = min(n_notes, hit_count)  # ← don't overwrite n_notes
    matches = hits.sample(k, random_state=123)

    for i in range(matches.shape[0]):
        if PRINT:
            print(str(matches["note_id"].iloc[i]))
        text = matches["note_text"].iloc[i]
        for m in _finditer(pat, text):
            start, stop = m.span()
            s = max(0, start - span)
            e = min(len(text), stop + span)  # to avoid overflow

            text_print = text[s:e]
            text_print = (
                text_print[0:(span)]
                + "\x1b[0;39;43m"
                + text_print[(span) : (span + 8)]
                + "\x1b[0m"
                + text_print[(span + 8) : -1]
            )

            for term in TERMS_LIST:
                x = re.search(term, text_print, flags=re.IGNORECASE | re.MULTILINE)
                if x:
                    s, e = x.span()
                    text_print = text_print[0:s] + "\x1b[0;39;43m" + text_print[s:e] + "\x1b[0m" + text_print[e:]

            neg = ["no ", "not ", "denies", "denial", "doubt", "never", "negative for"]
            for term in neg:
                x = re.search(term, text_print, flags=re.IGNORECASE | re.MULTILINE)
                if x:
                    s, e = x.span()
                    text_print = text_print[0:s] + "\x1b[0;39;43m" + text_print[s:e] + "\x1b[0m" + text_print[e:]

            if PRINT:
                print(text_print)
                print("\n")

    # print("Previews: RAM memory % used:", round((used_memory/total_memory) * 100, 2))
