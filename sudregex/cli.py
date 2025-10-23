#!/usr/bin/env python

import argparse
import os
import sys
import traceback

from . import __version__, extract, helper  # import extract from package root
from .validation import import_python_object, parse_text, validate_rows


def main():
    parser = argparse.ArgumentParser(prog="sudregex")
    parser.add_argument("-v", "--version", action="version", version=f"sudregex {__version__}")

    # Modes
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extract", action="store_true", help="Perform regular expression extraction")
    mode.add_argument("--validate", action="store_true", help="Validate a checklist against labeled examples")

    parser.add_argument("--checklist", help="Path to the checklist .py file (must define `checklist`)")

    # -------- Extract args --------
    parser.add_argument("--in_file", help="Path to text file for searching")
    parser.add_argument("--out_file", help="Path to where results should be exported")
    parser.add_argument("--separator", default=",", help="Custom separator (default ',')")
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to read (default: all)")
    parser.add_argument("--chunk_size", type=int, default=None, help="Rows to analyze per chunk (default: no chunking)")
    parser.add_argument("--run_tests", action="store_true", help="Run tests (currently unused)")
    parser.add_argument("--terms", type=str, help="Comma-separated list of EXTRA terms (in addition to group)")
    parser.add_argument("--termslist", type=str, help="Path to Python file with term lists")
    parser.add_argument("--terms_active", type=str, help="Comma-separated group names in --termslist to use")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument(
        "--n-workers", dest="n_workers", type=int, default=None, help="Number of workers for pandarallel (optional)"
    )
    parser.add_argument("--include_note_text", action="store_true", help="Include note text in output CSV")
    parser.add_argument(
        "--preview-count", type=int, default=0, help="Number of preview snippets per matching item (default: 0 = off)"
    )
    parser.add_argument(
        "--preview-span", type=int, default=100, help="Chars to show on each side of a match in previews (default: 100)"
    )
    parser.add_argument("--preview-file", type=str, default=None, help="If set, append previews to this file (text).")
    parser.add_argument(
        "--preview-csv",
        type=str,
        default=None,
        help="If set, write previews to this CSV (schema: item_key,note_id,span_start,span_end,snippet).",
    )

    # NEW: negation scope
    parser.add_argument(
        "--negation-scope",
        choices=["left", "right", "both"],
        default="left",
        help="Where to look for negation cues relative to a hit (default: left).",
    )

    # --- Discharge-mentions policy (default: EXCLUDE) ---
    dis = parser.add_mutually_exclusive_group()
    dis.add_argument(
        "--include-discharge-mentions",
        dest="exclude_discharge_mentions",
        action="store_false",
        help="Include matches even if they occur in discharge-instruction contexts.",
    )
    dis.add_argument(
        "--exclude-discharge-mentions",
        dest="exclude_discharge_mentions",
        action="store_true",
        help="Exclude matches that occur in discharge-instruction contexts (default).",
    )
    parser.add_argument(
        "--no-exclude-discharge-mentions",
        dest="exclude_discharge_mentions",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(exclude_discharge_mentions=True)

    # -------- Validate args --------
    parser.add_argument(
        "--examples", help="Path to examples file ('item | expected | note_text' or legacy '!^!' format)"
    )
    parser.add_argument(
        "--val_out",
        default="checklist_validation_results.csv",
        help="Detailed rows CSV (default: checklist_validation_results.csv)",
    )
    parser.add_argument(
        "--val_by_item",
        default="checklist_validation_by_item.csv",
        help="Per-item summary CSV (default: checklist_validation_by_item.csv)",
    )
    parser.add_argument("--print_mismatches", action="store_true", help="Print mismatched rows to stdout")
    parser.add_argument("--mismatch_limit", type=int, default=20, help="Max mismatches to print (default: 20)")

    # Debug flag
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    helper.PRINT = args.debug

    if not args.extract and not args.validate:
        parser.print_help()
        return

    try:
        if args.extract:
            # Required args
            if not args.in_file or not args.out_file or not args.checklist:
                print("[ERROR] --in_file, --out_file, and --checklist are required for --extract.")
                sys.exit(1)

            # Early file checks (friendlier errors)
            if not os.path.exists(args.in_file):
                print(f"[ERROR] Input file not found: {args.in_file}", file=sys.stderr)
                sys.exit(1)
            if not os.path.exists(args.checklist):
                print(f"[ERROR] Checklist file not found: {args.checklist}", file=sys.stderr)
                sys.exit(1)

            # Tidy terms (trim + drop empties)
            terms_list = [t.strip() for t in args.terms.split(",") if t.strip()] if args.terms else None

            if args.debug:
                print(
                    f"[DEBUG] negation_scope={args.negation_scope}, "
                    f"exclude_discharge={args.exclude_discharge_mentions}, "
                    f"preview_count={args.preview_count}, preview_span={args.preview_span}"
                )

            extract(
                in_file=args.in_file,
                out_file=args.out_file,
                checklist=args.checklist,
                separator=args.separator or "",
                terms=terms_list,
                termslist=args.termslist,
                terms_active=args.terms_active,
                parallel=args.parallel,
                n_workers=args.n_workers,
                include_note_text=args.include_note_text,
                nrows=args.nrows,
                chunk_size=args.chunk_size,
                debug=args.debug,
                exclude_discharge_mentions=args.exclude_discharge_mentions,
                preview_count=args.preview_count,
                preview_span=args.preview_span,
                preview_file=args.preview_file,
                preview_csv=args.preview_csv,
                # NEW passthrough
                negation_scope=args.negation_scope,
            )
            return

        if args.validate:
            if not args.checklist or not args.examples:
                print("[ERROR] --checklist and --examples are required for --validate.")
                sys.exit(1)

            if not os.path.exists(args.checklist):
                print(f"[ERROR] Checklist file not found: {args.checklist}", file=sys.stderr)
                sys.exit(1)
            if not os.path.exists(args.examples):
                print(f"[ERROR] Examples file not found: {args.examples}", file=sys.stderr)
                sys.exit(1)

            checklist = import_python_object(args.checklist, "checklist")
            df_examples = parse_text(args.examples)
            detailed, by_item = validate_rows(checklist, df_examples)

            detailed.to_csv(args.val_out, index=False)
            by_item.to_csv(args.val_by_item, index=False)

            total = len(detailed)
            correct = int((detailed["actual_match"] == detailed["expected"].astype(int)).sum())
            acc = (correct / total) if total else float("nan")
            print(f"Rows: {total} | Correct: {correct} | Accuracy: {acc:.3f}")
            print(f"Wrote: {args.val_out}")
            print(f"Wrote: {args.val_by_item}")

            if args.print_mismatches:
                mm = detailed[detailed["actual_match"] != detailed["expected"].astype(int)]
                if mm.empty:
                    print("\nNo mismatches")
                else:
                    print(f"\nMismatches (first {args.mismatch_limit}):")
                    print(mm.head(args.mismatch_limit).to_string(index=False))
            return

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
