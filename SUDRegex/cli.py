#!/usr/bin/env python

import argparse
import sys

from SUDRegex import __version__, extract, helper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--version", action="version", version=f"SUDRegex {__version__}"
    )
    parser.add_argument(
        "--extract", action="store_true", help="Perform regular expression extraction"
    )
   
    parser.add_argument("--in_file", help="Path to text file for searching")
    parser.add_argument("--out_file", help="Path to where results should be exported")
    parser.add_argument("--checklist", help="Path to the checklist file or variable")
    parser.add_argument(
        "--separator", help="Custom separator (default comma)", default=","
    )
    parser.add_argument(
        "--nrows", help="Number of text file rows to read in", default=None
    )
    parser.add_argument(
        "--chunk_size", help="Rows to analyze at a time (default 1M)", default=None
    )
    parser.add_argument("--run_tests", action="store_true", help="Run tests")
    parser.add_argument(
        "--terms",
        type=str,
        help="Comma-separated list of EXTRA terms (in addition to group)",
    )
    parser.add_argument(
        "--termslist", type=str, help="Path to Python file with term lists"
    )
    parser.add_argument(
        "--terms_active",
        type=str,
        help="Comma-separated group names in --termslist to use",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Enable parallel processing"
    )
    parser.add_argument(
        "--include_note_text",
        action="store_true",
        help="Include note text in output CSV",
    )
    parser.add_argument(
        "--results_path", help="Directory or file for aggregation/scoring"
    )
    parser.add_argument("--output_path", help="Path to save aggregate_score output CSV")
    parser.add_argument(
        "--plot_stats", action="store_true", help="Plot stats and save image"
    )
    parser.add_argument(
        "debug", nargs="?", default=False, help="Enable debug mode (default False)"
    )
    args = parser.parse_args()
    helper.PRINT = bool(args.debug)  # Set debug mode based on command line argument
    if args.extract:
        if not args.in_file or not args.out_file or not args.checklist:
            print(
                "[ERROR] --in_file, --out_file, and --checklist are required for extraction."
            )
            sys.exit(1)

        extract(
            in_file=args.in_file,
            out_file=args.out_file,
            checklist=args.checklist,
            separator=args.separator,
            terms=args.terms.split(",") if args.terms else None,
            termslist=args.termslist,
            terms_active=args.terms_active,
            parallel=args.parallel,
            include_note_text=args.include_note_text,
            nrows=args.nrows,
            chunk_size=args.chunk_size,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
