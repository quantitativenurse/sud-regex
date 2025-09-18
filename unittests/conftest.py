from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--notes-file", default="unittests/data/input_fake.txt", help="Path to researcher-editable notes (txt/csv/tsv)."
    )
    parser.addoption(
        "--notes-sep",
        default="auto",
        help="Notes separator: 'auto' (default), ',', '\\t', '|', ';', or regex as 're:<pattern>' "
        "(e.g. re:\\s*!\\^!\\s*).",
    )
    parser.addoption(
        "--expected-csv", default="unittests/data/output_expected2_part_0.csv", help="Path to the golden expected CSV."
    )
    parser.addoption("--regen-golden", action="store_true", help="Rewrite the golden CSV from current output.")
    parser.addoption(
        "--regen-from",
        default="df",
        choices=["df", "extract"],
        help="Which function to use when regenerating the golden (default: df).",
    )
    parser.addoption("--terms-active", default=None, help="Comma list of term groups if using termslist.")
    parser.addoption(
        "--terms",
        default=None,
        help="Semicolon-separated custom terms, e.g. 'term1;term2'. Falls back to a harmless placeholder.",
    )
    parser.addoption(
        "--check-binary", action="store_true", help="Also compute presence/absence binary diffs (>=1 -> 1)."
    )
    parser.addoption(
        "--artifacts-dir", default="unittests/artifacts", help="Directory to write artifacts (mismatches) to."
    )
    parser.addoption(
        "--summary-file", default=None, help="If set, write a one-line summary of the test run to this file."
    )


def _write_summary(artifacts_dir: str, summary_file: str | None):
    """
    Scan artifacts_dir for '*mismatches.csv' and write a human-friendly summary.
    """
    from pathlib import Path

    import pandas as pd

    adir = Path(artifacts_dir)
    adir.mkdir(parents=True, exist_ok=True)
    sfile = Path(summary_file) if summary_file else adir / "mismatch_summary.txt"

    mismatch_files = sorted(adir.glob("*mismatches.csv"))
    lines = []
    nonempty = 0

    for f in mismatch_files:
        try:
            df = pd.read_csv(f)
            nrows = len(df)
        except Exception:
            # fallback: count lines minus header
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                nrows = max(0, sum(1 for _ in fh) - 1)

        if nrows > 0:
            nonempty += 1
            lines.append(f"{f.name}: {nrows} mismatches")
            # show a tiny preview if columns are standard
            try:
                df_preview = df.head(5)
                keep = [c for c in df_preview.columns if c.lower() in ("note_id", "column", "actual", "expected")]
                if keep:
                    df_preview = df_preview[keep]
                lines.append(df_preview.to_string(index=False))
                lines.append("")
            except Exception:
                pass

    if nonempty == 0:
        lines = ["No mismatches found."]

    sfile.write_text("\n".join(lines), encoding="utf-8")
    # Also echo a short pointer to the console
    print(f"[summary] wrote {sfile} ({'no mismatches' if nonempty == 0 else 'has mismatches'})")


def pytest_sessionfinish(session, exitstatus):
    # Always write a summary at the end
    cfg = session.config
    _write_summary(
        artifacts_dir=cfg.getoption("--artifacts-dir"),
        summary_file=cfg.getoption("--summary-file"),
    )


@pytest.fixture(scope="session")
def test_cfg(pytestconfig):
    return {
        "notes_file": Path(pytestconfig.getoption("--notes-file")),
        "notes_sep": pytestconfig.getoption("--notes-sep"),
        "expected_csv": Path(pytestconfig.getoption("--expected-csv")),
        "regen_golden": pytestconfig.getoption("--regen-golden"),
        "regen_from": pytestconfig.getoption("--regen-from"),
        "terms_active": pytestconfig.getoption("--terms-active"),
        "terms": pytestconfig.getoption("--terms"),
        "check_binary": pytestconfig.getoption("--check-binary"),
    }
