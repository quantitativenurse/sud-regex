# unittests/test_spark.py
"""
Tests for sudregex.spark (Databricks / PySpark execution support).

Two tiers:

- Schema/unit tests (`Test*Schema`, `TestResolveLibrary`) exercise the pure
  pandas/Python logic with no SparkSession required — these run anywhere,
  including CI without a JVM.
- Integration tests (`TestRunDatabricks`) spin up a local SparkSession and
  run the real `mapInPandas` path end-to-end, asserting the distributed
  output matches the local `extract_df` output row-for-row. These are
  skipped automatically if pyspark isn't installed.
"""

import re

import pandas as pd
import pytest

from sudregex import spark as sudregex_spark
from sudregex.pattern_library import pattern_library as full_pattern_library

pyspark = pytest.importorskip("pyspark", reason="pyspark not installed; skipping Spark integration tests")

from pyspark.sql import SparkSession  # noqa: E402

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def spark_session():
    """A local, single-JVM SparkSession for tests. Torn down after the module."""
    session = (
        SparkSession.builder.master("local[2]")
        .appName("sudregex-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def mini_library():
    """A small, hand-built pattern library covering all four column-naming branches."""
    return {
        # substance=True, negation=True -> col, col_SUBSTANCE_MATCHED, col_SUBSTANCE_MATCHED_NEG
        "opioid_item": {
            "pat": re.compile(r"oxycodone", re.IGNORECASE),
            "col_name": "opioid_mention",
            "opioid": True,
            "negation": True,
            "preview": False,
        },
        # negation only (no substance) -> col, col_NEG
        "dui_item": {
            "pat": re.compile(r"\bDUI\b"),
            "col_name": "dui_history",
            "opioid": False,
            "negation": True,
            "preview": False,
        },
        # no substance, no negation -> col only
        "plain_item": {
            "pat": re.compile(r"hoard"),
            "col_name": "hoarding_mention",
            "opioid": False,
            "negation": False,
            "preview": False,
        },
    }


@pytest.fixture
def notes_df():
    return pd.DataFrame(
        {
            "note_id": ["n1", "n2", "n3"],
            "note_text": [
                "Patient prescribed oxycodone for pain, denies misuse.",
                "History notable for DUI in 2019.",
                "No concerning findings in this note.",
            ],
        }
    )


# ----------------------------------------------------------------------
# expected_count_columns / output_columns (pure logic, mirrors helper.py)
# ----------------------------------------------------------------------


class TestExpectedCountColumns:
    def test_substance_and_negation_item(self, mini_library):
        cols = sudregex_spark.expected_count_columns({"opioid_item": mini_library["opioid_item"]})
        assert cols == [
            "opioid_mention",
            "opioid_mention_SUBSTANCE_MATCHED",
            "opioid_mention_SUBSTANCE_MATCHED_NEG",
        ]

    def test_negation_only_item(self, mini_library):
        cols = sudregex_spark.expected_count_columns({"dui_item": mini_library["dui_item"]})
        assert cols == ["dui_history", "dui_history_NEG"]

    def test_plain_item(self, mini_library):
        cols = sudregex_spark.expected_count_columns({"plain_item": mini_library["plain_item"]})
        assert cols == ["hoarding_mention"]

    def test_full_mini_library_order_preserved(self, mini_library):
        cols = sudregex_spark.expected_count_columns(mini_library)
        assert cols == [
            "opioid_mention",
            "opioid_mention_SUBSTANCE_MATCHED",
            "opioid_mention_SUBSTANCE_MATCHED_NEG",
            "dui_history",
            "dui_history_NEG",
            "hoarding_mention",
        ]

    def test_matches_real_pattern_library_naming(self):
        """
        Cross-check against the real ABC pattern_library.py: every item with
        opioid=True should produce a _SUBSTANCE_MATCHED column, and every
        negation=True item without opioid should produce a _NEG column.
        """
        cols = sudregex_spark.expected_count_columns(full_pattern_library)
        for key, cfg in full_pattern_library.items():
            base = cfg["col_name"]
            assert base in cols
            if cfg.get("opioid"):
                assert f"{base}_SUBSTANCE_MATCHED" in cols
                if cfg.get("negation"):
                    assert f"{base}_SUBSTANCE_MATCHED_NEG" in cols
            elif cfg.get("negation"):
                assert f"{base}_NEG" in cols


class TestOutputColumns:
    def test_identifier_columns_come_first(self, mini_library):
        cols = sudregex_spark.output_columns(
            mini_library,
            id_column="note_id",
            person_column="grid",
            extra_id_columns=["station_id"],
        )
        assert cols[:3] == ["grid", "note_id", "station_id"]
        assert cols[3:] == sudregex_spark.expected_count_columns(mini_library)

    def test_no_person_or_extra_columns(self, mini_library):
        cols = sudregex_spark.output_columns(mini_library, id_column="note_id")
        assert cols[0] == "note_id"
        assert "grid" not in cols


# ----------------------------------------------------------------------
# build_spark_schema (requires pyspark, but no SparkSession)
# ----------------------------------------------------------------------


class TestBuildSparkSchema:
    def test_schema_field_names_and_order(self, mini_library):
        from pyspark.sql.types import LongType, StringType

        schema = sudregex_spark.build_spark_schema(mini_library, id_column="note_id", person_column="grid")
        names = [f.name for f in schema.fields]
        assert names == sudregex_spark.output_columns(mini_library, id_column="note_id", person_column="grid")

        # identifier columns are StringType
        assert isinstance(schema["grid"].dataType, StringType)
        assert isinstance(schema["note_id"].dataType, StringType)
        # count columns are LongType
        assert isinstance(schema["opioid_mention"].dataType, LongType)
        assert isinstance(schema["dui_history_NEG"].dataType, LongType)

    def test_schema_from_file_path(self, tmp_path, mini_library):
        """build_spark_schema should also accept a pattern_library.py file path."""
        lib_file = tmp_path / "mini_lib.py"
        lib_file.write_text(
            "import re\n"
            "pattern_library = {\n"
            "    'plain_item': {\n"
            "        'pat': re.compile(r'hoard'),\n"
            "        'col_name': 'hoarding_mention',\n"
            "        'opioid': False,\n"
            "        'negation': False,\n"
            "        'preview': False,\n"
            "    }\n"
            "}\n"
        )
        schema = sudregex_spark.build_spark_schema(str(lib_file), id_column="note_id")
        names = [f.name for f in schema.fields]
        assert names == ["note_id", "hoarding_mention"]


# ----------------------------------------------------------------------
# run_databricks: real end-to-end integration tests against a local Spark
# ----------------------------------------------------------------------


class TestRunDatabricks:
    def test_raises_without_spark_session(self, mini_library, notes_df):
        with pytest.raises(ValueError, match="requires an active SparkSession"):
            sudregex_spark.run_databricks(notes_df, mini_library, spark=None)

    def test_raises_on_wrong_input_type(self, spark_session, mini_library):
        with pytest.raises(TypeError, match="pandas DataFrame or a Spark DataFrame"):
            sudregex_spark.run_databricks(["not", "a", "dataframe"], mini_library, spark=spark_session)

    def test_raises_on_missing_required_columns(self, spark_session, mini_library):
        bad_df = pd.DataFrame({"note_id": ["n1"], "wrong_col": ["text"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            sudregex_spark.run_databricks(bad_df, mini_library, spark=spark_session)

    def test_distributed_output_matches_local_extract_df(self, spark_session, mini_library, notes_df):
        """
        The core correctness test: run the same notes through the local
        pandas path (extract_df) and the distributed Spark path
        (run_databricks), and confirm they produce the same match counts
        per note_id.
        """
        import sudregex as sudregex_pkg

        local_result = sudregex_pkg.extract_df(
            notes_df.copy(),
            mini_library,
            note_column="note_text",
            id_column="note_id",
            include_note_text=False,
            terms=["oxycodone"],
        ).set_index("note_id")

        spark_df = sudregex_spark.run_databricks(
            notes_df.copy(),
            mini_library,
            spark=spark_session,
            note_column="note_text",
            id_column="note_id",
            terms=["oxycodone"],
        )
        spark_result = spark_df.toPandas().set_index("note_id")

        count_cols = sudregex_spark.expected_count_columns(mini_library)
        for col in count_cols:
            assert col in local_result.columns, f"local extract_df missing {col}"
            assert col in spark_result.columns, f"spark output missing {col}"

        local_result = local_result.loc[sorted(local_result.index)]
        spark_result = spark_result.loc[sorted(spark_result.index)]

        for col in count_cols:
            local_vals = local_result[col].astype("int64")
            spark_vals = spark_result[col].astype("int64")
            assert local_vals.equals(
                spark_vals
            ), f"mismatch in column {col}:\nlocal:\n{local_vals}\nspark:\n{spark_vals}"

    def test_output_schema_matches_declared_schema(self, spark_session, mini_library, notes_df):
        spark_df = sudregex_spark.run_databricks(
            notes_df.copy(), mini_library, spark=spark_session, id_column="note_id", terms=["oxycodone"]
        )
        expected_schema = sudregex_spark.build_spark_schema(mini_library, id_column="note_id")
        assert spark_df.schema == expected_schema

    def test_aggregate_notes_combines_split_rows(self, spark_session, mini_library):
        """
        Two rows sharing a note_id should be concatenated into one note
        before extraction when aggregate_notes=True (the default).
        """
        split_df = pd.DataFrame(
            {
                "note_id": ["n1", "n1"],
                "note_text": ["Patient prescribed oxycodone", "for chronic pain, denies misuse."],
            }
        )
        spark_df = sudregex_spark.run_databricks(
            split_df,
            mini_library,
            spark=spark_session,
            id_column="note_id",
            aggregate_notes=True,
            terms=["oxycodone"],
        )
        result = spark_df.toPandas()
        assert len(result) == 1
        assert result.iloc[0]["opioid_mention"] >= 1

    def test_duplicate_note_id_raises_when_not_aggregating(self, spark_session, mini_library):
        split_df = pd.DataFrame(
            {
                "note_id": ["n1", "n1"],
                "note_text": ["first part", "second part"],
            }
        )
        with pytest.raises(ValueError, match="multiple rows for some"):
            sudregex_spark.run_databricks(
                split_df, mini_library, spark=spark_session, id_column="note_id", aggregate_notes=False
            )

    def test_person_and_extra_id_columns_preserved(self, spark_session, mini_library):
        """
        Regression test for a bug where extract_df() computed extras_df for
        extra_id_columns but never merged it back into the final result
        (_finalize_result only reattached person_column via crosswalk).
        Fixed in _build_crosswalk() / _prepare_grouped_input() by carrying
        extra_id_columns through the same crosswalk path as person_column.
        """
        df = pd.DataFrame(
            {
                "note_id": ["n1", "n2"],
                "grid": ["R100", "R100"],
                "station_id": ["S1", "S2"],
                "note_text": ["Patient prescribed oxycodone.", "History of DUI."],
            }
        )
        spark_df = sudregex_spark.run_databricks(
            df,
            mini_library,
            spark=spark_session,
            id_column="note_id",
            person_column="grid",
            extra_id_columns=["station_id"],
            terms=["oxycodone"],
        )
        result = spark_df.toPandas().set_index("note_id")
        assert {"grid", "station_id"}.issubset(result.columns)
        assert set(result["grid"]) == {"R100"}
        # Verify per-row correctness, not just column presence: station_id
        # must stay matched to the right note_id through the distributed path.
        assert result.loc["n1", "station_id"] == "S1"
        assert result.loc["n2", "station_id"] == "S2"
