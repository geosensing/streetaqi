from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from streetaqi.data import (
    OCR_RESULTS_SCHEMA,
    READINGS_SCHEMA,
    load_bundled_readings,
    load_id_mapping,
    load_ocr_results,
    load_readings,
    validate_readings_table,
    write_id_mapping,
    write_ocr_results,
    write_readings,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_bundled_readings_have_one_typed_source(readings: pd.DataFrame) -> None:
    assert readings.shape == (98, 14)
    assert readings["id"].nunique() == 98
    assert readings["co2_qc_pass"].value_counts().to_dict() == {True: 96, False: 2}
    resource = resources.files("streetaqi").joinpath("data/delhi_readings.parquet")
    with resources.as_file(resource) as path:
        assert pq.read_table(path).schema == READINGS_SCHEMA


def test_readings_parquet_round_trip(tmp_path: Path, readings: pd.DataFrame) -> None:
    output = write_readings(readings, tmp_path / "nested/readings.parquet")
    pd.testing.assert_frame_equal(load_readings(output), readings)


def test_readings_csv_is_an_import_boundary(
    tmp_path: Path, readings: pd.DataFrame
) -> None:
    path = tmp_path / "readings.csv"
    readings.to_csv(path, index=False)
    actual = load_readings(path)
    pd.testing.assert_frame_equal(actual, readings)


def test_readings_reject_unexpected_dataframe_and_csv_columns(
    tmp_path: Path, readings: pd.DataFrame
) -> None:
    changed = readings.assign(unexpected="do not discard")
    with pytest.raises(ValueError, match="columns must exactly match"):
        write_readings(changed, tmp_path / "readings.parquet")
    csv_path = tmp_path / "readings.csv"
    changed.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="columns must exactly match"):
        load_readings(csv_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.assign(id="duplicate"), "IDs must be unique"),
        (lambda frame: frame.assign(pm25=-1.0), "PM2.5 readings"),
        (lambda frame: frame.assign(pm25=float("inf")), "PM2.5 readings"),
        (lambda frame: frame.assign(co2=-1.0), "CO2 readings"),
        (lambda frame: frame.assign(co2=float("inf")), "CO2 readings"),
        (lambda frame: frame.assign(latitude=91.0), "Latitude"),
        (lambda frame: frame.assign(longitude=181.0), "Longitude"),
        (lambda frame: frame.assign(co2_qc_pass=True), "co2_qc_pass"),
    ],
)
def test_readings_invariants_fail(
    readings: pd.DataFrame,
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    changed = mutate(readings.copy())
    table = pa.Table.from_pandas(
        changed,
        schema=READINGS_SCHEMA.remove_metadata(),
        preserve_index=False,
    )
    with pytest.raises(ValueError, match=message):
        validate_readings_table(table)


def test_readings_reject_empty_and_schema_drift(readings: pd.DataFrame) -> None:
    empty = pa.Table.from_pandas(
        readings.iloc[:0],
        schema=READINGS_SCHEMA.remove_metadata(),
        preserve_index=False,
    )
    with pytest.raises(ValueError, match="at least one"):
        validate_readings_table(empty)
    with pytest.raises(ValueError, match="canonical schema"):
        validate_readings_table(pa.table({"id": ["x"]}))


def test_readings_reject_non_null_nan(readings: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(
        readings,
        schema=READINGS_SCHEMA.remove_metadata(),
        preserve_index=False,
    )
    index = table.schema.get_field_index("co2")
    nan_values = pa.array(
        [float("nan")] * table.num_rows,
        type=pa.float64(),
        from_pandas=False,
    )
    table = table.set_column(index, table.schema.field(index), nan_values)
    with pytest.raises(ValueError, match="CO2 readings"):
        validate_readings_table(table)


def test_readings_reject_nulls_in_required_fields(readings: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(
        readings,
        schema=READINGS_SCHEMA.remove_metadata(),
        preserve_index=False,
    )
    index = table.schema.get_field_index("title")
    titles = table["title"].to_pylist()
    titles[0] = None
    table = table.set_column(
        index,
        table.schema.field(index),
        pa.array(titles, type=pa.string()),
    )
    with pytest.raises(ValueError, match=r"title.*null"):
        validate_readings_table(table)


def test_readings_reject_unknown_extension_and_output(
    tmp_path: Path, readings: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match=r"parquet or \.csv"):
        load_readings(tmp_path / "readings.json")
    with pytest.raises(ValueError, match=r"written as \.parquet"):
        write_readings(readings, tmp_path / "readings.csv")


def test_bundled_loader_uses_public_contract() -> None:
    assert len(load_bundled_readings()) == 98


def test_ocr_results_round_trip(tmp_path: Path, ocr_results: pd.DataFrame) -> None:
    path = write_ocr_results(ocr_results, tmp_path / "ocr/results.parquet")
    actual = load_ocr_results(path)
    assert actual.shape == (2, 15)
    assert actual["status"].tolist() == ["ok", "partial_read"]
    assert pq.read_table(path).schema == OCR_RESULTS_SCHEMA


def test_ocr_results_reject_invalid_data(
    tmp_path: Path, ocr_results: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match=r"loaded from \.parquet"):
        load_ocr_results(tmp_path / "results.json")
    with pytest.raises(ValueError, match=r"written as \.parquet"):
        write_ocr_results(ocr_results, tmp_path / "results.json")
    with pytest.raises(ValueError, match="confidence"):
        write_ocr_results(
            ocr_results.assign(confidence=2.0), tmp_path / "confidence.parquet"
        )
    with pytest.raises(ValueError, match="IDs must be unique"):
        write_ocr_results(ocr_results.assign(id="same"), tmp_path / "ids.parquet")
    with pytest.raises(ValueError, match="columns must exactly match"):
        write_ocr_results(
            ocr_results.assign(unexpected="do not discard"),
            tmp_path / "columns.parquet",
        )
    drift = tmp_path / "drift.parquet"
    pq.write_table(pa.table({"id": ["x"]}), drift)
    with pytest.raises(ValueError, match="canonical schema"):
        load_ocr_results(drift)


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "ok", "pm25": None},
        {"status": "sensor_not_found", "pm25": 85.0, "co2": 412.0},
        {"status": "unknown"},
        {"pm25": -1.0},
        {"co2": float("inf")},
        {"pm25": True},
        {"co2": "412"},
        {"confidence": True},
        {"confidence": "0.9"},
        {"reference_pm25": float("inf")},
        {"reference_co2": -1.0},
        {"reference_pm25": True},
        {"reference_co2": "410"},
    ],
)
def test_ocr_storage_rejects_invalid_status_value_combinations(
    tmp_path: Path,
    ocr_results: pd.DataFrame,
    changes: dict[str, object],
) -> None:
    changed = ocr_results.copy()
    for name, value in changes.items():
        changed[name] = changed[name].astype(object)
        changed.loc[0, name] = value
    with pytest.raises(ValueError, match="OCR"):
        write_ocr_results(changed, tmp_path / "invalid.parquet")


def test_ocr_loader_revalidates_storage_invariants(
    tmp_path: Path, ocr_results: pd.DataFrame
) -> None:
    changed = ocr_results.copy()
    changed.loc[0, "status"] = "display_unreadable"
    table = pa.Table.from_pandas(
        changed,
        schema=OCR_RESULTS_SCHEMA,
        preserve_index=False,
    )
    path = tmp_path / "invalid.parquet"
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="OCR"):
        load_ocr_results(path)


@pytest.mark.parametrize("column", ["reference_pm25", "reference_co2"])
def test_ocr_loader_rejects_nonfinite_reference_concentrations(
    tmp_path: Path,
    ocr_results: pd.DataFrame,
    column: str,
) -> None:
    table = pa.Table.from_pandas(
        ocr_results,
        schema=OCR_RESULTS_SCHEMA,
        preserve_index=False,
    )
    index = table.schema.get_field_index(column)
    values = table[column].to_pylist()
    values[0] = float("inf")
    table = table.set_column(
        index,
        table.schema.field(index),
        pa.array(values, type=pa.float64(), from_pandas=False),
    )
    path = tmp_path / "invalid-reference.parquet"
    pq.write_table(table, path)
    with pytest.raises(ValueError, match=column):
        load_ocr_results(path)


def test_id_mapping_round_trip_and_validation(tmp_path: Path) -> None:
    path = write_id_mapping({"a": "one.jpg", "b": "two.png"}, tmp_path / "map.parquet")
    assert load_id_mapping(path) == {"a": "one.jpg", "b": "two.png"}
    duplicate = tmp_path / "duplicate.parquet"
    table = pa.Table.from_pydict(
        {"custom_id": ["a", "a"], "image_path": ["one", "two"]},
        schema=pq.read_table(path).schema,
    )
    pq.write_table(table, duplicate)
    with pytest.raises(ValueError, match="unique"):
        load_id_mapping(duplicate)
