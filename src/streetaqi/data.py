"""Typed storage for street-level air-quality readings."""

from __future__ import annotations

import math
from importlib import resources
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pathlib import Path

MIN_PLAUSIBLE_CO2_PPM = 350.0

MODEL_OCR_STATUSES = frozenset(
    {
        "ok",
        "sensor_not_found",
        "display_unreadable",
        "image_unclear",
        "partial_read",
    }
)
OCR_STATUSES = MODEL_OCR_STATUSES | {"api_error", "parse_error"}

READINGS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("day", pa.int64(), nullable=False),
        pa.field("itinerary_id", pa.int64(), nullable=False),
        pa.field("part", pa.int64(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("captured_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("pm25", pa.float64(), nullable=False),
        pa.field("co2", pa.float64(), nullable=False),
        pa.field("co2_qc_pass", pa.bool_(), nullable=False),
        pa.field("latitude", pa.float64(), nullable=False),
        pa.field("longitude", pa.float64(), nullable=False),
        pa.field("note", pa.string(), nullable=False),
        pa.field("image_local_path", pa.string(), nullable=False),
        pa.field("image_remote_url", pa.string(), nullable=False),
    ],
    metadata={
        b"description": (
            b"Street-level air-quality sensor readings collected in Delhi in May 2026"
        ),
        b"co2_qc_rule": b"co2 >= 350 ppm; raw values retained",
        b"source": b"soundscape rider export 2026-05-24",
    },
)

OCR_RESULTS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("image_path", pa.string(), nullable=False),
        pa.field("day", pa.int64(), nullable=True),
        pa.field("itinerary_id", pa.int64(), nullable=True),
        pa.field("latitude", pa.float64(), nullable=True),
        pa.field("longitude", pa.float64(), nullable=True),
        pa.field("pm25", pa.float64(), nullable=True),
        pa.field("co2", pa.float64(), nullable=True),
        pa.field("status", pa.string(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("reference_pm25", pa.float64(), nullable=True),
        pa.field("reference_co2", pa.float64(), nullable=True),
        pa.field("provider", pa.string(), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        pa.field("batch_id", pa.string(), nullable=False),
    ],
    metadata={b"description": b"streetaqi OCR results"},
)

ID_MAPPING_SCHEMA = pa.schema(
    [
        pa.field("custom_id", pa.string(), nullable=False),
        pa.field("image_path", pa.string(), nullable=False),
    ],
    metadata={b"description": b"Claude batch custom-ID mapping"},
)


def _require_exact_columns(
    frame: pd.DataFrame,
    schema: pa.Schema,
    label: str,
) -> None:
    expected = schema.names
    actual = list(frame.columns)
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError(
            f"{label} columns must exactly match the canonical schema; "
            f"expected {expected}, got {actual}"
        )


def _reject_nulls(table: pa.Table, schema: pa.Schema, label: str) -> None:
    for field in schema:
        if not field.nullable and table[field.name].null_count:
            raise ValueError(
                f"{label} field {field.name!r} must not contain null values"
            )


def validate_ocr_measurements(
    pm25: object,
    co2: object,
    status: object,
) -> None:
    """Validate the relationship between OCR status and measured values."""
    if not isinstance(status, str) or status not in OCR_STATUSES:
        raise ValueError(f"Unknown OCR status: {status!r}")
    for label, value in (("PM2.5", pm25), ("CO2", co2)):
        _validate_optional_concentration(value, f"OCR {label}")
    if status == "ok" and (pm25 is None or co2 is None):
        raise ValueError("OCR status 'ok' requires both readings")
    if status == "partial_read" and ((pm25 is None) == (co2 is None)):
        raise ValueError("OCR status 'partial_read' requires exactly one reading")
    if status not in {"ok", "partial_read"} and (pm25 is not None or co2 is not None):
        raise ValueError("OCR failure statuses require null readings")


def _validate_optional_concentration(value: object, label: str) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be finite and non-negative or null")


def _normalize_frame_null(value: object) -> object:
    if (
        value is None
        or value is pd.NA
        or (isinstance(value, Real) and math.isnan(value))
    ):
        return None
    return value


def _validate_ocr_frame_values(frame: pd.DataFrame) -> None:
    for pm25, co2, status in frame[["pm25", "co2", "status"]].itertuples(
        index=False,
        name=None,
    ):
        validate_ocr_measurements(
            _normalize_frame_null(pm25),
            _normalize_frame_null(co2),
            status,
        )
    for column in ("reference_pm25", "reference_co2"):
        for value in frame[column]:
            _validate_optional_concentration(
                _normalize_frame_null(value),
                f"OCR {column}",
            )
    for confidence in frame["confidence"]:
        if isinstance(confidence, bool) or not isinstance(confidence, Real):
            raise ValueError("OCR confidence must be finite and between 0 and 1")
        numeric_confidence = float(confidence)
        if not math.isfinite(numeric_confidence) or not 0 <= numeric_confidence <= 1:
            raise ValueError("OCR confidence must be finite and between 0 and 1")


def validate_readings_table(table: pa.Table) -> None:
    """Validate a readings table against the canonical schema and invariants.

    Args:
        table: Arrow table to validate.

    Raises:
        ValueError: If the schema or values violate the data contract.
    """
    if table.schema.remove_metadata() != READINGS_SCHEMA.remove_metadata():
        raise ValueError(
            "Readings must use the canonical schema: "
            f"{READINGS_SCHEMA.remove_metadata()}"
        )
    if table.num_rows == 0:
        raise ValueError("Readings must contain at least one row")

    _reject_nulls(table, READINGS_SCHEMA, "Readings")

    frame = table.to_pandas()
    if frame["id"].duplicated().any():
        raise ValueError("Readings IDs must be unique")
    if not frame["pm25"].map(math.isfinite).all() or (frame["pm25"] < 0).any():
        raise ValueError("PM2.5 readings must be finite and non-negative")
    if not frame["co2"].map(math.isfinite).all() or (frame["co2"] < 0).any():
        raise ValueError("CO2 readings must be finite and non-negative")
    if not frame["latitude"].between(-90, 90).all():
        raise ValueError("Latitude must be between -90 and 90")
    if not frame["longitude"].between(-180, 180).all():
        raise ValueError("Longitude must be between -180 and 180")

    expected_qc = frame["co2"] >= MIN_PLAUSIBLE_CO2_PPM
    if not frame["co2_qc_pass"].equals(expected_qc):
        raise ValueError(f"co2_qc_pass must equal co2 >= {MIN_PLAUSIBLE_CO2_PPM:g} ppm")


def load_readings(path: Path) -> pd.DataFrame:
    """Load and validate readings from Parquet or an import-boundary CSV.

    Args:
        path: Input file with the canonical columns.

    Returns:
        A validated pandas DataFrame in canonical column order.

    Raises:
        ValueError: If the file type, schema, or values are invalid.
    """
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        table = pq.read_table(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
        _require_exact_columns(frame, READINGS_SCHEMA, "CSV")
        if "captured_at" in frame:
            frame["captured_at"] = pd.to_datetime(frame["captured_at"], utc=True)
        try:
            table = pa.Table.from_pandas(
                frame,
                schema=READINGS_SCHEMA.remove_metadata(),
                preserve_index=False,
                safe=True,
            )
        except (pa.ArrowInvalid, pa.ArrowTypeError, KeyError) as error:
            raise ValueError(
                "CSV does not satisfy the canonical readings schema"
            ) from error
    else:
        raise ValueError("Readings must be a .parquet or .csv file")

    validate_readings_table(table)
    return table.to_pandas()


def load_bundled_readings() -> pd.DataFrame:
    """Load the bundled Delhi readings through the public data contract."""
    resource = resources.files("streetaqi").joinpath("data/delhi_readings.parquet")
    with resources.as_file(resource) as path:
        return load_readings(path)


def write_readings(frame: pd.DataFrame, path: Path) -> Path:
    """Validate and write canonical readings as compressed Parquet.

    Args:
        frame: Readings in canonical column order.
        path: Destination, which must end in ``.parquet``.

    Returns:
        The destination path.

    Raises:
        ValueError: If the destination or readings are invalid.
    """
    if path.suffix.lower() != ".parquet":
        raise ValueError("Canonical readings must be written as .parquet")
    _require_exact_columns(frame, READINGS_SCHEMA, "DataFrame")
    try:
        table = pa.Table.from_pandas(
            frame,
            schema=READINGS_SCHEMA,
            preserve_index=False,
            safe=True,
        )
    except (pa.ArrowInvalid, pa.ArrowTypeError, KeyError) as error:
        raise ValueError(
            "DataFrame does not satisfy the canonical readings schema"
        ) from error
    validate_readings_table(table)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", version="2.6")
    return path


def load_ocr_results(path: Path) -> pd.DataFrame:
    """Load typed OCR results from Parquet.

    Args:
        path: OCR-result Parquet file.

    Returns:
        A validated result frame.

    Raises:
        ValueError: If the schema, IDs, or confidence values are invalid.
    """
    if path.suffix.lower() != ".parquet":
        raise ValueError("OCR results must be loaded from .parquet")
    table = pq.read_table(path)
    validate_ocr_results_table(table)
    return table.to_pandas()


def validate_ocr_results_table(table: pa.Table) -> None:
    """Validate an OCR result table against its schema and invariants."""
    if table.schema.remove_metadata() != OCR_RESULTS_SCHEMA.remove_metadata():
        raise ValueError("OCR results do not use the canonical schema")
    _reject_nulls(table, OCR_RESULTS_SCHEMA, "OCR results")
    frame = table.to_pandas()
    if frame["id"].duplicated().any():
        raise ValueError("OCR result IDs must be unique")
    if (
        not frame["confidence"].map(math.isfinite).all()
        or not frame["confidence"].between(0, 1).all()
    ):
        raise ValueError("OCR confidence must be finite and between 0 and 1")
    for pm25, co2, status in zip(
        table["pm25"].to_pylist(),
        table["co2"].to_pylist(),
        table["status"].to_pylist(),
        strict=True,
    ):
        validate_ocr_measurements(pm25, co2, status)
    for column in ("reference_pm25", "reference_co2"):
        for value in table[column].to_pylist():
            _validate_optional_concentration(value, f"OCR {column}")


def write_ocr_results(frame: pd.DataFrame, path: Path) -> Path:
    """Write typed OCR results as compressed Parquet.

    Args:
        frame: OCR results in canonical column order.
        path: Destination Parquet file.

    Returns:
        The destination path.

    Raises:
        ValueError: If the destination or result data are invalid.
    """
    if path.suffix.lower() != ".parquet":
        raise ValueError("OCR results must be written as .parquet")
    _require_exact_columns(frame, OCR_RESULTS_SCHEMA, "OCR result")
    _validate_ocr_frame_values(frame)
    try:
        table = pa.Table.from_pandas(
            frame,
            schema=OCR_RESULTS_SCHEMA,
            preserve_index=False,
            safe=True,
        )
    except (pa.ArrowInvalid, pa.ArrowTypeError, KeyError) as error:
        raise ValueError("OCR results do not satisfy the canonical schema") from error
    validate_ocr_results_table(table)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", version="2.6")
    return path


def write_id_mapping(mapping: dict[str, str], path: Path) -> Path:
    """Write a Claude batch custom-ID mapping as Parquet."""
    table = pa.Table.from_pydict(
        {
            "custom_id": list(mapping),
            "image_path": list(mapping.values()),
        },
        schema=ID_MAPPING_SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", version="2.6")
    return path


def load_id_mapping(path: Path) -> dict[str, str]:
    """Load a Claude batch custom-ID mapping from Parquet."""
    table = pq.read_table(path)
    if table.schema.remove_metadata() != ID_MAPPING_SCHEMA.remove_metadata():
        raise ValueError("Batch mapping does not use the canonical schema")
    frame = table.to_pandas()
    if frame["custom_id"].duplicated().any():
        raise ValueError("Batch custom IDs must be unique")
    return dict(zip(frame["custom_id"], frame["image_path"], strict=True))
