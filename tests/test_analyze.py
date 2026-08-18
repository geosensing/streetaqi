from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from streetaqi.analyze import (
    compute_category_distribution,
    compute_per_stop_stats,
    compute_summary_stats,
    compute_threshold_exceedance,
    get_pm25_category,
    make_map,
    make_scatter,
    process,
)
from streetaqi.data import write_readings

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "Good"),
        (9.0, "Good"),
        (9.05, "Good"),
        (35.4, "Moderate"),
        (35.45, "Moderate"),
        (55.4, "Unhealthy for Sensitive Groups"),
        (55.5, "Unhealthy"),
        (125.5, "Very Unhealthy"),
        (225.5, "Hazardous"),
    ],
)
def test_pm25_categories_truncate_to_one_decimal(value: float, expected: str) -> None:
    assert get_pm25_category(value) == expected


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_pm25_category_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        get_pm25_category(value)


def test_summary_reproduces_claims_with_explicit_co2_denominator(
    readings: pd.DataFrame,
) -> None:
    summary = compute_summary_stats(readings)
    assert summary["n_readings"] == 98
    assert summary["n_co2_qc_pass"] == 96
    assert summary["n_days"] == 5
    assert summary["pm25_mean"] == pytest.approx(117.6367346939)
    assert summary["pm25_median"] == 94.0
    assert summary["co2_mean"] == pytest.approx(551.875)
    assert summary["co2_median"] == 415.0
    assert summary["co2_min"] == 401.0


def test_summary_requires_data_and_qc_passing_co2(readings: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="At least one reading"):
        compute_summary_stats(readings.iloc[:0])
    with pytest.raises(ValueError, match="pass quality control"):
        compute_summary_stats(readings.assign(co2_qc_pass=False))


def test_threshold_estimand_is_observation_level(readings: pd.DataFrame) -> None:
    result = compute_threshold_exceedance(readings)
    assert result["n_readings"].tolist() == [98, 97, 97, 21, 6]
    assert result["share_of_readings"].tolist() == pytest.approx(
        [1.0, 97 / 98, 97 / 98, 21 / 98, 6 / 98]
    )
    assert result.loc[1, "category_or_worse"] == (
        "Unhealthy for Sensitive Groups or worse"
    )
    assert result.loc[2, "category_or_worse"] == "Unhealthy or worse"


def test_category_distribution_is_complete(readings: pd.DataFrame) -> None:
    result = compute_category_distribution(readings)
    assert result["n_readings"].sum() == 98
    assert result["share_of_readings"].sum() == pytest.approx(1.0)
    assert result.set_index("category").loc["Hazardous", "n_readings"] == 6


def test_per_stop_stats_conserve_rows_and_qc(readings: pd.DataFrame) -> None:
    result = compute_per_stop_stats(readings)
    assert result["n_readings"].sum() == 98
    assert result["n_co2_qc_pass"].sum() == 96
    assert not result.duplicated(["day", "itinerary_id"]).any()
    assert "share_unhealthy_or_worse" in result


def test_analysis_process_writes_typed_outputs(
    tmp_path: Path, readings: pd.DataFrame
) -> None:
    input_path = write_readings(readings, tmp_path / "readings.parquet")
    artifacts = process(input_path, tmp_path / "analysis")
    assert {
        "summary",
        "threshold_exceedance",
        "category_distribution",
        "per_stop",
        "histograms",
        "scatter",
        "map",
    } == set(artifacts)
    assert all(path.is_file() for path in artifacts.values())
    summary = pd.read_parquet(artifacts["summary"])
    assert summary.loc[0, "n_co2_qc_pass"] == 96


def test_map_skips_invalid_coordinates(tmp_path: Path, readings: pd.DataFrame) -> None:
    invalid = readings.assign(latitude=999.0, longitude=999.0)
    assert make_map(invalid, tmp_path) is None


def test_scatter_handles_one_qc_pair(tmp_path: Path, readings: pd.DataFrame) -> None:
    path = make_scatter(readings.iloc[:1].assign(co2_qc_pass=True), tmp_path)
    assert path.is_file()
