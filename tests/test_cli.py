from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

from streetaqi.cli import main
from streetaqi.data import write_ocr_results

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import pytest


def test_version_command() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_sample_data_command(tmp_path: Path) -> None:
    output = tmp_path / "sample.parquet"
    result = CliRunner().invoke(main, ["sample-data", "--output", str(output)])
    assert result.exit_code == 0
    assert output.is_file()
    assert "Sample data:" in result.output


def test_analyze_command_reports_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, readings: pd.DataFrame
) -> None:
    input_path = tmp_path / "readings.parquet"
    from streetaqi.data import write_readings

    write_readings(readings, input_path)
    expected = tmp_path / "summary.parquet"
    expected.write_bytes(b"parquet")
    monkeypatch.setattr(
        "streetaqi.analyze.process", lambda *_args: {"summary": expected}
    )
    result = CliRunner().invoke(
        main,
        ["analyze", "--readings", str(input_path), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 0
    assert f"summary: {expected}" in result.output


def test_viewer_command(tmp_path: Path, ocr_results: pd.DataFrame) -> None:
    readings_path = write_ocr_results(ocr_results, tmp_path / "results.parquet")
    output = tmp_path / "viewer.html"
    result = CliRunner().invoke(
        main,
        ["viewer", "--readings", str(readings_path), "--output", str(output)],
    )
    assert result.exit_code == 0
    assert output.is_file()
    assert "Generated viewer:" in result.output


def test_annotate_command_rejects_empty_glob(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["annotate", "--images", str(tmp_path / "*.jpg")],
    )
    assert result.exit_code != 0
    assert "No images found" in result.output
