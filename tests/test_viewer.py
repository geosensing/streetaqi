from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from streetaqi.data import write_ocr_results
from streetaqi.viewer import generate_html, image_to_data_uri

if TYPE_CHECKING:
    import pandas as pd


def test_image_embedding_requires_trusted_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "images"
    root.mkdir()
    image = root / "sensor.jpg"
    image.write_bytes(b"image")
    assert (
        image_to_data_uri(Path("sensor.jpg"), root) == "data:image/jpeg;base64,aW1hZ2U="
    )
    monkeypatch.chdir(tmp_path)
    assert (
        image_to_data_uri(Path("images/sensor.jpg"), Path("images"))
        == "data:image/jpeg;base64,aW1hZ2U="
    )
    assert image_to_data_uri(image, None) is None
    assert image_to_data_uri(Path("missing.jpg"), root) is None

    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"secret")
    with pytest.raises(ValueError, match="outside"):
        image_to_data_uri(outside, root)
    unsupported = root / "sensor.gif"
    unsupported.write_bytes(b"gif")
    with pytest.raises(ValueError, match="Unsupported"):
        image_to_data_uri(unsupported, root)


def test_generate_html_escapes_untrusted_values_and_has_no_csv_export(
    tmp_path: Path, ocr_results: pd.DataFrame
) -> None:
    malicious = ocr_results.copy()
    malicious.loc[0, "image_path"] = "</script><img src=x onerror=alert(1)>.jpg"
    readings_path = write_ocr_results(malicious, tmp_path / "results.parquet")
    output = generate_html(readings_path, tmp_path / "nested/viewer.html")
    html = output.read_text()
    assert output.is_file()
    assert "Export reviewed JSON" in html
    assert "Export CSV" not in html
    assert "<\\/script><img" in html
    assert "</script><img src=x" not in html
    assert "Image not embedded" in html


def test_generate_html_embeds_images_from_root(
    tmp_path: Path, ocr_results: pd.DataFrame
) -> None:
    root = tmp_path / "images"
    image_path = root / "day-09/001_itinerary-1-photo.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    one = ocr_results.iloc[:1].copy()
    one.loc[:, "image_path"] = "day-09/001_itinerary-1-photo.jpg"
    readings_path = write_ocr_results(one, tmp_path / "results.parquet")
    output = generate_html(readings_path, image_root=root)
    assert "data:image/jpeg;base64,aW1hZ2U=" in output.read_text()
