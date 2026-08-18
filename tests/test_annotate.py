from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import streetaqi.annotate as annotate
from streetaqi.data import load_ocr_results


def test_image_helpers_and_custom_ids(tmp_path: Path) -> None:
    jpg = tmp_path / "image.jpg"
    jpg.write_bytes(b"image")
    png = tmp_path / "image.png"
    png.write_bytes(b"png")
    assert annotate.image_mime_type(jpg) == "image/jpeg"
    assert annotate.image_mime_type(png) == "image/png"
    assert annotate.encode_image_base64(jpg) == "aW1hZ2U="
    assert len(annotate.path_to_custom_id(str(jpg))) == 64
    assert annotate.path_to_custom_id(str(jpg)) == annotate.path_to_custom_id(str(jpg))
    with pytest.raises(ValueError, match="Unsupported"):
        annotate.image_mime_type(tmp_path / "image.gif")


@pytest.mark.parametrize(
    ("response", "status"),
    [
        ('{"pm25": 85, "co2": 412, "status": "ok", "confidence": 0.95}', "ok"),
        (
            'text ```json\n{"pm25": 92, "co2": null, "status": "partial_read", "confidence": 0.75}\n```',
            "partial_read",
        ),
        ("not json", "parse_error"),
        ('{"pm25": -1, "co2": 412, "status": "ok", "confidence": 1}', "parse_error"),
        ('{"pm25": 1, "co2": null, "status": "ok", "confidence": 1}', "parse_error"),
        (
            '{"pm25": 1, "co2": 2, "status": "partial_read", "confidence": 1}',
            "parse_error",
        ),
        ('{"pm25": 1, "co2": 2, "status": "invented", "confidence": 1}', "parse_error"),
        ('{"pm25": 1, "co2": 2, "status": "ok", "confidence": 2}', "parse_error"),
        ('{"pm25": true, "co2": 2, "status": "ok", "confidence": 1}', "parse_error"),
        (
            '{"pm25": 1, "co2": 2, "status": "sensor_not_found", "confidence": 1}',
            "parse_error",
        ),
        (
            '{"pm25": null, "co2": null, "status": "sensor_not_found", "confidence": 1}',
            "sensor_not_found",
        ),
    ],
)
def test_parse_ocr_response(response: str, status: str) -> None:
    assert annotate.parse_ocr_response(response)["status"] == status


def test_model_and_path_metadata() -> None:
    path = Path("exports/images/day-09/001_itinerary-12-photo.jpg")
    assert annotate.is_gemini_model("gemini-3.5-flash")
    assert not annotate.is_gemini_model("claude-haiku-4-5-20251001")
    assert annotate.extract_metadata_from_path(path) == {"day": 9, "itinerary_id": 12}
    assert annotate.extract_metadata_from_path(Path("day-bad/plain.jpg")) == {
        "day": None,
        "itinerary_id": None,
    }


def test_reference_match_is_exact_and_detects_ambiguity(
    readings: pd.DataFrame,
) -> None:
    source = readings.iloc[0]
    candidate = Path("prefix") / str(source["image_local_path"])
    match = annotate._reference_for_image(candidate, readings)
    assert match["reference_pm25"] == source["pm25"]
    shallow = Path(*Path(str(source["image_local_path"])).parts[-2:])
    assert (
        annotate._reference_for_image(shallow, readings)["reference_pm25"]
        == source["pm25"]
    )
    assert (
        annotate._reference_for_image(Path("missing.jpg"), readings)["reference_pm25"]
        is None
    )
    lookalike = readings.iloc[:1].assign(image_local_path="notphoto.jpg")
    assert (
        annotate._reference_for_image(Path("photo.jpg"), lookalike)["reference_pm25"]
        is None
    )
    duplicated = pd.concat([readings.iloc[:1], readings.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Ambiguous"):
        annotate._reference_for_image(candidate, duplicated)


def test_create_claude_batch_request_uses_real_mime(tmp_path: Path) -> None:
    image = tmp_path / "sensor.png"
    image.write_bytes(b"png")
    request = annotate.create_claude_batch_request(image, "custom")
    source = request["params"]["messages"][0]["content"][0]["source"]
    assert source["media_type"] == "image/png"
    assert request["params"]["model"] == annotate.DEFAULT_CLAUDE_MODEL
    assert request["params"]["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": annotate.OCR_RESPONSE_SCHEMA,
        }
    }
    with pytest.raises(FileNotFoundError):
        annotate.create_claude_batch_request(tmp_path / "missing.jpg", "custom")


def test_claude_client_operations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = tmp_path / "sensor.jpg"
    image.write_bytes(b"jpg")
    result_object = SimpleNamespace(custom_id="one")
    statuses = iter(["in_progress", "ended"])
    batches = SimpleNamespace(
        create=lambda **_kwargs: SimpleNamespace(id="batch-1"),
        retrieve=lambda _batch_id: SimpleNamespace(processing_status=next(statuses)),
        results=lambda _batch_id: [result_object],
    )
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    monkeypatch.setattr("anthropic.Anthropic", lambda **_kwargs: client)
    monkeypatch.setattr(annotate.time, "sleep", lambda _seconds: None)

    batch_id, mapping = annotate.submit_claude_batch([image])
    assert batch_id == "batch-1"
    assert list(mapping.values()) == [str(image)]
    annotate.poll_claude_batch(batch_id, poll_interval=1)
    assert annotate.fetch_claude_batch_results(batch_id) == [result_object]
    with pytest.raises(ValueError, match="At least one"):
        annotate.submit_claude_batch([])
    with pytest.raises(ValueError, match="at least one second"):
        annotate.poll_claude_batch(batch_id, poll_interval=0)


def test_process_claude_results() -> None:
    text_block = SimpleNamespace(
        type="text",
        text='{"pm25": 85, "co2": 412, "status": "ok", "confidence": 0.9}',
    )
    succeeded = SimpleNamespace(
        custom_id="a",
        result=SimpleNamespace(
            type="succeeded", message=SimpleNamespace(content=[text_block])
        ),
    )
    failed = SimpleNamespace(custom_id="b", result=SimpleNamespace(type="errored"))
    rows = annotate.process_claude_batch_results(
        [failed, succeeded],
        {"a": "day-09/001_itinerary-1.jpg", "b": "day-09/002_itinerary-1.jpg"},
        None,
        annotate.DEFAULT_CLAUDE_MODEL,
        "batch",
    )
    assert [row["status"] for row in rows] == ["ok", "api_error"]
    with pytest.raises(ValueError, match="Unknown"):
        annotate.process_claude_batch_results(
            [succeeded], {}, None, annotate.DEFAULT_CLAUDE_MODEL, "batch"
        )
    with pytest.raises(ValueError, match="Duplicate"):
        annotate.process_claude_batch_results(
            [succeeded, succeeded],
            {"a": "one.jpg"},
            None,
            annotate.DEFAULT_CLAUDE_MODEL,
            "batch",
        )
    with pytest.raises(ValueError, match="omitted"):
        annotate.process_claude_batch_results(
            [succeeded],
            {"a": "one.jpg", "b": "two.jpg"},
            None,
            annotate.DEFAULT_CLAUDE_MODEL,
            "batch",
        )


class _FakeGeminiModels:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = iter(responses)

    def generate_content(self, **_kwargs: Any) -> Any:
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_gemini_sync_success_and_api_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    images = [tmp_path / "one.jpg", tmp_path / "two.png"]
    for image in images:
        image.write_bytes(b"image")
    responses = [
        SimpleNamespace(
            text='{"pm25": 85, "co2": 412, "status": "ok", "confidence": 0.9}'
        ),
        RuntimeError("offline"),
    ]
    client = SimpleNamespace(models=_FakeGeminiModels(responses))
    monkeypatch.setattr("google.genai.Client", lambda **_kwargs: client)
    rows = annotate.process_gemini_sync(images, api_key="key")
    assert [row["status"] for row in rows] == ["ok", "api_error"]
    with pytest.raises(ValueError, match="At least one"):
        annotate.process_gemini_sync([], api_key="key")


def test_gemini_batch_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image = tmp_path / "sensor.jpg"
    image.write_bytes(b"image")
    response = SimpleNamespace(
        text='{"pm25": 85, "co2": 412, "status": "ok", "confidence": 0.9}'
    )
    completed = SimpleNamespace(
        name="jobs/1",
        state=SimpleNamespace(name="JOB_STATE_SUCCEEDED"),
        dest=SimpleNamespace(inlined_responses=[SimpleNamespace(response=response)]),
    )
    batches = SimpleNamespace(
        create=lambda **_kwargs: SimpleNamespace(name="jobs/1"),
        get=lambda **_kwargs: completed,
    )
    monkeypatch.setattr(
        "google.genai.Client", lambda **_kwargs: SimpleNamespace(batches=batches)
    )
    rows = annotate.process_gemini_batch([image], api_key="key", poll_interval=1)
    assert rows[0]["status"] == "ok"


def test_gemini_batch_validates_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one"):
        annotate.process_gemini_batch([], api_key="key")
    image = tmp_path / "sensor.jpg"
    image.write_bytes(b"image")
    with pytest.raises(ValueError, match="at least one second"):
        annotate.process_gemini_batch([image], api_key="key", poll_interval=0)


def test_process_persists_parquet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = tmp_path / "day-09/001_itinerary-1.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")

    def fake_sync(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            annotate._result_row(
                image,
                {"pm25": 85.0, "co2": 412.0, "status": "ok", "confidence": 0.9},
                None,
                "google",
                annotate.DEFAULT_GEMINI_MODEL,
                "run",
            )
        ]

    monkeypatch.setattr(annotate, "process_gemini_sync", fake_sync)
    output = annotate.process([image], tmp_path / "output")
    assert load_ocr_results(output).loc[0, "pm25"] == 85.0
    with pytest.raises(ValueError, match="only valid for Claude"):
        annotate.process([image], tmp_path / "output", batch_id="wrong")


def test_find_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ["b.png", "a.jpg", "ignored.txt"]:
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    assert annotate.find_images("*") == [Path("a.jpg"), Path("b.png")]
