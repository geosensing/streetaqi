"""Extract air-quality sensor readings from images with supported LLM APIs."""

from __future__ import annotations

import base64
import glob as glob_module
import hashlib
import json
import logging
import math
import time
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from streetaqi.data import (
    MODEL_OCR_STATUSES,
    load_id_mapping,
    load_readings,
    validate_ocr_measurements,
    write_id_mapping,
    write_ocr_results,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"

OCR_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pm25", "co2", "status", "confidence"],
    "properties": {
        "pm25": {"anyOf": [{"type": "number", "minimum": 0}, {"type": "null"}]},
        "co2": {"anyOf": [{"type": "number", "minimum": 0}, {"type": "null"}]},
        "status": {"type": "string", "enum": sorted(MODEL_OCR_STATUSES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

POLLUTION_OCR_PROMPT = """Read the PM2.5 and CO2 values shown on the handheld air-quality sensor display.

Return null for a value that is not readable. Use status "ok" when both values
are readable, "partial_read" when one is readable, "sensor_not_found" when no
sensor is visible, "display_unreadable" when the sensor is visible but its
display cannot be read, and "image_unclear" when the image itself is unusable.
Confidence must describe confidence in the returned status and values. Return
only the JSON object defined by the response schema, without prose or Markdown."""


def image_mime_type(image_path: Path) -> str:
    """Return the supported MIME type for an image path."""
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    raise ValueError(f"Unsupported image type: {image_path.suffix or '(none)'}")


def encode_image_base64(image_path: Path) -> str:
    """Encode an image as base64 text."""
    return base64.standard_b64encode(image_path.read_bytes()).decode("ascii")


def is_gemini_model(model: str) -> bool:
    """Return whether a model ID targets Gemini."""
    return model.startswith("gemini-")


def path_to_custom_id(image_path: str) -> str:
    """Create a stable, valid batch custom ID from an image path."""
    return hashlib.sha256(image_path.encode()).hexdigest()


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("OCR values must be numbers or null")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("OCR values must be finite and non-negative")
    return parsed


def parse_ocr_response(response_text: str) -> dict[str, Any]:
    """Parse and validate an OCR response.

    Malformed responses become an explicit ``parse_error`` result rather than
    leaking partially valid values into the analytical data.
    """
    try:
        start = response_text.index("{")
        end = response_text.rindex("}") + 1
        payload = json.loads(response_text[start:end])
        if not isinstance(payload, dict):
            raise ValueError("OCR response must be an object")
        status = payload["status"]
        confidence = payload["confidence"]
        if status not in MODEL_OCR_STATUSES:
            raise ValueError("Unknown OCR status")
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise ValueError("Confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        pm25 = _parse_number(payload["pm25"])
        co2 = _parse_number(payload["co2"])
        validate_ocr_measurements(pm25, co2, status)
        return {
            "pm25": pm25,
            "co2": co2,
            "status": status,
            "confidence": confidence,
        }
    except (KeyError, ValueError, json.JSONDecodeError):
        return {
            "pm25": None,
            "co2": None,
            "status": "parse_error",
            "confidence": 0.0,
        }


def extract_metadata_from_path(image_path: Path) -> dict[str, int | None]:
    """Extract day and itinerary identifiers encoded in an image path."""
    day = None
    for part in image_path.parts:
        if part.startswith("day-"):
            try:
                day = int(part.removeprefix("day-"))
            except ValueError:
                continue

    itinerary_id = None
    if "_itinerary-" in image_path.stem:
        with suppress(IndexError, ValueError):
            itinerary_id = int(
                image_path.stem.split("_itinerary-", maxsplit=1)[1].split(
                    "-", maxsplit=1
                )[0]
            )
    return {"day": day, "itinerary_id": itinerary_id}


def load_reference_readings(reference_path: Path | None) -> pd.DataFrame | None:
    """Load optional canonical reference readings for OCR comparison."""
    return load_readings(reference_path) if reference_path is not None else None


def _reference_for_image(
    image_path: Path,
    reference: pd.DataFrame | None,
) -> dict[str, float | None]:
    if reference is None:
        return {
            "latitude": None,
            "longitude": None,
            "reference_pm25": None,
            "reference_co2": None,
        }

    candidate_parts = PurePosixPath(image_path.as_posix()).parts

    def is_component_suffix(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
        return bool(right) and len(left) >= len(right) and left[-len(right) :] == right

    matches = reference[
        reference["image_local_path"].map(
            lambda value: (
                is_component_suffix(candidate_parts, PurePosixPath(str(value)).parts)
                or is_component_suffix(PurePosixPath(str(value)).parts, candidate_parts)
            )
        )
    ]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous reference image path: {image_path}")
    if matches.empty:
        return {
            "latitude": None,
            "longitude": None,
            "reference_pm25": None,
            "reference_co2": None,
        }
    row = matches.iloc[0]
    return {
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "reference_pm25": float(row["pm25"]),
        "reference_co2": float(row["co2"]),
    }


def _result_row(
    image_path: Path,
    reading: dict[str, Any],
    reference: pd.DataFrame | None,
    provider: str,
    model: str,
    batch_id: str,
) -> dict[str, Any]:
    metadata = extract_metadata_from_path(image_path)
    return {
        "id": path_to_custom_id(str(image_path)),
        "image_path": str(image_path),
        "day": metadata["day"],
        "itinerary_id": metadata["itinerary_id"],
        **_reference_for_image(image_path, reference),
        **reading,
        "provider": provider,
        "model": model,
        "batch_id": batch_id,
    }


def create_claude_batch_request(
    image_path: Path,
    custom_id: str,
    model: str = DEFAULT_CLAUDE_MODEL,
) -> dict[str, Any]:
    """Build one Claude Message Batches API request."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": 150,
            "system": POLLUTION_OCR_PROMPT,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": OCR_RESPONSE_SCHEMA,
                }
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_mime_type(image_path),
                                "data": encode_image_base64(image_path),
                            },
                        },
                        {"type": "text", "text": "Read the sensor display."},
                    ],
                }
            ],
        },
    }


def submit_claude_batch(
    images: list[Path],
    model: str = DEFAULT_CLAUDE_MODEL,
    api_key: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Submit images to the Claude Message Batches API."""
    import anthropic

    if not images:
        raise ValueError("At least one image is required")
    mapping = {path_to_custom_id(str(path)): str(path) for path in images}
    requests: Any = [
        create_claude_batch_request(Path(path), custom_id, model)
        for custom_id, path in mapping.items()
    ]
    client = anthropic.Anthropic(api_key=api_key)
    batch = client.messages.batches.create(requests=requests)
    return batch.id, mapping


def poll_claude_batch(
    batch_id: str,
    api_key: str | None = None,
    poll_interval: int = 30,
) -> None:
    """Poll a Claude batch until all requests reach a terminal state."""
    import anthropic

    if poll_interval < 1:
        raise ValueError("poll_interval must be at least one second")
    client = anthropic.Anthropic(api_key=api_key)
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        LOGGER.info("Claude batch %s: %s", batch_id, batch.processing_status)
        if batch.processing_status == "ended":
            return
        time.sleep(poll_interval)


def fetch_claude_batch_results(
    batch_id: str,
    api_key: str | None = None,
) -> list[Any]:
    """Fetch all results from a completed Claude batch."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    return list(client.messages.batches.results(batch_id))


def process_claude_batch_results(
    results: list[Any],
    id_to_path: dict[str, str],
    reference: pd.DataFrame | None,
    model: str,
    batch_id: str,
) -> list[dict[str, Any]]:
    """Normalize Claude batch responses into canonical OCR rows."""
    rows_by_id = {}
    for result in results:
        if result.custom_id not in id_to_path:
            raise ValueError(f"Unknown Claude batch custom ID: {result.custom_id}")
        if result.custom_id in rows_by_id:
            raise ValueError(f"Duplicate Claude batch custom ID: {result.custom_id}")
        image_path = Path(id_to_path[result.custom_id])
        if result.result.type == "succeeded":
            text = next(
                (
                    block.text
                    for block in result.result.message.content
                    if block.type == "text"
                ),
                "",
            )
            reading = parse_ocr_response(text)
        else:
            reading = {
                "pm25": None,
                "co2": None,
                "status": "api_error",
                "confidence": 0.0,
            }
        rows_by_id[result.custom_id] = _result_row(
            image_path, reading, reference, "anthropic", model, batch_id
        )
    missing = set(id_to_path).difference(rows_by_id)
    if missing:
        raise ValueError(f"Claude batch omitted {len(missing)} result(s)")
    return [rows_by_id[custom_id] for custom_id in id_to_path]


def _gemini_config() -> Any:
    return {
        "response_mime_type": "application/json",
        "response_json_schema": OCR_RESPONSE_SCHEMA,
        "max_output_tokens": 150,
    }


def process_gemini_sync(
    images: list[Path],
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    reference: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Process images synchronously with the Gemini GenerateContent API."""
    from google import genai
    from google.genai import types

    if not images:
        raise ValueError("At least one image is required")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    run_id = f"gemini-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    rows = []
    for image_path in images:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = types.Part.from_bytes(
            data=image_path.read_bytes(),
            mime_type=image_mime_type(image_path),
        )
        try:
            response = client.models.generate_content(
                model=model,
                contents=[POLLUTION_OCR_PROMPT, image],
                config=_gemini_config(),
            )
            reading = parse_ocr_response(response.text or "")
        except Exception:  # API errors are persisted without leaking credentials.
            LOGGER.exception("Gemini request failed for %s", image_path)
            reading = {
                "pm25": None,
                "co2": None,
                "status": "api_error",
                "confidence": 0.0,
            }
        rows.append(
            _result_row(image_path, reading, reference, "google", model, run_id)
        )
    return rows


def process_gemini_batch(
    images: list[Path],
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: str | None = None,
    reference: pd.DataFrame | None = None,
    poll_interval: int = 30,
) -> list[dict[str, Any]]:
    """Process images with the Gemini Batch API."""
    from google import genai

    if not images:
        raise ValueError("At least one image is required")
    if poll_interval < 1:
        raise ValueError("poll_interval must be at least one second")
    for image_path in images:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    requests: Any = [
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": POLLUTION_OCR_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": image_mime_type(image_path),
                                "data": encode_image_base64(image_path),
                            }
                        },
                    ],
                }
            ],
            "config": _gemini_config(),
        }
        for image_path in images
    ]
    job = client.batches.create(
        model=model,
        src=requests,
        config={
            "display_name": f"streetaqi-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        },
    )
    job_name = job.name
    if job_name is None:
        raise RuntimeError("Gemini batch response did not include a job name")
    while True:
        job = client.batches.get(name=job_name)
        if job.state is None:
            raise RuntimeError("Gemini batch response did not include a state")
        state = job.state.name
        LOGGER.info("Gemini batch %s: %s", job_name, state)
        if state == "JOB_STATE_SUCCEEDED":
            break
        if state in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
            raise RuntimeError(f"Gemini batch ended with state: {state}")
        time.sleep(poll_interval)

    if job.dest is None:
        raise RuntimeError("Gemini batch response did not include a destination")
    responses = job.dest.inlined_responses or []
    if len(responses) != len(images):
        raise RuntimeError(
            f"Gemini returned {len(responses)} responses for {len(images)} images"
        )
    rows = []
    for image_path, inline_response in zip(images, responses, strict=True):
        response = inline_response.response
        reading = parse_ocr_response(
            response.text if response and response.text else ""
        )
        rows.append(
            _result_row(image_path, reading, reference, "google", model, job_name)
        )
    return rows


def process(
    images: list[Path],
    output_dir: Path,
    model: str = DEFAULT_GEMINI_MODEL,
    reference_path: Path | None = None,
    batch_id: str | None = None,
    use_batch: bool = False,
    poll_interval: int = 30,
) -> Path:
    """Run OCR and persist canonical results as Parquet."""
    if not images:
        raise ValueError("At least one image is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_reference_readings(reference_path)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    if is_gemini_model(model):
        if batch_id is not None:
            raise ValueError("batch_id is only valid for Claude batches")
        rows = (
            process_gemini_batch(
                images,
                model,
                reference=reference,
                poll_interval=poll_interval,
            )
            if use_batch
            else process_gemini_sync(images, model, reference=reference)
        )
    else:
        if use_batch:
            raise ValueError("Claude already uses the Message Batches API")
        if batch_id is None:
            batch_id, mapping = submit_claude_batch(images, model)
            write_id_mapping(mapping, output_dir / f"claude_mapping_{batch_id}.parquet")
            poll_claude_batch(batch_id, poll_interval=poll_interval)
        else:
            mapping_path = output_dir / f"claude_mapping_{batch_id}.parquet"
            if not mapping_path.is_file():
                raise FileNotFoundError(f"Batch mapping not found: {mapping_path}")
            mapping = load_id_mapping(mapping_path)
        results = fetch_claude_batch_results(batch_id)
        rows = process_claude_batch_results(
            results,
            mapping,
            reference,
            model,
            batch_id,
        )

    output_path = output_dir / f"ocr_results_{timestamp}.parquet"
    return write_ocr_results(pd.DataFrame(rows), output_path)


def find_images(pattern: str) -> list[Path]:
    """Return supported images matching a recursive glob pattern."""
    return [
        Path(path)
        for path in sorted(glob_module.glob(pattern, recursive=True))  # noqa: PTH207
        if Path(path).suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
