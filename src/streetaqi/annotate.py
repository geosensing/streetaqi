"""OCR air quality sensor readings from images using Claude or Gemini APIs."""

import base64
import glob as glob_module
import hashlib
import json
import os
import time
from pathlib import Path

POLLUTION_OCR_PROMPT = """You are analyzing images containing a handheld air quality sensor.
Your task is to read the PM2.5 and CO₂ values shown on the LCD display.

The sensor shows:
- PM2.5 value (μg/m³) - typically 2-3 digits
- CO₂ value (ppm) - typically 3-4 digits

Respond with ONLY a JSON object in this exact format:
{"pm25": <number or null>, "co": <number or null>, "status": "<string>", "confidence": <0.0-1.0>}

Status values:
- "ok": Sensor found and readings extracted successfully
- "sensor_not_found": No air quality sensor visible in the image
- "display_unreadable": Sensor visible but values cannot be read (blur, angle, glare)
- "image_unclear": Image too blurry or dark to analyze
- "partial_read": Only some values readable (fill in what you can)

Examples:
- Clear readings: {"pm25": 85, "co": 412, "status": "ok", "confidence": 0.95}
- Only PM2.5 visible: {"pm25": 92, "co": null, "status": "partial_read", "confidence": 0.75}
- Blurry image: {"pm25": null, "co": null, "status": "image_unclear", "confidence": 0.8}

Do not include any other text or explanation."""


def encode_image_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def read_image_bytes(image_path: Path) -> bytes:
    """Read image as bytes."""
    with open(image_path, "rb") as f:
        return f.read()


def is_gemini_model(model: str) -> bool:
    """Check if model is a Gemini model."""
    return model.startswith("gemini-")


def path_to_custom_id(image_path: str) -> str:
    """Convert image path to valid custom_id (alphanumeric, max 64 chars)."""
    return hashlib.sha256(image_path.encode()).hexdigest()[:64]


def parse_ocr_response(response_text: str) -> dict:
    """Parse OCR response JSON."""
    try:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = (
                "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
            )

        data = json.loads(text)
        return {
            "pm25": data.get("pm25"),
            "co": data.get("co"),
            "status": data.get("status", "unknown"),
            "confidence": data.get("confidence", 0.0),
        }
    except (json.JSONDecodeError, KeyError):
        return {"pm25": None, "co": None, "status": "parse_error", "confidence": 0.0}


def extract_metadata_from_path(image_path: Path) -> dict:
    """Extract day and itinerary info from image path."""
    parts = image_path.parts
    day = None
    itinerary_id = None

    for part in parts:
        if part.startswith("day-"):
            try:
                day = int(part.replace("day-", ""))
            except ValueError:
                pass

    name = image_path.stem
    if "_itinerary-" in name:
        try:
            itinerary_part = name.split("_itinerary-")[1]
            itinerary_id = int(itinerary_part.split("-")[0])
        except (ValueError, IndexError):
            pass

    return {"day": day, "itinerary_id": itinerary_id}


def load_manifest(manifest_path: Path) -> dict:
    """Load manifest and create lookup by image path."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    lookup = {}
    for log in manifest.get("pollution_logs", []):
        if "image" in log and "local_path" in log["image"]:
            lookup[log["image"]["local_path"]] = log

    return lookup


def create_batch_request(
    image_path: Path, custom_id: str, model: str = "claude-haiku-4-5"
) -> dict:
    """Create a single batch request for an image."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_data = encode_image_base64(image_path)

    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": 150,
            "system": POLLUTION_OCR_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Read the PM2.5 and CO₂ values from this air quality sensor image.",
                        },
                    ],
                }
            ],
        },
    }


def submit_claude_batch(
    images: list[Path],
    model: str = "claude-haiku-4-5",
    api_key: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Submit a batch of images for OCR processing via Claude Batch API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    requests = []
    id_to_path = {}

    for image_path in images:
        try:
            custom_id = path_to_custom_id(str(image_path))
            id_to_path[custom_id] = str(image_path)
            req = create_batch_request(image_path, custom_id, model)
            requests.append(req)
        except Exception as e:
            print(f"Warning: Failed to create request for {image_path}: {e}")

    if not requests:
        raise ValueError("No valid requests to submit")

    print(f"Submitting batch with {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)

    return batch.id, id_to_path


def poll_claude_batch(
    batch_id: str,
    api_key: str | None = None,
    poll_interval: int = 30,
) -> dict:
    """Poll Claude batch status until completion."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts

        print(
            f"Batch {batch_id}: {status} "
            f"(succeeded={counts.succeeded}, processing={counts.processing}, "
            f"errored={counts.errored})"
        )

        if status == "ended":
            return {
                "id": batch.id,
                "status": status,
                "results_url": batch.results_url,
                "request_counts": {
                    "succeeded": counts.succeeded,
                    "errored": counts.errored,
                    "canceled": counts.canceled,
                    "expired": counts.expired,
                    "processing": counts.processing,
                },
            }

        time.sleep(poll_interval)


def fetch_claude_batch_results(
    batch_id: str,
    api_key: str | None = None,
) -> list[dict]:
    """Fetch results from a completed Claude batch."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    results = []
    for result in client.messages.batches.results(batch_id):
        results.append(result)

    return results


def process_claude_batch_results(
    results: list,
    id_to_path: dict[str, str],
    manifest_lookup: dict | None = None,
) -> list[dict]:
    """Process Claude batch results and merge with metadata."""
    readings = []

    for result in results:
        custom_id = result.custom_id
        image_path = id_to_path.get(custom_id, custom_id)
        path_obj = Path(image_path)

        if result.result.type == "succeeded":
            message = result.result.message
            response_text = ""
            for block in message.content:
                if block.type == "text":
                    response_text = block.text
                    break
            reading = parse_ocr_response(response_text)
        else:
            reading = {
                "pm25": None,
                "co": None,
                "status": "api_error",
                "confidence": 0.0,
            }

        metadata = extract_metadata_from_path(path_obj)

        entry = {
            "image_path": image_path,
            "id": custom_id,
            "day": metadata["day"],
            "itinerary_id": metadata["itinerary_id"],
            "reading": reading,
        }

        if manifest_lookup:
            rel_path = None
            for key in manifest_lookup:
                if key.endswith(path_obj.name) or image_path.endswith(key):
                    rel_path = key
                    break
            if rel_path and rel_path in manifest_lookup:
                log = manifest_lookup[rel_path]
                entry["gps"] = {
                    "latitude": log.get("latitude"),
                    "longitude": log.get("longitude"),
                }
                entry["logged_pm25"] = log.get("pm25")
                entry["logged_co"] = log.get("co")

        readings.append(entry)

    return readings


def process_gemini_sync(
    images: list[Path],
    model: str = "gemini-2.0-flash",
    api_key: str | None = None,
    manifest_lookup: dict | None = None,
) -> list[dict]:
    """Process images synchronously using Gemini API."""
    from google import genai
    from google.genai import types

    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    readings = []
    total = len(images)

    for i, image_path in enumerate(images, 1):
        print(f"Processing {i}/{total}: {image_path.name}...", end=" ", flush=True)

        try:
            if not image_path.exists():
                print("SKIP (not found)")
                readings.append(
                    {
                        "image_path": str(image_path),
                        "id": path_to_custom_id(str(image_path)),
                        "reading": {
                            "pm25": None,
                            "co": None,
                            "status": "file_not_found",
                            "confidence": 0.0,
                        },
                    }
                )
                continue

            image_bytes = read_image_bytes(image_path)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

            response = client.models.generate_content(
                model=model,
                contents=[
                    POLLUTION_OCR_PROMPT,
                    image_part,
                    "Read the PM2.5 and CO₂ values from this air quality sensor image.",
                ],
            )

            response_text = response.text if response.text else ""
            reading = parse_ocr_response(response_text)
            print(f"{reading['status']}: PM2.5={reading['pm25']}, CO₂={reading['co']}")

        except Exception as e:
            print(f"ERROR: {e}")
            reading = {
                "pm25": None,
                "co": None,
                "status": "api_error",
                "confidence": 0.0,
            }

        metadata = extract_metadata_from_path(image_path)

        entry = {
            "image_path": str(image_path),
            "id": path_to_custom_id(str(image_path)),
            "day": metadata["day"],
            "itinerary_id": metadata["itinerary_id"],
            "reading": reading,
        }

        if manifest_lookup:
            for key in manifest_lookup:
                if key.endswith(image_path.name) or str(image_path).endswith(key):
                    log = manifest_lookup[key]
                    entry["gps"] = {
                        "latitude": log.get("latitude"),
                        "longitude": log.get("longitude"),
                    }
                    entry["logged_pm25"] = log.get("pm25")
                    entry["logged_co"] = log.get("co")
                    break

        readings.append(entry)

    return readings


def process_gemini_batch(
    images: list[Path],
    model: str = "gemini-2.0-flash",
    api_key: str | None = None,
    manifest_lookup: dict | None = None,
    poll_interval: int = 30,
) -> list[dict]:
    """Process images using Gemini Batch API (50% cost savings)."""
    from google import genai

    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    print(f"Preparing batch with {len(images)} images...")

    inline_requests = []
    image_list = []

    for image_path in images:
        if not image_path.exists():
            continue

        image_data = encode_image_base64(image_path)
        image_list.append(image_path)

        inline_requests.append(
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": POLLUTION_OCR_PROMPT},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_data,
                                }
                            },
                            {
                                "text": "Read the PM2.5 and CO₂ values from this air quality sensor image."
                            },
                        ],
                    }
                ],
            }
        )

    print(f"Submitting batch job with {len(inline_requests)} requests...")
    batch_job = client.batches.create(
        model=model,
        src=inline_requests,
        config={"display_name": f"pollution-ocr-{time.strftime('%Y%m%d_%H%M%S')}"},
    )
    print(f"Batch job created: {batch_job.name}")

    while True:
        batch_job = client.batches.get(name=batch_job.name)
        state = batch_job.state.name
        print(f"Batch status: {state}")

        if state == "JOB_STATE_SUCCEEDED":
            break
        elif state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            raise RuntimeError(f"Batch job failed with state: {state}")

        time.sleep(poll_interval)

    print("Processing results...")
    readings = []
    inlined_responses = batch_job.dest.inlined_responses or []

    for i, inlined_resp in enumerate(inlined_responses):
        image_path = image_list[i] if i < len(image_list) else None

        try:
            response = inlined_resp.response
            if response and response.candidates:
                parts = response.candidates[0].content.parts
                response_text = ""
                for part in parts:
                    if hasattr(part, "text") and part.text:
                        response_text = part.text
                        break
                reading = parse_ocr_response(response_text)
            else:
                reading = {
                    "pm25": None,
                    "co": None,
                    "status": "no_response",
                    "confidence": 0.0,
                }
        except Exception as e:
            print(f"Error parsing response {i}: {e}")
            reading = {
                "pm25": None,
                "co": None,
                "status": "parse_error",
                "confidence": 0.0,
            }

        metadata = extract_metadata_from_path(image_path) if image_path else {}

        entry = {
            "image_path": str(image_path) if image_path else "",
            "id": path_to_custom_id(str(image_path)) if image_path else str(i),
            "day": metadata.get("day"),
            "itinerary_id": metadata.get("itinerary_id"),
            "reading": reading,
        }

        if manifest_lookup and image_path:
            for key in manifest_lookup:
                if key.endswith(image_path.name) or str(image_path).endswith(key):
                    log = manifest_lookup[key]
                    entry["gps"] = {
                        "latitude": log.get("latitude"),
                        "longitude": log.get("longitude"),
                    }
                    entry["logged_pm25"] = log.get("pm25")
                    entry["logged_co"] = log.get("co")
                    break

        readings.append(entry)

    print(f"Processed {len(readings)} results")
    return readings


def save_readings(readings: list[dict], output_path: Path, batch_id: str) -> None:
    """Save readings to JSON file."""
    output = {
        "batch_id": batch_id,
        "reading_count": len(readings),
        "readings": readings,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def save_id_mapping(id_to_path: dict[str, str], output_path: Path) -> None:
    """Save ID to path mapping for later retrieval."""
    with open(output_path, "w") as f:
        json.dump(id_to_path, f, indent=2)


def load_id_mapping(mapping_path: Path) -> dict[str, str]:
    """Load ID to path mapping."""
    with open(mapping_path) as f:
        return json.load(f)


def process(
    images: list[Path],
    output_dir: Path,
    model: str = "gemini-2.0-flash",
    manifest_path: Path | None = None,
    batch_id: str | None = None,
    use_batch: bool = False,
    poll_interval: int = 30,
) -> Path:
    """Run OCR on images using Claude or Gemini API.

    For Claude models: uses batch API with polling.
    For Gemini models: processes synchronously by default, batch with --batch flag.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_lookup = None
    if manifest_path and manifest_path.exists():
        manifest_lookup = load_manifest(manifest_path)
        print(f"Loaded manifest with {len(manifest_lookup)} pollution log entries")

    model_short = (
        model.replace("claude-", "").replace("gemini-", "").replace("-4-5", "")
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"pollution_readings_{model_short}_{timestamp}.json"
    mapping_file = output_dir / f"pollution_id_mapping_{timestamp}.json"

    if is_gemini_model(model):
        print(f"Using Gemini model: {model}")
        print(f"Processing {len(images)} images...")

        if use_batch:
            readings = process_gemini_batch(
                images,
                model,
                manifest_lookup=manifest_lookup,
                poll_interval=poll_interval,
            )
        else:
            readings = process_gemini_sync(
                images, model, manifest_lookup=manifest_lookup
            )

        run_id = f"gemini_{timestamp}"
        save_readings(readings, output_file, run_id)
        print(f"Saved {len(readings)} readings -> {output_file}")
        return output_file

    print(f"Using Claude model: {model}")
    print(f"Processing {len(images)} images via Batch API...")

    if batch_id:
        print(f"Retrieving results for batch {batch_id}...")
        if mapping_file.exists():
            id_to_path = load_id_mapping(mapping_file)
        else:
            id_to_path = {path_to_custom_id(str(p)): str(p) for p in images}
    else:
        batch_id, id_to_path = submit_claude_batch(images, model)
        print(f"Submitted batch: {batch_id}")

        save_id_mapping(id_to_path, mapping_file)
        print(f"Saved ID mapping -> {mapping_file}")

        print("Polling for completion...")
        poll_claude_batch(batch_id, poll_interval=poll_interval)

    results = fetch_claude_batch_results(batch_id)
    readings = process_claude_batch_results(results, id_to_path, manifest_lookup)
    save_readings(readings, output_file, batch_id)

    print(f"Saved {len(readings)} readings -> {output_file}")
    return output_file


def find_images(pattern: str) -> list[Path]:
    """Find images matching glob pattern."""
    paths = glob_module.glob(pattern, recursive=True)
    return [
        Path(p) for p in sorted(paths) if p.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
