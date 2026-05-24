"""Generate HTML viewer for pollution OCR results verification with correction capability."""

import base64
import json
from pathlib import Path


def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 data URI."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def normalize_readings(data: dict) -> list[dict]:
    """Normalize readings to a common format for the viewer.

    Handles both OCR output format and raw readings.json format.
    """
    readings = data.get("readings", [])
    if not readings:
        return []

    first = readings[0]
    is_ocr_format = (
        "reading" in first
        and isinstance(first["reading"], dict)
        and "status" in first.get("reading", {})
    )

    if is_ocr_format:
        return readings

    normalized = []

    for r in readings:
        reading_data = r.get("reading", {})
        metadata = r.get("metadata", {})

        frame_path = r.get("frame_path", "")
        if frame_path:
            img_path = Path(frame_path)
        else:
            local_path = r.get("image", {}).get("local_path", "")
            img_path = Path(local_path) if local_path else Path("")

        pm25 = reading_data.get("pm25")
        co = reading_data.get("co")

        if pm25 is not None and co is not None:
            status = "ok"
        elif pm25 is not None or co is not None:
            status = "partial_read"
        else:
            status = "display_unreadable"

        normalized.append(
            {
                "id": r.get("id", ""),
                "image_path": str(img_path),
                "day": metadata.get("day", r.get("day", "?")),
                "itinerary_id": metadata.get("itinerary", r.get("itinerary_id", "?")),
                "gps": r.get("gps", {}),
                "reading": {
                    "pm25": pm25,
                    "co": co,
                    "status": status,
                    "confidence": 1.0,
                },
                "logged_pm25": pm25,
                "logged_co": co,
                "metadata": metadata,
            }
        )

    return normalized


def generate_html(readings_path: Path, output_path: Path | None = None) -> Path:
    """Generate HTML viewer for OCR results with editable corrections."""
    with open(readings_path) as f:
        data = json.load(f)

    readings = normalize_readings(data)

    if output_path is None:
        output_path = readings_path.with_suffix(".html")

    total = len(readings)
    ok_count = sum(1 for r in readings if r["reading"]["status"] == "ok")
    partial_count = sum(1 for r in readings if r["reading"]["status"] == "partial_read")
    fail_count = total - ok_count - partial_count

    readings_json = json.dumps(readings)

    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Pollution OCR Results Viewer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: #2d5a27;
            color: white;
            padding: 20px;
            margin: -20px -20px 20px -20px;
        }
        .header h1 { margin: 0 0 10px 0; }
        .stats {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .stat {
            background: rgba(255,255,255,0.1);
            padding: 10px 20px;
            border-radius: 5px;
        }
        .stat-value { font-size: 24px; font-weight: bold; }
        .stat-label { font-size: 12px; opacity: 0.8; }
        .filters {
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        .filters label {
            margin-right: 20px;
            cursor: pointer;
        }
        .export-btn {
            background: #2d5a27;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .export-btn:hover { background: #1e3d1a; }
        .card {
            background: white;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .card.hidden { display: none; }
        .card.corrected { border-left: 4px solid #28a745; }
        .card-header {
            padding: 10px 15px;
            background: #f8f8f8;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .card-header h3 { margin: 0; font-size: 14px; }
        .status {
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        .status-ok { background: #d4edda; color: #155724; }
        .status-partial_read { background: #fff3cd; color: #856404; }
        .status-display_unreadable { background: #f8d7da; color: #721c24; }
        .status-sensor_not_found { background: #f8d7da; color: #721c24; }
        .status-other { background: #e2e3e5; color: #383d41; }
        .card-body {
            display: flex;
            padding: 15px;
            gap: 20px;
        }
        .image-container {
            flex: 0 0 350px;
        }
        .image-container img {
            width: 350px;
            height: auto;
            border-radius: 4px;
        }
        .annotation {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .readings {
            display: flex;
            gap: 30px;
            margin-bottom: 10px;
        }
        .reading-box {
            text-align: center;
        }
        .reading-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        .reading-value {
            font-size: 36px;
            font-weight: bold;
            color: #333;
        }
        .reading-value.null { color: #999; font-size: 18px; }
        .reading-unit {
            font-size: 14px;
            color: #666;
        }
        .logged-comparison {
            font-size: 13px;
            color: #666;
            margin-top: 5px;
            padding: 8px;
            background: #f0f0f0;
            border-radius: 4px;
        }
        .logged-comparison .match { color: #28a745; }
        .logged-comparison .mismatch { color: #dc3545; }
        .confidence {
            font-size: 14px;
            color: #666;
            margin-top: 10px;
        }
        .correction {
            margin-top: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .correction label {
            font-size: 12px;
            color: #666;
            display: block;
            margin-bottom: 5px;
        }
        .correction-inputs {
            display: flex;
            gap: 15px;
        }
        .correction-field {
            display: flex;
            flex-direction: column;
        }
        .correction-field label {
            font-size: 11px;
            margin-bottom: 3px;
        }
        .correction input {
            width: 80px;
            padding: 8px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .correction input:focus {
            outline: none;
            border-color: #2d5a27;
        }
        .correction input.has-value {
            border-color: #28a745;
            background: #f0fff4;
        }
        .metadata {
            font-size: 12px;
            color: #999;
            margin-top: 15px;
        }
        .metadata div { margin: 3px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Pollution OCR Results Viewer</h1>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">""",
        str(total),
        """</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat">
                <div class="stat-value">""",
        str(ok_count),
        """</div>
                <div class="stat-label">OK</div>
            </div>
            <div class="stat">
                <div class="stat-value">""",
        str(partial_count),
        """</div>
                <div class="stat-label">Partial</div>
            </div>
            <div class="stat">
                <div class="stat-value">""",
        str(fail_count),
        """</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="corrected-count">0</div>
                <div class="stat-label">Corrected</div>
            </div>
        </div>
    </div>

    <div class="filters">
        <div>
            <strong>Filter:</strong>
            <label><input type="checkbox" checked onchange="filterCards()"
                id="filter-ok"> OK</label>
            <label><input type="checkbox" checked onchange="filterCards()"
                id="filter-partial"> Partial</label>
            <label><input type="checkbox" checked onchange="filterCards()"
                id="filter-failed"> Failed</label>
        </div>
        <div>
            <button class="export-btn" onclick="exportGroundTruth('json')">Export JSON</button>
            <button class="export-btn" onclick="exportGroundTruth('csv')">Export CSV</button>
        </div>
    </div>

    <div id="cards">
""",
    ]

    for i, r in enumerate(readings):
        reading = r["reading"]
        status = reading["status"]
        pm25 = reading.get("pm25")
        co = reading.get("co")
        confidence = reading.get("confidence", 0)
        day = r.get("day", "?")
        itinerary_id = r.get("itinerary_id", "?")
        image_path = r.get("image_path", "")

        logged_pm25 = r.get("logged_pm25")
        logged_co = r.get("logged_co")

        img_path = Path(image_path)
        if img_path.exists():
            img_data = image_to_base64(img_path)
        else:
            img_data = ""

        known_statuses = [
            "ok",
            "partial_read",
            "display_unreadable",
            "sensor_not_found",
        ]
        status_class = (
            f"status-{status}" if status in known_statuses else "status-other"
        )

        if status == "ok":
            filter_status = "ok"
        elif status == "partial_read":
            filter_status = "partial"
        else:
            filter_status = "failed"

        if pm25 is not None:
            pm25_display = str(pm25)
            pm25_class = ""
        else:
            pm25_display = "[--]"
            pm25_class = "null"

        if co is not None:
            co_display = str(co)
            co_class = ""
        else:
            co_display = "[--]"
            co_class = "null"

        gps = r.get("gps", {})
        lat = gps.get("latitude", "N/A") if gps else "N/A"
        lon = gps.get("longitude", "N/A") if gps else "N/A"

        comparison_html = ""
        if logged_pm25 is not None or logged_co is not None:
            pm25_match = (
                pm25 == logged_pm25
                if pm25 is not None and logged_pm25 is not None
                else None
            )
            co_match = (
                co == logged_co if co is not None and logged_co is not None else None
            )

            pm25_class_cmp = (
                "match" if pm25_match else ("mismatch" if pm25_match is False else "")
            )
            co_class_cmp = (
                "match" if co_match else ("mismatch" if co_match is False else "")
            )

            comparison_html = f"""
                    <div class="logged-comparison">
                        <strong>Logged values:</strong>
                        PM2.5: <span class="{pm25_class_cmp}">{logged_pm25 if logged_pm25 is not None else "--"}</span> |
                        CO₂: <span class="{co_class_cmp}">{logged_co if logged_co is not None else "--"}</span>
                    </div>"""

        lat_str = f"{lat:.6f}" if isinstance(lat, (int, float)) else str(lat)
        lon_str = f"{lon:.6f}" if isinstance(lon, (int, float)) else str(lon)

        html_parts.append(
            f"""
        <div class="card" data-status="{filter_status}" data-index="{i}">
            <div class="card-header">
                <h3>#{i + 1} - Day {day}, Itinerary {itinerary_id}</h3>
                <span class="status {status_class}">{status}</span>
            </div>
            <div class="card-body">
                <div class="image-container">
                    <img src="{img_data}" alt="Air quality sensor reading">
                </div>
                <div class="annotation">
                    <div class="readings">
                        <div class="reading-box">
                            <div class="reading-label">PM2.5</div>
                            <div class="reading-value {pm25_class}">{pm25_display}</div>
                            <div class="reading-unit">μg/m³</div>
                        </div>
                        <div class="reading-box">
                            <div class="reading-label">CO₂</div>
                            <div class="reading-value {co_class}">{co_display}</div>
                            <div class="reading-unit">ppm</div>
                        </div>
                    </div>{comparison_html}
                    <div class="confidence">Confidence: {confidence:.0%}</div>
                    <div class="correction">
                        <label>Correct readings (-1 = unreadable):</label>
                        <div class="correction-inputs">
                            <div class="correction-field">
                                <label>PM2.5</label>
                                <input type="number" step="1" placeholder="e.g. 85"
                                       data-index="{i}" data-field="pm25" onchange="markCorrected(this)">
                            </div>
                            <div class="correction-field">
                                <label>CO₂</label>
                                <input type="number" step="1" placeholder="e.g. 412"
                                       data-index="{i}" data-field="co" onchange="markCorrected(this)">
                            </div>
                        </div>
                    </div>
                    <div class="metadata">
                        <div>Day: {day} | Itinerary: {itinerary_id}</div>
                        <div>GPS: {lat_str}, {lon_str}</div>
                        <div>Path: {Path(image_path).name}</div>
                    </div>
                </div>
            </div>
        </div>
"""
        )

    html_parts.append(
        """
    </div>

    <script>
        const originalReadings = """
        + readings_json
        + """;

        function filterCards() {
            const showOk = document.getElementById('filter-ok').checked;
            const showPartial = document.getElementById('filter-partial').checked;
            const showFailed = document.getElementById('filter-failed').checked;

            document.querySelectorAll('.card').forEach(card => {
                const status = card.dataset.status;
                const show = (status === 'ok' && showOk) ||
                             (status === 'partial' && showPartial) ||
                             (status === 'failed' && showFailed);
                card.classList.toggle('hidden', !show);
            });
        }

        function markCorrected(input) {
            const card = input.closest('.card');
            const idx = parseInt(input.dataset.index);

            if (input.value && input.value.trim() !== '') {
                input.classList.add('has-value');
            } else {
                input.classList.remove('has-value');
            }

            // Check if any input in this card has a value
            const cardInputs = card.querySelectorAll('.correction input');
            const hasAnyValue = Array.from(cardInputs).some(inp => inp.value && inp.value.trim() !== '');
            card.classList.toggle('corrected', hasAnyValue);

            updateCorrectedCount();
        }

        function updateCorrectedCount() {
            const count = document.querySelectorAll('.card.corrected').length;
            document.getElementById('corrected-count').textContent = count;
        }

        function exportGroundTruth(format) {
            const groundTruth = [];
            let correctionCount = 0;

            document.querySelectorAll('.card').forEach((card, idx) => {
                const pm25Input = card.querySelector('input[data-field="pm25"]');
                const coInput = card.querySelector('input[data-field="co"]');
                const original = originalReadings[idx];

                const pm25Corrected = pm25Input.value.trim();
                const coCorrected = coInput.value.trim();
                const hasPm25Correction = pm25Corrected !== '';
                const hasCoCoorection = coCorrected !== '';
                const hasAnyCorrention = hasPm25Correction || hasCoCoorection;

                if (hasAnyCorrention) correctionCount++;

                let finalPm25, finalCo, finalStatus;

                if (hasPm25Correction) {
                    const numValue = parseFloat(pm25Corrected);
                    finalPm25 = numValue === -1 ? null : numValue;
                } else {
                    finalPm25 = original.reading ? original.reading.pm25 : null;
                }

                if (hasCoCoorection) {
                    const numValue = parseFloat(coCorrected);
                    finalCo = numValue === -1 ? null : numValue;
                } else {
                    finalCo = original.reading ? original.reading.co : null;
                }

                if (finalPm25 !== null && finalCo !== null) {
                    finalStatus = 'ok';
                } else if (finalPm25 !== null || finalCo !== null) {
                    finalStatus = 'partial_read';
                } else {
                    finalStatus = original.reading ? original.reading.status : 'display_unreadable';
                }

                const gps = original.gps || {};

                groundTruth.push({
                    image_path: original.image_path,
                    id: original.id,
                    day: original.day,
                    itinerary_id: original.itinerary_id,
                    latitude: gps.latitude || null,
                    longitude: gps.longitude || null,
                    ocr_pm25: original.reading ? original.reading.pm25 : null,
                    ocr_co: original.reading ? original.reading.co : null,
                    ocr_status: original.reading ? original.reading.status : null,
                    ocr_confidence: original.reading ? original.reading.confidence : null,
                    logged_pm25: original.logged_pm25 || null,
                    logged_co: original.logged_co || null,
                    final_pm25: finalPm25,
                    final_co: finalCo,
                    final_status: finalStatus,
                    was_corrected: hasAnyCorrention
                });
            });

            const dateStr = new Date().toISOString().slice(0,10);

            if (format === 'csv') {
                const headers = ['image_path', 'id', 'day', 'itinerary_id',
                    'latitude', 'longitude', 'ocr_pm25', 'ocr_co',
                    'ocr_status', 'ocr_confidence', 'logged_pm25', 'logged_co',
                    'final_pm25', 'final_co', 'final_status', 'was_corrected'];
                const rows = groundTruth.map(r =>
                    headers.map(h => r[h] === null ? '' : r[h]).join(',')
                );
                const csv = headers.join(',') + '\\n' + rows.join('\\n');

                const blob = new Blob([csv], {type: 'text/csv'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'pollution_ground_truth_' + dateStr + '.csv';
                a.click();
                URL.revokeObjectURL(url);
            } else {
                const output = {
                    export_time: new Date().toISOString(),
                    total_count: groundTruth.length,
                    correction_count: correctionCount,
                    readings: groundTruth
                };

                const jsonStr = JSON.stringify(output, null, 2);
                const blob = new Blob([jsonStr], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'pollution_ground_truth_' + dateStr + '.json';
                a.click();
                URL.revokeObjectURL(url);
            }
        }
    </script>
</body>
</html>
"""
    )

    html = "".join(html_parts)

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def process(
    readings_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Generate HTML viewer."""
    output = generate_html(readings_path, output_path)
    print(f"Generated viewer: {output}")
    return output
