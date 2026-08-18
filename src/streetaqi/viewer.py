"""Generate a safe local quality-control viewer for OCR results."""

from __future__ import annotations

import base64
import json
from html import escape
from pathlib import Path

import pandas as pd

from streetaqi.data import load_ocr_results


def image_to_data_uri(image_path: Path, image_root: Path | None) -> str | None:
    """Encode an image only when it is contained by an explicit trusted root."""
    if image_root is None:
        return None
    root = image_root.resolve(strict=True)
    if image_path.is_absolute():
        candidate = image_path
    else:
        recorded = image_path.resolve()
        candidate = recorded if recorded.is_relative_to(root) else root / image_path
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return None
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Image is outside the trusted root: {image_path}")
    suffix = resolved.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif suffix == ".png":
        mime_type = "image/png"
    else:
        raise ValueError(f"Unsupported image type: {resolved.suffix}")
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _json_records(frame: pd.DataFrame) -> str:
    records = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
    return json.dumps(records, ensure_ascii=False).replace("</", "<\\/")


def generate_html(
    readings_path: Path,
    output_path: Path | None = None,
    image_root: Path | None = None,
) -> Path:
    """Generate an HTML quality-control viewer from typed OCR results.

    Args:
        readings_path: Canonical OCR-result Parquet file.
        output_path: Destination HTML path. Defaults beside the input.
        image_root: Trusted directory from which image bytes may be embedded.

    Returns:
        The generated HTML path.
    """
    readings = load_ocr_results(readings_path)
    if output_path is None:
        output_path = readings_path.with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    status_counts = readings["status"].value_counts()
    original_json = _json_records(readings)
    cards = []
    for index, (_, row) in enumerate(readings.iterrows()):
        status = str(row["status"])
        filter_status = (
            "ok"
            if status == "ok"
            else "partial"
            if status == "partial_read"
            else "failed"
        )
        image_path = Path(str(row["image_path"]))
        data_uri = image_to_data_uri(image_path, image_root)
        image_html = (
            f'<img src="{data_uri}" alt="Air-quality sensor">'
            if data_uri
            else '<div class="image-missing">Image not embedded</div>'
        )

        pm25 = None if pd.isna(row["pm25"]) else float(row["pm25"])
        co2 = None if pd.isna(row["co2"]) else float(row["co2"])
        pm25_display = "--" if pm25 is None else f"{pm25:g}"
        co2_display = "--" if co2 is None else f"{co2:g}"
        day = "?" if pd.isna(row["day"]) else str(int(row["day"]))
        itinerary = (
            "?" if pd.isna(row["itinerary_id"]) else str(int(row["itinerary_id"]))
        )
        latitude = "N/A" if pd.isna(row["latitude"]) else f"{row['latitude']:.6f}"
        longitude = "N/A" if pd.isna(row["longitude"]) else f"{row['longitude']:.6f}"

        cards.append(
            f"""
            <article class="card" data-status="{filter_status}" data-index="{index}">
              <header><strong>#{index + 1} · Day {escape(day)} · Itinerary {escape(itinerary)}</strong>
                <span class="status status-{escape(filter_status)}">{escape(status)}</span></header>
              <div class="body"><div class="image">{image_html}</div><div class="details">
                <div class="values"><div><b>{escape(pm25_display)}</b><small>PM2.5 μg/m³</small></div>
                  <div><b>{escape(co2_display)}</b><small>CO2 ppm</small></div></div>
                <p>Confidence: {float(row["confidence"]):.0%}</p>
                <div class="corrections">
                  <label>Correct PM2.5 <input type="number" min="0" step="any" data-field="pm25"></label>
                  <label>Correct CO2 <input type="number" min="0" step="any" data-field="co2"></label>
                  <label><input type="checkbox" data-field="unreadable"> Display unreadable</label>
                </div>
                <small>GPS: {escape(latitude)}, {escape(longitude)}<br>File: {escape(image_path.name)}</small>
              </div></div>
            </article>"""
        )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>streetaqi OCR quality control</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f3f5f4;color:#17221a}}header.top{{background:#245b35;color:white;padding:1.25rem 2rem}}main{{max-width:1100px;margin:auto;padding:1rem}}
.stats,.filters,.values,.body{{display:flex;gap:1rem;flex-wrap:wrap}}.stats span{{background:#ffffff22;padding:.5rem .8rem;border-radius:.4rem}}.filters{{background:white;padding:1rem;border-radius:.5rem;margin-bottom:1rem;align-items:center}}
.card{{background:white;border-radius:.5rem;margin:0 0 1rem;box-shadow:0 1px 4px #0002;overflow:hidden}}.card>header{{display:flex;justify-content:space-between;background:#edf1ee;padding:.75rem 1rem}}.body{{padding:1rem}}.image{{flex:0 0 360px}}.image img{{max-width:360px;max-height:300px}}.image-missing{{height:180px;display:grid;place-items:center;background:#eee;color:#666}}.details{{flex:1;min-width:260px}}
.values div{{text-align:center;min-width:120px}}.values b{{display:block;font-size:2rem}}.values small{{display:block}}.status{{padding:.2rem .6rem;border-radius:1rem}}.status-ok{{background:#d5f1dd}}.status-partial{{background:#fff0bd}}.status-failed{{background:#f7d6d6}}.corrections{{display:grid;gap:.5rem;margin:1rem 0}}.hidden{{display:none}}button{{background:#245b35;color:white;border:0;border-radius:.35rem;padding:.6rem 1rem;cursor:pointer}}
</style></head><body>
<header class="top"><h1>streetaqi OCR quality control</h1><div class="stats">
<span>{len(readings)} total</span><span>{int(status_counts.get("ok", 0))} OK</span><span>{int(status_counts.get("partial_read", 0))} partial</span><span>{int(len(readings) - status_counts.get("ok", 0) - status_counts.get("partial_read", 0))} failed</span>
</div></header><main><div class="filters"><strong>Show:</strong>
<label><input id="show-ok" type="checkbox" checked> OK</label><label><input id="show-partial" type="checkbox" checked> Partial</label><label><input id="show-failed" type="checkbox" checked> Failed</label>
<button id="export">Export reviewed JSON</button></div>{"".join(cards)}</main>
<script>
const original = {original_json};
function filterCards() {{
  const enabled = {{ok: document.querySelector('#show-ok').checked, partial: document.querySelector('#show-partial').checked, failed: document.querySelector('#show-failed').checked}};
  document.querySelectorAll('.card').forEach(card => card.classList.toggle('hidden', !enabled[card.dataset.status]));
}}
document.querySelectorAll('.filters input[type=checkbox]').forEach(input => input.addEventListener('change', filterCards));
document.querySelector('#export').addEventListener('click', () => {{
  const reviewed = original.map((row, index) => {{
    const card = document.querySelector(`.card[data-index="${{index}}"]`);
    const pm25Text = card.querySelector('[data-field=pm25]').value.trim();
    const co2Text = card.querySelector('[data-field=co2]').value.trim();
    const unreadable = card.querySelector('[data-field=unreadable]').checked;
    const pm25 = unreadable ? null : (pm25Text === '' ? row.pm25 : Number(pm25Text));
    const co2 = unreadable ? null : (co2Text === '' ? row.co2 : Number(co2Text));
    if ((pm25 !== null && (!Number.isFinite(pm25) || pm25 < 0)) || (co2 !== null && (!Number.isFinite(co2) || co2 < 0))) throw new Error('Corrections must be non-negative numbers');
    const finalStatus = unreadable ? 'display_unreadable' : (pm25 !== null && co2 !== null ? 'ok' : (pm25 !== null || co2 !== null ? 'partial_read' : row.status));
    return {{...row, reviewed_pm25: pm25, reviewed_co2: co2, reviewed_status: finalStatus, was_corrected: unreadable || pm25Text !== '' || co2Text !== ''}};
  }});
  const blob = new Blob([JSON.stringify({{exported_at: new Date().toISOString(), readings: reviewed}}, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = `streetaqi_review_${{new Date().toISOString().slice(0,10)}}.json`; link.click(); URL.revokeObjectURL(url);
}});
</script></body></html>"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def process(
    readings_path: Path,
    output_path: Path | None = None,
    image_root: Path | None = None,
) -> Path:
    """Generate the OCR quality-control viewer."""
    return generate_html(readings_path, output_path, image_root)
