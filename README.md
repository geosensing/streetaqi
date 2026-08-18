# streetaqi

[![PyPI](https://img.shields.io/pypi/v/streetaqi)](https://pypi.org/project/streetaqi/)
[![Python](https://img.shields.io/pypi/pyversions/streetaqi)](https://pypi.org/project/streetaqi/)

`streetaqi` validates, summarizes, maps, and quality-checks street-level PM2.5
and CO2 sensor readings. Persistent tables use explicit-schema Parquet. CSV is
accepted only as an import boundary.

## Installation

```bash
pip install streetaqi
```

Install optional map or LLM integrations only when needed:

```bash
pip install "streetaqi[maps]"
pip install "streetaqi[llm]"
```

## Analyze the bundled example

```bash
streetaqi sample-data --output delhi_readings.parquet
streetaqi analyze --readings delhi_readings.parquet --output output/analysis
```

The analysis writes typed Parquet tables for the summary, breakpoint
comparisons, category distribution, and per-stop statistics, plus PDF figures.
When the `maps` extra is installed, it also writes an interactive HTML map.

The same workflow is available from Python:

```python
from streetaqi.analyze import compute_summary_stats
from streetaqi.data import load_bundled_readings

readings = load_bundled_readings()
summary = compute_summary_stats(readings)
```

## Measurement semantics

The package compares individual PM2.5 readings with the EPA's 2024 AQI
concentration breakpoints. It labels the result as a *share of readings*, not an
AQI or a regulatory exceedance: mobile instantaneous observations do not by
themselves estimate the EPA's 24-hour or annual regulatory quantities.

Raw CO2 values remain in the canonical data. A persisted quality-control flag
marks values below 350 ppm, the lower end of the EPA sensor guide's typical
outdoor range. CO2 summaries report their QC-passing denominator and exclude
flagged values; PM2.5 summaries are unaffected.

## Canonical readings

The schema is defined once as `streetaqi.data.READINGS_SCHEMA`. It includes the
reading ID, collection time, itinerary fields, PM2.5, CO2, the CO2 QC flag,
coordinates, note, and image references. Loaders reject duplicate IDs, invalid
coordinates, negative concentrations, schema drift, and inconsistent QC flags.

```python
from pathlib import Path

from streetaqi.data import load_readings, write_readings

readings = load_readings(Path("delhi_readings.parquet"))
write_readings(readings, Path("validated_readings.parquet"))
```

## OCR sensor images

Gemini is the default provider. Set `GEMINI_API_KEY` for Gemini or
`ANTHROPIC_API_KEY` for Claude; both official SDKs read these environment
variables.

```bash
streetaqi annotate \
  --images "exports/images/pollution/**/*.jpg" \
  --model gemini-3.5-flash \
  --reference delhi_readings.parquet \
  --output output/annotations
```

Use `--batch` for Gemini's asynchronous Batch API. Claude models use the
Message Batches API automatically. OCR results and Claude batch mappings are
stored as typed Parquet.

## Quality-control viewer

```bash
streetaqi viewer \
  --readings output/annotations/ocr_results_20260817T120000Z.parquet \
  --image-root exports/images \
  --output output/annotations/viewer.html
```

The explicit image root prevents an untrusted results file from embedding
arbitrary local files. The browser's reviewed JSON download is an interchange
boundary; analytical storage remains Parquet.

## Development

```bash
uv sync --all-groups --all-extras
make ci
make docs
```

## License

MIT
