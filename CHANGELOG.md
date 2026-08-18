# Changelog

## [Unreleased]

## 0.2.0

- Adopt the current py-canon CI, documentation, and tag-driven release workflow.
- Replace duplicate JSON tables and committed generated outputs with one
  explicit-schema Parquet dataset bundled in the wheel.
- Correct EPA breakpoint labels and apply the required one-decimal PM2.5
  truncation rule.
- Preserve implausible CO2 sensor values while excluding them from summaries
  through an explicit, persisted quality-control flag.
- Replace the retired Gemini 2.0 default with Gemini 3.5 Flash and require
  structured OCR output.
- Store OCR results and Claude batch mappings as Parquet.
- Enforce the OCR JSON schema through Claude structured outputs and reject
  unexpected DataFrame or CSV columns instead of silently discarding them.
- Reject nulls in required fields, non-finite concentrations, and inconsistent
  OCR status/value pairs at both OCR read and write boundaries; require
  reference-image matches on complete path components.
- Prevent the QC viewer from embedding files outside an explicit trusted image
  root, resolve recorded paths already beneath that root, and remove its ad hoc
  CSV export.
- Add regression coverage for data contracts, analysis, API adapters, CLI, and
  viewer safety.
