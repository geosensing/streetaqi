from __future__ import annotations

import pandas as pd
import pytest

from streetaqi.data import load_bundled_readings


@pytest.fixture
def readings() -> pd.DataFrame:
    return load_bundled_readings()


@pytest.fixture
def ocr_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "a" * 64,
                "image_path": "day-09/001_itinerary-1-photo.jpg",
                "day": 9,
                "itinerary_id": 1,
                "latitude": 28.6,
                "longitude": 77.2,
                "pm25": 85.0,
                "co2": 412.0,
                "status": "ok",
                "confidence": 0.95,
                "reference_pm25": 84.0,
                "reference_co2": 410.0,
                "provider": "google",
                "model": "gemini-3.5-flash",
                "batch_id": "run-1",
            },
            {
                "id": "b" * 64,
                "image_path": "day-09/002_itinerary-1-photo.png",
                "day": 9,
                "itinerary_id": 1,
                "latitude": None,
                "longitude": None,
                "pm25": 92.0,
                "co2": None,
                "status": "partial_read",
                "confidence": 0.75,
                "reference_pm25": None,
                "reference_co2": None,
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "batch_id": "batch-1",
            },
        ]
    )
