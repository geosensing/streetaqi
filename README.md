# Street Air Quality

Street-level air pollution measurements from Delhi and other cities.

## Data

### Delhi (May 2026)

| Metric | Count |
|--------|-------|
| PM2.5 readings | 98 |
| CO readings | 98 |
| Unique locations | ~98 |

### Data Schema

Each reading in `data/rider/{city}/readings.json`:

```json
{
  "id": "47a27458-...",
  "timestamp": "2026-05-16T16:46:33.097000+05:30",
  "timestamp_utc": "2026-05-16T11:16:33.097+00:00",
  "gps": {
    "latitude": 28.643719,
    "longitude": 77.2989862
  },
  "reading": {
    "pm25": 99,
    "co": 414
  },
  "image": {
    "local_path": "images/pollution/day-09/001_itinerary-1-part-2.jpg",
    "original_name": "17789301803242784560681094233323.jpg",
    "remote_url": "https://..."
  },
  "metadata": {
    "day": 9,
    "itinerary": "1-2",
    "title": "Itinerary 1 - Part 2",
    "stop_id": "1.2",
    "is_traffic_stop": false,
    "is_traffic_jam": false,
    "note_raw": "1.2",
    "address": "Karkari Mor Flyover, East Delhi, Delhi, India",
    "road_type": "primary",
    "road_name": "Karkari Mor Flyover"
  }
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `reading.pm25` | PM2.5 concentration (μg/m³) |
| `reading.co` | Carbon monoxide (ppm) |
| `metadata.road_type` | OSM highway classification |
| `metadata.is_traffic_stop` | Reading taken at traffic signal |
| `metadata.is_traffic_jam` | Reading taken in traffic jam |
| `metadata.note_raw` | Raw rider note (preserved for reference) |

### Road Type Classifications

From OpenStreetMap highway tags:

| Type | Description |
|------|-------------|
| `motorway` | Expressway/freeway |
| `trunk` | Major arterial road |
| `primary` | Main road (like national highway) |
| `secondary` | State highway level |
| `tertiary` | Local connecting road |
| `residential` | Neighborhood street |

### Traffic Classification Logic

The rider enters notes like "1.2" or "2.1 traffic stop":

- `is_traffic_stop = True` if note contains "traffic stop" (case-insensitive)
- `is_traffic_jam = True` if note contains "traffic jam" (case-insensitive)
- `stop_id` is extracted as first token (e.g., "1.2" from "1.2 traffic stop")

## Data Collection

Data collected using the [rider route tool](https://github.com/soodoku/missing-women-rider-route-tool) and processed via [soundscape](https://github.com/soodoku/soundscape).

```bash
# Convert rider export to streetaqi format
cd ../soundscape
uv run soundscape convert-rider-pollution \
    --manifest data/rider/export-2026-05-24/manifest.json \
    --output ../streetaqi/data/rider \
    --city delhi \
    --geocode
```

## License

MIT
