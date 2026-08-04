#%%
"""
extract.py
----------
Unified data-extraction entry point for the crop-stress platform.

Usage (CLI):
    python src/extract.py --lat 51.5 --lon -1.2 --start 2024-04-01 --end 2024-06-30

Usage (Python):
    from src.extract import extract_all
    result = extract_all(lat=51.5, lon=-1.2, start_date="2024-04-01", end_date="2024-06-30")
"""

import argparse
import json
import os
import sys
from typing import Optional

# Ensure src/ is on the path when running interactively or as a script
_src_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.path.abspath("src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from ingestion.weather import fetch_weather
from ingestion.soilgrids import fetch_soil
from ingestion.sentinel_stac import search_items, get_asset_urls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def point_geometry(lat: float, lon: float, buffer_deg: float = 0.01) -> dict:
    """Return a GeoJSON Polygon bbox centred on (lat, lon)."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - buffer_deg, lat - buffer_deg],
            [lon + buffer_deg, lat - buffer_deg],
            [lon + buffer_deg, lat + buffer_deg],
            [lon - buffer_deg, lat + buffer_deg],
            [lon - buffer_deg, lat - buffer_deg],
        ]],
    }


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_all(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_cloud: int = 80,
    buffer_deg: float = 0.01,
) -> dict:
    """
    Extract weather, soil, and Sentinel-2 data for a given location and
    date range.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees (WGS-84).
    lon : float
        Longitude in decimal degrees (WGS-84).
    start_date : str
        ISO date string, e.g. "2024-04-01".
    end_date : str
        ISO date string, e.g. "2024-06-30".
    max_cloud : int
        Maximum cloud cover percentage for Sentinel-2 scenes (default 30).
    buffer_deg : float
        Bounding-box half-width in degrees around the point (default 0.01 ≈ 1 km).

    Returns
    -------
    dict with keys: "weather", "soil", "sentinel"
    """
    if not (-90 <= lat <= 90):
        raise ValueError(f"lat must be between -90 and 90, got {lat}.")
    if not (-180 <= lon <= 180):
        raise ValueError(f"lon must be between -180 and 180, got {lon}.")

    print(f"[extract] Location : lat={lat}, lon={lon}")
    print(f"[extract] Period   : {start_date} → {end_date}")

    # --- Weather -----------------------------------------------------------
    print("[extract] Fetching weather …")
    weather = fetch_weather(lat, lon, start_date, end_date)

    # --- Soil --------------------------------------------------------------
    print("[extract] Fetching soil properties …")
    soil = fetch_soil(lat, lon)

    # --- Sentinel-2 --------------------------------------------------------
    print("[extract] Searching Sentinel-2 scenes …")
    geometry = point_geometry(lat, lon, buffer_deg)
    items = search_items(geometry, start_date, end_date, max_cloud=max_cloud)
    sentinel = [
        {
            "id": item.id,
            "date": item.datetime.isoformat() if item.datetime else None,
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "assets": get_asset_urls(item),
        }
        for item in items
    ]
    print(f"[extract] Found {len(sentinel)} Sentinel-2 scene(s).")

    return {
        "weather": weather,
        "soil": soil,
        "sentinel": sentinel,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract crop-stress signals for a location and date range."
    )
    parser.add_argument("--lat",   type=float, required=True,  help="Latitude (decimal degrees)")
    parser.add_argument("--lon",   type=float, required=True,  help="Longitude (decimal degrees)")
    parser.add_argument("--start", type=str,   required=True,  help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   type=str,   required=True,  help="End date   YYYY-MM-DD")
    parser.add_argument("--max-cloud", type=int, default=30,   help="Max cloud cover %% (default 30)")
    parser.add_argument("--buffer",    type=float, default=0.01, help="Bbox buffer in degrees (default 0.01)")
    parser.add_argument("--output", type=str, default=None,    help="Optional path to save JSON output")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = extract_all(
        lat=args.lat,
        lon=args.lon,
        start_date=args.start,
        end_date=args.end,
        max_cloud=args.max_cloud,
        buffer_deg=args.buffer,
    )
       #breakpoint created by ipdb
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[extract] Results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2, default=str))


