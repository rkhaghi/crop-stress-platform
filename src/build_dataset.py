#%%
"""
build_dataset.py
----------------
Batch feature extraction pipeline.

Reads a labelled locations CSV, runs the full extraction + feature pipeline
for each row, and saves the result as a parquet file ready for train.py.

Usage:
    python src/build_dataset.py \
        --locations data/locations.csv \
        --output    data/features.parquet

locations.csv required columns:
    lat, lon, start_date, end_date, stress_index

Optional columns (passed through as metadata):
    crop_type, field_id
"""

import argparse
import pathlib
import sys
import os
import traceback

import numpy as np
import pandas as pd

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract import extract_all, _point_geometry
from processing.raster_clip import load_band_array
from features.satellite_features import extract_satellite_features
from features.weather_features import build_weather_features
from features.build_features import build_feature_vector


# ---------------------------------------------------------------------------
# Satellite band loader
# ---------------------------------------------------------------------------

def _load_satellite_features(sentinel_scenes: list, geometry: dict) -> dict:
    """
    Attempt to extract satellite features from the most recent valid scene.

    Returns a flat feature dict, or {} if all scenes fail / are too cloudy.
    """
    required_bands = ["red", "nir", "red_edge", "swir1", "swir2", "scene_classification"]

    for scene in sentinel_scenes:
        assets = scene.get("assets", {})
        if not all(k in assets for k in required_bands):
            continue
        try:
            bands = {
                "red":      load_band_array(assets["red"],      geometry),
                "nir":      load_band_array(assets["nir"],      geometry),
                "red_edge": load_band_array(assets["red_edge"], geometry),
                "swir1":    load_band_array(assets["swir1"],    geometry),
                "swir2":    load_band_array(assets["swir2"],    geometry),
            }
            scl = load_band_array(
                assets["scene_classification"], geometry, scale=1.0
            ).astype(np.uint8)

            feats = extract_satellite_features(bands, scl)
            if feats is not None:
                return feats          # first valid scene wins
        except Exception as exc:
            print(f"    [skip scene {scene.get('id', '?')}] {exc}")
            continue

    return {}   # no valid scene found


# ---------------------------------------------------------------------------
# Per-row extraction
# ---------------------------------------------------------------------------

def _extract_row(row: pd.Series) -> dict | None:
    """
    Run the full extraction + feature pipeline for one labelled location.

    Returns a flat feature dict (with stress_index), or None on failure.
    """
    lat        = float(row["lat"])
    lon        = float(row["lon"])
    start_date = str(row["start_date"])
    end_date   = str(row["end_date"])
    label      = float(row["stress_index"])

    print(f"  → lat={lat}, lon={lon}  {start_date} / {end_date}")

    try:
        # Step 1 — ingest
        raw = extract_all(lat=lat, lon=lon, start_date=start_date, end_date=end_date)

        # Step 2 — satellite features
        geometry  = _point_geometry(lat, lon)
        sat_feats = _load_satellite_features(raw["sentinel"], geometry)

        # Step 3 — weather + soil features
        # Anchor weather rolling windows to the actual scene date if available,
        # so features describe conditions leading up to that specific image.
        scene_date = raw["sentinel"][0]["date"][:10] if raw["sentinel"] else end_date
        weather_feats = build_weather_features(raw["weather"], ref_date=scene_date)
        soil_feats    = raw["soil"]

        # Step 4 — optional metadata columns
        from datetime import date as date_cls
        doy = date_cls.fromisoformat(end_date).timetuple().tm_yday
        metadata = {"doy": doy}
        for col in ["crop_type", "field_id"]:
            if col in row.index:
                metadata[col] = row[col]

        # Step 5 — merge into flat vector
        feature_vec = build_feature_vector(sat_feats, weather_feats, soil_feats,
                                           metadata=metadata)

        # Add label and location info
        feature_vec["stress_index"] = label
        feature_vec["lat"]          = lat
        feature_vec["lon"]          = lon
        feature_vec["date"]         = end_date

        return feature_vec

    except Exception:
        print(f"    [FAILED] {traceback.format_exc()}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset(locations_path: str, output_path: str) -> None:
    """
    Loop over all rows in locations_path, extract features, save parquet.
    """
    locations = pd.read_csv(locations_path)
    print(f"[build_dataset] {len(locations)} locations to process.")

    rows = []
    failed = 0

    for idx, row in locations.iterrows():
        print(f"[{idx + 1}/{len(locations)}] Processing …")
        result = _extract_row(row)
        if result is not None:
            rows.append(result)
        else:
            failed += 1

    if not rows:
        print("[build_dataset] No rows extracted — check your locations file and API access.")
        return

    df = pd.DataFrame(rows)

    # Ensure date column is datetime for correct temporal sorting in train.py
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\n[build_dataset] Done.")
    print(f"  Rows saved : {len(df)}")
    print(f"  Failed     : {failed}")
    print(f"  Features   : {len(df.columns)} columns")
    print(f"  Output     : {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build features parquet from labelled locations.")
    parser.add_argument("--locations", required=True, help="Path to locations CSV")
    parser.add_argument("--output",    required=True, help="Path to save features.parquet")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_dataset(args.locations, args.output)
