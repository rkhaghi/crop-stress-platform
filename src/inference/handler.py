#%%
"""
handler.py
----------
AWS Lambda entry point for the crop-stress inference service.

Expected event payload:
{
    "lat":        51.5,
    "lon":        -1.2,
    "start_date": "2024-04-01",
    "end_date":   "2024-06-30"
}

Environment variables required:
    MODEL_PATH  — S3 path or local path to the joblib model bundle.
"""
import json
import os
import traceback
import tempfile

import numpy as np

from extract import extract_all
from ingestion.sentinel_stac import get_asset_urls
from processing.raster_clip import load_band_array
from processing.cloud_mask import build_cloud_mask
from features.satellite_features import extract_satellite_features
from features.weather_features import build_weather_features
from features.build_features import build_feature_vector
from inference.predictor import predict


def _load_bands(assets: dict, geometry: dict) -> tuple[dict, np.ndarray] | tuple[None, None]:
    """
    Download and clip all required bands for one Sentinel-2 scene.

    Returns (bands_dict, scl_array) or (None, None) if any band is missing.
    """
    required = ["red", "nir", "red_edge", "swir1", "swir2", "scene_classification"]
    if not all(k in assets for k in required):
        return None, None

    try:
        bands = {
            "red":      load_band_array(assets["red"],      geometry),
            "nir":      load_band_array(assets["nir"],      geometry),
            "red_edge": load_band_array(assets["red_edge"], geometry),
            "swir1":    load_band_array(assets["swir1"],    geometry),
            "swir2":    load_band_array(assets["swir2"],    geometry),
        }
        scl = load_band_array(assets["scene_classification"], geometry, scale=1.0).astype(np.uint8)
        return bands, scl
    except Exception as exc:
        print(f"[handler] Band download failed: {exc}")
        return None, None


def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda handler.

    Returns an API-Gateway-compatible response dict.
    """
    try:
        # API Gateway wraps the POST body in event["body"]; direct invocations pass fields top-level
        body = json.loads(event["body"]) if "body" in event else event
        lat        = float(body["lat"])
        lon        = float(body["lon"])
        start_date = body["start_date"]
        end_date   = body["end_date"]

        # --- Ingest raw data ------------------------------------------------
        raw = extract_all(lat=lat, lon=lon, start_date=start_date, end_date=end_date)

        # --- Build features -------------------------------------------------
        scene_date = raw["sentinel"][0]["date"][:10] if raw["sentinel"] else end_date
        weather_feats = build_weather_features(raw["weather"], ref_date=scene_date)
        soil_feats    = raw["soil"]

        # Use the most recent valid Sentinel-2 scene
        from extract import point_geometry
        geometry  = point_geometry(lat, lon)
        sat_feats = {}

        for scene in raw["sentinel"]:
            assets = scene.get("assets", {})
            bands, scl = _load_bands(assets, geometry)
            if bands is None:
                continue
            feats = extract_satellite_features(bands, scl)
            if feats is not None:
                sat_feats = feats
                print(f"[handler] Satellite features from scene {scene['id']}")
                break  # use the first (most recent) valid scene

        feature_vec = build_feature_vector(sat_feats, weather_feats, soil_feats,
                                           metadata={"doy": _doy(end_date)})

        # --- Predict --------------------------------------------------------
        result = predict(feature_vec)

        # Strip large asset URL dicts — dashboard only needs scene metadata
        sentinel_meta = [
            {"id": s["id"], "date": s["date"], "cloud_cover": s["cloud_cover"]}
            for s in raw["sentinel"]
        ]

        return {
            "statusCode": 200,
            "body": json.dumps({
                "lat": lat,
                "lon": lon,
                "start_date": start_date,
                "end_date": end_date,
                **result,
                "weather": raw["weather"],
                "soil": raw["soil"],
                "sentinel": sentinel_meta,
            }, default=str),
        }

    except KeyError as exc:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Missing required field: {exc}"}),
        }
    except Exception:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": traceback.format_exc()}),
        }


def _doy(date_str: str) -> int:
    """Return day-of-year (1–366) for an ISO date string."""
    from datetime import date
    d = date.fromisoformat(date_str)
    return d.timetuple().tm_yday
