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
import tempfile
import traceback

import numpy as np
from rasterio.warp import Resampling

from extract import extract_all, point_geometry
from ingestion.cdse_s3 import download_asset
from processing.raster_clip import (
    align_array_to_reference,
    grids_match,
    load_band_with_metadata,
)
from features.satellite_features import extract_satellite_features
from features.weather_features import build_weather_features
from features.build_features import build_feature_vector
from inference.predictor import predict


def _load_bands(assets: dict, geometry: dict) -> tuple[dict, np.ndarray] | tuple[None, None]:
    """
    Download via CDSE S3, clip, scale, and align all bands for one scene.

    Returns (bands_dict, scl_array) or (None, None) if any band is missing.
    """
    required = ["red", "nir", "red_edge", "swir1", "swir2", "scene_classification"]
    if not all(k in assets for k in required):
        return None, None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name in required:
                dest = os.path.join(tmp, f"{name}.jp2")
                download_asset(assets[name], dest)
                paths[name] = dest

            red,      ref_meta  = load_band_with_metadata(paths["red"],      geometry)
            nir,      nir_meta  = load_band_with_metadata(paths["nir"],      geometry)
            red_edge, re_meta   = load_band_with_metadata(paths["red_edge"], geometry)
            swir1,    sw1_meta  = load_band_with_metadata(paths["swir1"],    geometry)
            swir2,    sw2_meta  = load_band_with_metadata(paths["swir2"],    geometry)
            scl_raw,  scl_meta  = load_band_with_metadata(
                paths["scene_classification"], geometry, scale=1.0, zero_is_nodata=False
            )

            def _align(arr, meta):
                if grids_match(meta, ref_meta):
                    return arr
                return align_array_to_reference(arr, meta, ref_meta, Resampling.bilinear)

            def _align_nearest(arr, meta):
                if grids_match(meta, ref_meta):
                    return arr
                return align_array_to_reference(arr, meta, ref_meta, Resampling.nearest)

            bands = {
                "red":      red,
                "nir":      _align(nir,      nir_meta),
                "red_edge": _align(red_edge, re_meta),
                "swir1":    _align(swir1,    sw1_meta),
                "swir2":    _align(swir2,    sw2_meta),
            }
            scl = _align_nearest(scl_raw, scl_meta).astype(np.uint8)

        return bands, scl
    except Exception as exc:
        print(f"[handler] Band load failed: {exc}")
        return None, None


def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda handler.

    Returns an API-Gateway-compatible response dict.
    """
    try:
        lat        = float(event["lat"])
        lon        = float(event["lon"])
        start_date = event["start_date"]
        end_date   = event["end_date"]

        # --- Ingest raw data ------------------------------------------------
        raw = extract_all(lat=lat, lon=lon, start_date=start_date, end_date=end_date)

        # --- Build features -------------------------------------------------
        scene_date = raw["sentinel"][0]["date"][:10] if raw["sentinel"] else end_date
        weather_feats = build_weather_features(raw["weather"], ref_date=scene_date)
        soil_feats    = raw["soil"]

        # Use the most recent valid Sentinel-2 scene
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

        return {
            "statusCode": 200,
            "body": json.dumps({
                "lat": lat,
                "lon": lon,
                "start_date": start_date,
                "end_date": end_date,
                **result,
            }),
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
