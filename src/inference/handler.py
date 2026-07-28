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

from extract import extract_all
from features.satellite_features import extract_satellite_features
from features.weather_features import build_weather_features
from features.build_features import build_feature_vector
from inference.predictor import predict


def handler(event: dict, context) -> dict:
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
        weather_feats = build_weather_features(raw["weather"], ref_date=end_date)
        soil_feats    = raw["soil"]

        # Use the most recent non-null Sentinel scene
        sat_feats = {}
        for scene in raw["sentinel"]:
            # TODO: download bands, clip, and call extract_satellite_features
            # Placeholder — real implementation downloads assets via raster_clip
            pass

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
