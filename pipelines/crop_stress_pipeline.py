#%%
"""
crop_stress_pipeline.py
-----------------------
End-to-end orchestration pipeline: ingest → process → featurise → predict.

Can be run locally or triggered by AWS Step Functions.

Usage:
    python pipelines/crop_stress_pipeline.py  --lat 51.5 --lon -1.2 --start 2024-04-01 --end 2024-06-30 --model models/crop_stress_model.json
"""
import argparse
import json
import sys
import os

# Allow running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from extract import extract_all
from features.weather_features import build_weather_features
from features.build_features import build_feature_vector
from inference.predictor import predict


def run_pipeline(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    model_path: str,
    max_cloud: int = 80,
) -> dict:
    """
    Full crop-stress prediction pipeline for a single location and date window.

    Steps
    -----
    1. Ingest  — weather, soil, Sentinel-2 scene list
    2. Feature — engineer weather + soil feature vectors
    3. Predict — run model, return stress score and level

    Note: Sentinel-2 band download + spectral features are wired in handler.py
    for the Lambda path. In the batch pipeline, add raster_clip + extract_satellite_features here.

    Returns
    -------
    dict with lat, lon, period, stress_index, stress_level
    """
    print(f"[pipeline] Starting for lat={lat}, lon={lon}, {start_date} → {end_date}")

    # Step 1 — Ingest
    raw = extract_all(lat=lat, lon=lon, start_date=start_date, end_date=end_date,
                      max_cloud=max_cloud)

    # Step 2 — Features
    scene_date = raw["sentinel"][0]["date"][:10] if raw["sentinel"] else end_date
    weather_feats = build_weather_features(raw["weather"], ref_date=scene_date)
    soil_feats    = raw["soil"]

    # Satellite features — use the most recent valid scene
    import tempfile
    import numpy as np
    from extract import point_geometry
    from ingestion.cdse_s3 import download_asset
    from processing.raster_clip import load_band_with_metadata, align_array_to_reference, grids_match
    from features.satellite_features import extract_satellite_features
    from rasterio.warp import Resampling

    geometry  = point_geometry(lat, lon)
    sat_feats = {}
    required  = ["red", "nir", "red_edge", "swir1", "swir2", "scene_classification"]

    for scene in raw["sentinel"]:
        assets = scene.get("assets", {})
        if not all(k in assets for k in required):
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                paths = {}
                for name in required:
                    dest = os.path.join(tmp, f"{name}.jp2")
                    download_asset(assets[name], dest)
                    paths[name] = dest

                red,      ref_meta = load_band_with_metadata(paths["red"],      geometry)
                nir,      nir_meta = load_band_with_metadata(paths["nir"],      geometry)
                red_edge, re_meta  = load_band_with_metadata(paths["red_edge"], geometry)
                swir1,    sw1_meta = load_band_with_metadata(paths["swir1"],    geometry)
                swir2,    sw2_meta = load_band_with_metadata(paths["swir2"],    geometry)
                scl_raw,  scl_meta = load_band_with_metadata(
                    paths["scene_classification"], geometry, scale=1.0, zero_is_nodata=False
                )

                def _align(arr, meta):
                    return arr if grids_match(meta, ref_meta) else align_array_to_reference(arr, meta, ref_meta, Resampling.bilinear)

                def _align_nn(arr, meta):
                    return arr if grids_match(meta, ref_meta) else align_array_to_reference(arr, meta, ref_meta, Resampling.nearest)

                bands = {
                    "red":      red,
                    "nir":      _align(nir,      nir_meta),
                    "red_edge": _align(red_edge, re_meta),
                    "swir1":    _align(swir1,    sw1_meta),
                    "swir2":    _align(swir2,    sw2_meta),
                }
                scl = _align_nn(scl_raw, scl_meta).astype(np.uint8)

            feats = extract_satellite_features(bands, scl)
            if feats is not None:
                sat_feats = feats
                print(f"[pipeline] Satellite features from scene {scene['id']}")
                break
        except Exception as exc:
            print(f"[pipeline] Scene {scene['id']} failed: {exc}")
            continue

    from datetime import date
    doy = date.fromisoformat(end_date).timetuple().tm_yday
    feature_vec = build_feature_vector(sat_feats, weather_feats, soil_feats,
                                       metadata={"doy": doy})

    # Step 3 — Predict
    result = predict(feature_vec, model_path=model_path)

    output = {
        "lat":          lat,
        "lon":          lon,
        "start_date":   start_date,
        "end_date":     end_date,
        "stress_index": result["stress_index"],
        "stress_level": result["stress_level"],
        "sentinel_scenes_found": len(raw["sentinel"]),
    }
    print(f"[pipeline] Result: {output}")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop stress prediction pipeline")
    parser.add_argument("--lat",       type=float, required=True)
    parser.add_argument("--lon",       type=float, required=True)
    parser.add_argument("--start",     type=str,   required=True)
    parser.add_argument("--end",       type=str,   required=True)
    parser.add_argument("--model",     type=str,   required=True)
    parser.add_argument("--max-cloud", type=int,   default=80)
    parser.add_argument("--output",    type=str,   default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_pipeline(
        lat=args.lat,
        lon=args.lon,
        start_date=args.start,
        end_date=args.end,
        model_path=args.model,
        max_cloud=args.max_cloud,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[pipeline] Output saved → {args.output}")
    else:
        print(json.dumps(result, indent=2))
