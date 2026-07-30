# %%
"""
build_dataset.py
----------------

Batch feature-extraction pipeline.

The script:

1. Reads labelled locations from a CSV file.
2. Retrieves weather, soil and Sentinel-2 scene metadata.
3. Downloads the required Sentinel-2 assets from CDSE S3.
4. Clips the satellite bands to the location geometry.
5. Aligns all 20 m bands to the 10 m reference grid.
6. Calculates satellite, weather and soil features.
7. Merges the features into one flat training row.
8. Saves the final dataset as a Parquet file.

Usage
-----

Windows Command Prompt:

    python src/build_dataset.py ^
        --locations data/locations.csv ^
        --output data/features.parquet

Single line:

    python src/build_dataset.py --locations data/locations.csv --output data/features.parquet

Required CSV columns
--------------------

    lat
    lon
    start_date
    end_date
    stress_index

Optional CSV columns
--------------------

    crop_type
    field_id
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile
import traceback
from datetime import date as date_cls
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rasterio.warp import Resampling


# Ensure src/ is importable when running:
# python src/build_dataset.py ...
SRC_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

if SRC_DIRECTORY not in sys.path:
    sys.path.insert(0, SRC_DIRECTORY)


from extract import extract_all, point_geometry
from features.build_features import build_feature_vector
from features.satellite_features import extract_satellite_features
from features.weather_features import build_weather_features
from ingestion.cdse_s3 import download_asset
from processing.raster_clip import (
    align_array_to_reference,
    grids_match,
    load_band_with_metadata,
)


REQUIRED_INPUT_COLUMNS = {
    "lat",
    "lon",
    "start_date",
    "end_date",
    "stress_index",
}

OPTIONAL_METADATA_COLUMNS = [
    "crop_type",
    "field_id",
]

REQUIRED_BANDS = [
    "red",
    "nir",
    "red_edge",
    "swir1",
    "swir2",
    "scene_classification",
]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _parse_iso_date(value: Any, column_name: str) -> str:
    """
    Parse a value as a date and return YYYY-MM-DD.
    Accepts both YYYY-MM-DD and DD/MM/YYYY formats.
    """
    try:
        parsed = pd.to_datetime(value, dayfirst=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{column_name} could not be parsed as a date. "
            f"Received: {value!r}"
        ) from exc

    return parsed.strftime("%Y-%m-%d")


def _validate_coordinates(lat: float, lon: float) -> None:
    """
    Validate latitude and longitude values.
    """
    if not np.isfinite(lat):
        raise ValueError("Latitude must be a finite number.")

    if not np.isfinite(lon):
        raise ValueError("Longitude must be a finite number.")

    if not -90 <= lat <= 90:
        raise ValueError(
            f"Latitude must be between -90 and 90. Received: {lat}"
        )

    if not -180 <= lon <= 180:
        raise ValueError(
            f"Longitude must be between -180 and 180. Received: {lon}"
        )


def _validate_locations_dataframe(
    locations: pd.DataFrame,
) -> None:
    """
    Validate the locations input dataframe.
    """
    missing_columns = (
        REQUIRED_INPUT_COLUMNS - set(locations.columns)
    )

    if missing_columns:
        raise ValueError(
            "Locations CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if locations.empty:
        raise ValueError("Locations CSV does not contain any rows.")


# ---------------------------------------------------------------------------
# Satellite asset handling
# ---------------------------------------------------------------------------

def _download_scene_assets(
    scene: dict[str, Any],
    destination_directory: Path,
) -> dict[str, Path]:
    """
    Download all required assets for one Sentinel-2 scene.
    """
    assets = scene.get("assets", {})

    missing_assets = [
        band_name
        for band_name in REQUIRED_BANDS
        if band_name not in assets
    ]

    if missing_assets:
        raise ValueError(
            "Scene is missing required assets: "
            + ", ".join(missing_assets)
        )

    local_paths: dict[str, Path] = {}

    for band_name in REQUIRED_BANDS:
        destination = (
            destination_directory /
            f"{band_name}.jp2"
        )

        local_paths[band_name] = download_asset(
            s3_url=assets[band_name],
            output_path=destination,
        )

    return local_paths


def _load_and_align_scene_bands(
    local_paths: dict[str, Path],
    geometry: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load and align all bands to the red-band 10 m reference grid.

    Reflectance bands use bilinear resampling.
    SCL uses nearest-neighbour resampling.
    """
    red, red_meta = load_band_with_metadata(
        src_path=str(local_paths["red"]),
        geometry=geometry,
        scale=1e-4,
        zero_is_nodata=True,
    )

    nir, nir_meta = load_band_with_metadata(
        src_path=str(local_paths["nir"]),
        geometry=geometry,
        scale=1e-4,
        zero_is_nodata=True,
    )

    red_edge, red_edge_meta = load_band_with_metadata(
        src_path=str(local_paths["red_edge"]),
        geometry=geometry,
        scale=1e-4,
        zero_is_nodata=True,
    )

    swir1, swir1_meta = load_band_with_metadata(
        src_path=str(local_paths["swir1"]),
        geometry=geometry,
        scale=1e-4,
        zero_is_nodata=True,
    )

    swir2, swir2_meta = load_band_with_metadata(
        src_path=str(local_paths["swir2"]),
        geometry=geometry,
        scale=1e-4,
        zero_is_nodata=True,
    )

    scl, scl_meta = load_band_with_metadata(
        src_path=str(local_paths["scene_classification"]),
        geometry=geometry,
        scale=1.0,
        zero_is_nodata=False,
    )

    # B08 is also a 10 m band, but its clipped window can occasionally
    # differ by one row or column because of geometry/grid boundaries.
    if not grids_match(nir_meta, red_meta):
        nir = align_array_to_reference(
            source_array=nir,
            source_metadata=nir_meta,
            reference_metadata=red_meta,
            resampling=Resampling.bilinear,
        )

    red_edge = align_array_to_reference(
        source_array=red_edge,
        source_metadata=red_edge_meta,
        reference_metadata=red_meta,
        resampling=Resampling.bilinear,
    )

    swir1 = align_array_to_reference(
        source_array=swir1,
        source_metadata=swir1_meta,
        reference_metadata=red_meta,
        resampling=Resampling.bilinear,
    )

    swir2 = align_array_to_reference(
        source_array=swir2,
        source_metadata=swir2_meta,
        reference_metadata=red_meta,
        resampling=Resampling.bilinear,
    )

    scl = align_array_to_reference(
        source_array=scl,
        source_metadata=scl_meta,
        reference_metadata=red_meta,
        resampling=Resampling.nearest,
        source_nodata=0,
        destination_nodata=0,
    )

    scl = np.where(
        np.isfinite(scl),
        scl,
        0,
    ).astype(np.uint8)

    bands = {
        "red": red.astype(np.float32),
        "nir": nir.astype(np.float32),
        "red_edge": red_edge.astype(np.float32),
        "swir1": swir1.astype(np.float32),
        "swir2": swir2.astype(np.float32),
    }

    expected_shape = red.shape

    for band_name, band_array in bands.items():
        if band_array.shape != expected_shape:
            raise ValueError(
                f"{band_name} was not aligned correctly. "
                f"Expected {expected_shape}, received "
                f"{band_array.shape}."
            )

    if scl.shape != expected_shape:
        raise ValueError(
            "SCL was not aligned correctly. "
            f"Expected {expected_shape}, received {scl.shape}."
        )

    return bands, scl


def _load_satellite_features(
    sentinel_scenes: list[dict[str, Any]],
    geometry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """
    Extract features from the first usable Sentinel-2 scene.

    Scenes are expected to be ordered newest first.

    Returns
    -------
    tuple
        satellite_features:
            Flat dictionary of calculated satellite features.

        selected_scene:
            Scene metadata for the scene used, or None.
    """
    if not sentinel_scenes:
        print("    [satellite] No Sentinel-2 scenes found.")
        return {}, None

    for scene in sentinel_scenes:
        scene_id = scene.get("id", "unknown-scene")
        assets = scene.get("assets", {})

        missing_assets = [
            band_name
            for band_name in REQUIRED_BANDS
            if band_name not in assets
        ]

        if missing_assets:
            print(
                f"    [skip scene {scene_id}] "
                f"missing assets: {missing_assets}"
            )
            continue

        try:
            with tempfile.TemporaryDirectory(
                prefix="terrasignal_"
            ) as temporary_directory:
                temporary_path = Path(temporary_directory)

                local_paths = _download_scene_assets(
                    scene=scene,
                    destination_directory=temporary_path,
                )

                bands, scl = _load_and_align_scene_bands(
                    local_paths=local_paths,
                    geometry=geometry,
                )

                features = extract_satellite_features(
                    bands=bands,
                    scl=scl,
                )
            if features is not None and len(features) > 0:
                print(
                    f"    [satellite] Using scene {scene_id}; "
                    f"{len(features)} satellite features extracted."
                )
                return features, scene

            print(
                f"    [skip scene {scene_id}] "
                "feature extraction returned no valid features. "
                f"Valid red pixels={np.isfinite(bands['red']).sum()}, "
                f"valid NIR pixels={np.isfinite(bands['nir']).sum()}, "
                f"SCL classes={np.unique(scl).tolist()}"
            )


        except Exception as exc:
            print(
                f"    [skip scene {scene_id}] "
                f"{type(exc).__name__}: {exc}"
            )

    return {}, None


# ---------------------------------------------------------------------------
# Per-row extraction
# ---------------------------------------------------------------------------

def _extract_row(
    row: pd.Series,
) -> dict[str, Any] | None:
    """
    Run the full extraction and feature pipeline for one row.

    Returns
    -------
    dict | None
        Flat training row, or None when extraction fails.
    """
    try:
        lat = float(row["lat"])
        lon = float(row["lon"])
        label = float(row["stress_index"])

        _validate_coordinates(lat=lat, lon=lon)

        if not np.isfinite(label):
            raise ValueError(
                "stress_index must be a finite number."
            )

        start_date = _parse_iso_date(
            value=row["start_date"],
            column_name="start_date",
        )

        end_date = _parse_iso_date(
            value=row["end_date"],
            column_name="end_date",
        )

        if start_date > end_date:
            raise ValueError(
                f"start_date {start_date} is after "
                f"end_date {end_date}."
            )

        print(
            f"  -> lat={lat}, lon={lon}  "
            f"{start_date} / {end_date}"
        )

        # ---------------------------------------------------------------
        # Step 1: Ingest weather, soil and Sentinel scene metadata
        # ---------------------------------------------------------------
        raw = extract_all(
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
        )

        sentinel_scenes = raw.get("sentinel", [])
        weather_data = raw.get("weather")
        soil_data = raw.get("soil")

        if weather_data is None:
            raise ValueError("Weather extraction returned no data.")

        if soil_data is None:
            raise ValueError("Soil extraction returned no data.")

        # ---------------------------------------------------------------
        # Step 2: Satellite feature extraction
        # ---------------------------------------------------------------
        geometry = point_geometry(lat, lon)

        satellite_features, selected_scene = (
            _load_satellite_features(
                sentinel_scenes=sentinel_scenes,
                geometry=geometry,
            )
        )

        # This first model is intended to use satellite information,
        # so rows with no valid image are excluded.
        if not satellite_features or selected_scene is None:
            print(
                "    [FAILED] No usable Sentinel-2 scene "
                "was available."
            )
            return None

        scene_date_value = selected_scene.get("date")

        if not scene_date_value:
            raise ValueError(
                "Selected Sentinel scene does not contain a date."
            )

        scene_date = str(scene_date_value)[:10]

        # ---------------------------------------------------------------
        # Step 3: Weather and soil features
        # ---------------------------------------------------------------
        weather_features = build_weather_features(
            weather_data,
            ref_date=scene_date,
        )

        soil_features = soil_data

        # ---------------------------------------------------------------
        # Step 4: Metadata
        # ---------------------------------------------------------------
        day_of_year = (
            date_cls
            .fromisoformat(scene_date)
            .timetuple()
            .tm_yday
        )

        metadata: dict[str, Any] = {
            "doy": day_of_year,
            "satellite_available": 1,
            "sentinel_scene_id": selected_scene.get("id"),
            "scene_cloud_cover": selected_scene.get(
                "cloud_cover"
            ),
        }

        for column_name in OPTIONAL_METADATA_COLUMNS:
            if (
                column_name in row.index
                and pd.notna(row[column_name])
            ):
                metadata[column_name] = row[column_name]

        # ---------------------------------------------------------------
        # Step 5: Build final feature vector
        # ---------------------------------------------------------------
        feature_vector = build_feature_vector(
            satellite_features,
            weather_features,
            soil_features,
            metadata=metadata,
        )

        feature_vector["stress_index"] = label
        feature_vector["lat"] = lat
        feature_vector["lon"] = lon
        feature_vector["date"] = scene_date
        feature_vector["requested_start_date"] = start_date
        feature_vector["requested_end_date"] = end_date

        return feature_vector

    except Exception:
        print(
            "    [FAILED]\n"
            + traceback.format_exc()
        )

        return None


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(
    locations_path: str,
    output_path: str,
) -> None:
    """
    Extract features for every CSV row and save a Parquet dataset.
    """
    locations = pd.read_csv(locations_path)

    _validate_locations_dataframe(locations)

    total_rows = len(locations)

    print(
        f"[build_dataset] {total_rows} locations to process."
    )

    extracted_rows: list[dict[str, Any]] = []
    failed_count = 0

    for row_number, (_, row) in enumerate(
        locations.iterrows(),
        start=1,
    ):
        print(
            f"[{row_number}/{total_rows}] Processing ..."
        )

        result = _extract_row(row)

        if result is None:
            failed_count += 1
        else:
            extracted_rows.append(result)

    if not extracted_rows:
        raise RuntimeError(
            "No rows were extracted. Check API access, "
            "CDSE credentials, input dates and location geometry."
        )

    dataset = pd.DataFrame(extracted_rows)

    dataset["date"] = pd.to_datetime(
        dataset["date"],
        errors="raise",
    )

    dataset["requested_start_date"] = pd.to_datetime(
        dataset["requested_start_date"],
        errors="raise",
    )

    dataset["requested_end_date"] = pd.to_datetime(
        dataset["requested_end_date"],
        errors="raise",
    )

    dataset = (
        dataset
        .sort_values(["date", "lat", "lon"])
        .reset_index(drop=True)
    )

    output = pathlib.Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_parquet(
        output,
        index=False,
    )

    print()
    print("[build_dataset] Done.")
    print(f"  Input rows : {total_rows}")
    print(f"  Rows saved : {len(dataset)}")
    print(f"  Failed     : {failed_count}")
    print(f"  Columns    : {len(dataset.columns)}")
    print(f"  Output     : {output.resolve()}")

    if failed_count:
        failure_rate = failed_count / total_rows

        print(
            f"  Failure rate: {failure_rate:.1%}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build a crop-stress feature dataset from "
            "labelled locations."
        )
    )

    parser.add_argument(
        "--locations",
        required=True,
        help="Path to the labelled locations CSV.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output Parquet file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()

    build_dataset(
        locations_path=arguments.locations,
        output_path=arguments.output,
    )