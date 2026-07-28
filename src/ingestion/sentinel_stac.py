# %%
import time

import pystac_client
from pystac_client.exceptions import APIError


STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
COLLECTION = "sentinel-2-l2a"

BAND_CANDIDATES = {
    "red": [
        "B04_10m",
        "B04",
        "red",
    ],
    "nir": [
        "B08_10m",
        "B08",
        "nir",
    ],
    "red_edge": [
        "B05_20m",
        "B05",
        "rededge1",
    ],
    "swir1": [
        "B11_20m",
        "B11",
        "swir16",
    ],
    "swir2": [
        "B12_20m",
        "B12",
        "swir22",
    ],
    "scene_classification": [
        "SCL_20m",
        "SCL",
        "scl",
    ],
}




def search_items(
    geometry: dict,
    start_date: str,
    end_date: str,
    max_cloud: int = 30,
    retries: int = 3,
) -> list:
    """
    Search for Sentinel-2 Level-2A scenes covering a polygon.

    Parameters
    ----------
    geometry:
        GeoJSON Polygon or MultiPolygon in EPSG:4326.
    start_date:
        Start date in YYYY-MM-DD format.
    end_date:
        End date in YYYY-MM-DD format.
    max_cloud:
        Maximum scene-level cloud-cover percentage.
    retries:
        Number of attempts after temporary API failures.
    """
    if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise ValueError(
            "geometry must be a GeoJSON Polygon or MultiPolygon."
        )

    if not 0 <= max_cloud <= 100:
        raise ValueError("max_cloud must be between 0 and 100.")

    datetime_range = (
        f"{start_date}T00:00:00Z/"
        f"{end_date}T23:59:59Z"
    )

    for attempt in range(1, retries + 1):
        try:
            # Reopen the client on every attempt so a failed or closed
            # HTTP connection is not reused.
            catalog = pystac_client.Client.open(
                STAC_URL,
                timeout=60,
            )

            results = catalog.search(
                collections=[COLLECTION],

                # Search using the actual polygon.
                intersects=geometry,

                datetime=datetime_range,

                # Filter cloudy scenes on the server.
                query={
                    "eo:cloud_cover": {
                        "lte": max_cloud,
                    }
                },

                # limit controls each page size.
                limit=20,

                # max_items controls the total returned.
                max_items=20,

                method="POST",
            )

            items = list(results.items())

            items.sort(
                key=lambda item: (
                    item.datetime
                    or item.properties.get("datetime", "")
                ),
                reverse=True,
            )

            return items

        except (APIError, ConnectionError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    "Sentinel-2 search failed after "
                    f"{retries} attempts."
                ) from exc

            wait = 2 ** attempt

            print(
                f"[sentinel_stac] Attempt {attempt} failed "
                f"({exc}). Retrying in {wait}s..."
            )

            time.sleep(wait)

    return []




def get_asset_urls(item) -> dict:
    """
    Return URLs for the required Sentinel-2 bands.

    Supports resolution-specific Copernicus keys and alternative
    STAC naming conventions.
    """
    selected_assets = {}

    for alias, candidate_keys in BAND_CANDIDATES.items():
        for asset_key in candidate_keys:
            asset = item.assets.get(asset_key)

            if asset is not None:
                selected_assets[alias] = asset.href
                break

    return selected_assets


