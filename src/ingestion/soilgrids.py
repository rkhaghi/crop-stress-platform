import time

import requests

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

PROPERTIES = ["clay", "sand", "silt", "soc", "phh2o", "bdod", "cec"]
DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]

_MAX_RETRIES = 3


def fetch_soil(lat: float, lon: float) -> dict:
    params = {
        "lon": lon,
        "lat": lat,
        "property": PROPERTIES,
        "depth": DEPTHS,
        "value": ["mean"],
    }
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.get(SOILGRIDS_URL, params=params, timeout=90)
            response.raise_for_status()
            data = response.json()

            result = {}
            for layer in data["properties"]["layers"]:
                prop = layer["name"]
                for depth_info in layer["depths"]:
                    label = depth_info["label"]
                    result[f"{prop}_{label}"] = depth_info["values"]["mean"]
            return result
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"SoilGrids fetch failed after {_MAX_RETRIES} attempts."
                ) from exc
            wait = 2 ** attempt
            print(f"[soilgrids] Attempt {attempt} failed ({exc}). Retrying in {wait}s…")
            time.sleep(wait)
