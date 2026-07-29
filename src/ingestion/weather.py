import requests
import time

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARIABLES = [
    "precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_max",
    "et0_fao_evapotranspiration",
    "windspeed_10m_mean",
    "soil_moisture_0_to_7cm_mean",
]

_MAX_RETRIES = 3


def fetch_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    retries: int = _MAX_RETRIES,
) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "UTC",
    }
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()["daily"]
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Weather fetch failed after {retries} attempts."
                ) from exc
            wait = 2 ** attempt
            print(f"[weather] Attempt {attempt} failed ({exc}). Retrying in {wait}s\u2026")
            time.sleep(wait)
