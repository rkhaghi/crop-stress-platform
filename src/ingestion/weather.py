import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARIABLES = [
    "precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_max",
    "et0_fao_evapotranspiration",
    "windspeed_10m_mean",
    "soil_moisture_0_to_7cm_mean",
]


def fetch_weather(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "UTC",
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()["daily"]
