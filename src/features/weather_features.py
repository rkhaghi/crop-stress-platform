#%%
from typing import Optional

import pandas as pd

ROLL_WINDOWS = [7, 14, 30]
WEATHER_COLS = [
    "precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_max",
    "et0_fao_evapotranspiration",
    "windspeed_10m_mean",
    "soil_moisture_0_to_7cm_mean",
]


def build_weather_features(daily: dict, ref_date: Optional[str] = None) -> dict:
    """
    Engineer time-series features from a weather dict (output of ingestion.weather.fetch_weather).

    Parameters
    ----------
    daily    : dict — keys are column names including "time".
    ref_date : str  — ISO date. Features are computed relative to this date.
                      Defaults to the last date in the series.

    Returns
    -------
    Flat feature dict with rolling means, sums, and derived stress proxies.
    """
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    if ref_date is not None:
        df = df[df["time"] <= pd.Timestamp(ref_date)]

    features: dict = {}

    # Rolling statistics
    for col in WEATHER_COLS:
        if col not in df.columns:
            continue
        series = df[col]
        for w in ROLL_WINDOWS:
            window = series.tail(w)
            features[f"wx_{col}_mean_{w}d"] = float(window.mean())
            features[f"wx_{col}_sum_{w}d"]  = float(window.sum())

    # Cumulative water deficit (ET₀ − precipitation) — drought proxy
    if "et0_fao_evapotranspiration" in df.columns and "precipitation_sum" in df.columns:
        df["water_deficit"] = df["et0_fao_evapotranspiration"] - df["precipitation_sum"]
        for w in [14, 30]:
            features[f"wx_water_deficit_sum_{w}d"] = float(df.tail(w)["water_deficit"].sum())

    # Heat stress: days above 30 °C in the last 14 days
    if "temperature_2m_max" in df.columns:
        features["wx_heat_stress_days_14d"] = int((df.tail(14)["temperature_2m_max"] > 30).sum())
        features["wx_temp_max_14d"]         = float(df.tail(14)["temperature_2m_max"].max())

    return features
