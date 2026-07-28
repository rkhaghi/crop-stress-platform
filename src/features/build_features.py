#%%
from typing import Optional


def build_feature_vector(
    satellite_features: dict,
    weather_features: dict,
    soil_features: dict,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Merge satellite, weather, and soil feature dicts into a single flat
    feature vector ready for model training or inference.

    Parameters
    ----------
    satellite_features : output of features.satellite_features.extract_satellite_features
                         Pass empty dict {} when no valid Sentinel-2 scene is available.
    weather_features   : output of features.weather_features.build_weather_features
    soil_features      : output of ingestion.soilgrids.fetch_soil
    metadata           : optional extra fields, e.g. {"doy": 120, "crop_type": "wheat"}

    Returns
    -------
    Flat dict — all keys are strings, all values are numeric.
    """
    vector: dict = {}
    vector.update(satellite_features)                                     # sat_ keys already prefixed
    vector.update(weather_features)                                       # wx_  keys already prefixed
    vector.update({f"soil_{k}": v for k, v in soil_features.items()})    # add soil_ prefix

    # Explicit flag so the model knows whether satellite data was available
    vector["sat_data_available"] = 1 if satellite_features else 0

    if metadata:
        vector.update(metadata)

    return vector
