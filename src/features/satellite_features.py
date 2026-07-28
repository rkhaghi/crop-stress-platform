#%%
import numpy as np

from processing.cloud_mask import build_cloud_mask, apply_mask, valid_pixel_fraction
from processing.spectral_indices import compute_all
from processing.zonal_statistics import multi_index_stats

MIN_VALID_FRACTION = 0.3  # discard scenes where >70 % pixels are cloudy


def extract_satellite_features(bands: dict, scl: np.ndarray) -> dict | None:
    """
    Compute zonal statistics of spectral indices for one Sentinel-2 scene.

    Parameters
    ----------
    bands : dict with keys nir, red, red_edge, swir1, swir2
            (2-D float32 arrays, reflectance 0–1, clipped to AOI)
    scl   : np.ndarray — 2-D SCL band (uint8)

    Returns
    -------
    Flat dict of index statistics, or None if the scene is too cloudy.
    """
    mask = build_cloud_mask(scl)
    if valid_pixel_fraction(mask) < MIN_VALID_FRACTION:
        return None  # scene too cloudy — skip

    masked_bands = {k: apply_mask(v, mask) for k, v in bands.items()}
    indices = compute_all(masked_bands)
    return multi_index_stats(indices)
