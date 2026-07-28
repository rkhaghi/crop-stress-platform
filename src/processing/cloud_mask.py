#%%
import numpy as np

# Sentinel-2 SCL classes considered invalid
# 0=No data, 1=Saturated, 3=Cloud shadow, 8=Cloud med, 9=Cloud high, 10=Cirrus, 11=Snow
MASK_CLASSES = {0, 1, 3, 8, 9, 10, 11}


def build_cloud_mask(scl_array: np.ndarray) -> np.ndarray:
    """
    Build a valid-pixel mask from a Sentinel-2 SCL band.

    Parameters
    ----------
    scl_array : np.ndarray
        2-D array of SCL integer class values.

    Returns
    -------
    np.ndarray of bool — True where the pixel is valid (not cloud/shadow/snow).
    """
    return ~np.isin(scl_array, list(MASK_CLASSES))


def apply_mask(band_array: np.ndarray, mask: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """
    Apply a boolean valid-pixel mask to a band array.

    Parameters
    ----------
    band_array : np.ndarray  — float reflectance band (0–1 or 0–10000).
    mask       : np.ndarray  — True where valid (output of build_cloud_mask).
    fill_value : float       — Value to assign to masked pixels (default NaN).

    Returns
    -------
    np.ndarray of float with invalid pixels replaced by fill_value.
    """
    result = band_array.astype(float)
    result[~mask] = fill_value
    return result


def valid_pixel_fraction(mask: np.ndarray) -> float:
    """Return the fraction (0–1) of valid pixels in a mask."""
    return float(mask.sum()) / mask.size
