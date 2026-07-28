#%%
import numpy as np


def _normalised_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where denominator is zero."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where((a + b) != 0, (a - b) / (a + b), np.nan)
    return ratio.astype(np.float32)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalised Difference Vegetation Index — general greenness."""
    return _normalised_ratio(nir, red)


def ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    """Normalised Difference Red-Edge Index — chlorophyll / early stress."""
    return _normalised_ratio(nir, red_edge)


def ndwi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Normalised Difference Water Index — canopy water content."""
    return _normalised_ratio(nir, swir1)


def lswi(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Land Surface Water Index — soil + vegetation moisture."""
    return _normalised_ratio(nir, swir2)


def compute_all(bands: dict) -> dict:
    """
    Compute all spectral indices from a dict of 2-D band arrays.

    Expected keys: nir, red, red_edge, swir1, swir2  (float32, reflectance 0–1).

    Returns
    -------
    dict with keys: ndvi, ndre, ndwi, lswi
    """
    return {
        "ndvi": ndvi(bands["nir"], bands["red"]),
        "ndre": ndre(bands["nir"], bands["red_edge"]),
        "ndwi": ndwi(bands["nir"], bands["swir1"]),
        "lswi": lswi(bands["nir"], bands["swir2"]),
    }
