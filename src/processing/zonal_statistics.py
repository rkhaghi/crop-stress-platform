#%%
import numpy as np

PERCENTILES = [10, 25, 50, 75, 90]


def zonal_stats(index_array: np.ndarray, name: str = "index") -> dict:
    """
    Compute summary statistics over a 2-D index array, ignoring NaN values.

    Parameters
    ----------
    index_array : np.ndarray — 2-D float array (e.g. NDVI raster tile).
    name        : str        — Prefix for output key names.

    Returns
    -------
    Flat dict: {name_mean, name_std, name_p10, name_p25, name_p50, name_p75, name_p90,
                name_valid_frac}
    """
    flat = index_array[~np.isnan(index_array)]

    if flat.size == 0:
        stats = {f"{name}_mean": np.nan, f"{name}_std": np.nan, f"{name}_valid_frac": 0.0}
        stats.update({f"{name}_p{p}": np.nan for p in PERCENTILES})
        return stats

    stats = {
        f"{name}_mean":        float(np.mean(flat)),
        f"{name}_std":         float(np.std(flat)),
        f"{name}_valid_frac":  float(flat.size / index_array.size),
    }
    for p, val in zip(PERCENTILES, np.percentile(flat, PERCENTILES)):
        stats[f"{name}_p{p}"] = float(val)

    return stats


def multi_index_stats(indices: dict) -> dict:
    """
    Run zonal_stats for every index in a dict and merge results.

    Parameters
    ----------
    indices : dict[str, np.ndarray] — e.g. output of spectral_indices.compute_all

    Returns
    -------
    Single flat dict of all statistics.
    """
    combined = {}
    for name, array in indices.items():
        combined.update(zonal_stats(array, name=name))
    return combined
