#%%
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask


def clip_to_geometry(src_path: str, geometry: dict) -> tuple[np.ndarray, dict]:
    """
    Clip a raster file to a GeoJSON geometry.

    Parameters
    ----------
    src_path : str   — Path (or VSICURL URL) to a raster file readable by rasterio.
    geometry : dict  — GeoJSON geometry dict (Polygon or MultiPolygon, EPSG:4326).

    Returns
    -------
    (out_image, out_meta) where:
        out_image : np.ndarray  shape (bands, rows, cols)
        out_meta  : dict        updated rasterio metadata for the clipped tile
    """
    with rasterio.open(src_path) as src:
        out_image, out_transform = rio_mask(src, [geometry], crop=True, nodata=np.nan)
        out_meta = src.meta.copy()

    out_meta.update(
        driver="GTiff",
        height=out_image.shape[1],
        width=out_image.shape[2],
        transform=out_transform,
    )
    return out_image, out_meta


def load_band_array(src_path: str, geometry: dict, scale: float = 1e-4) -> np.ndarray:
    """
    Load a single-band raster clipped to geometry and apply a scale factor.
    Sentinel-2 L2A reflectance is stored as uint16 (divide by 10000).

    Returns
    -------
    2-D float32 array.
    """
    image, _ = clip_to_geometry(src_path, geometry)
    return (image[0] * scale).astype(np.float32)
