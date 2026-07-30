# %%
"""
raster_clip.py
--------------

Raster loading, clipping and grid-alignment utilities.

Expected geometry input:
    GeoJSON Polygon or MultiPolygon in EPSG:4326.

Sentinel-2 bands:
    B04 and B08: 10 m
    B05, B11, B12 and SCL: 20 m

The functions below allow all bands to be clipped and aligned to the same
10 m reference grid before spectral indices are calculated.
"""

from __future__ import annotations

from typing import Any

import ipdb
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_geom
#%%

WGS84 = CRS.from_epsg(4326)


def _validate_geometry(geometry: dict[str, Any]) -> None:
    """
    Validate a GeoJSON geometry dictionary.
    """
    if not isinstance(geometry, dict):
        raise TypeError("geometry must be a GeoJSON dictionary.")

    geometry_type = geometry.get("type")

    if geometry_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(
            "geometry must be a GeoJSON Polygon or MultiPolygon."
        )

    if "coordinates" not in geometry:
        raise ValueError("geometry is missing coordinates.")


def _reproject_geometry(
    geometry: dict[str, Any],
    destination_crs: CRS,
) -> dict[str, Any]:
    """
    Reproject a GeoJSON geometry from EPSG:4326 to the raster CRS.
    """
    _validate_geometry(geometry)

    if destination_crs is None:
        raise ValueError("The raster does not contain CRS information.")

    if destination_crs == WGS84:
        return geometry

    return transform_geom(
        src_crs=WGS84,
        dst_crs=destination_crs,
        geom=geometry,
        precision=6,
    )


def clip_to_geometry(
    src_path: str,
    geometry: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Clip a raster to a GeoJSON geometry.

    Parameters
    ----------
    src_path:
        Path to a raster readable by Rasterio.

    geometry:
        GeoJSON Polygon or MultiPolygon in EPSG:4326.

    Returns
    -------
    tuple[np.ndarray, dict]
        out_image:
            Float32 array with shape (bands, rows, columns).
            Pixels outside the polygon are represented by NaN.

        out_meta:
            Raster metadata updated for the clipped raster.
    """
    _validate_geometry(geometry)

    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(
                f"Raster does not contain CRS information: {src_path}"
            )

        geometry_native = _reproject_geometry(
            geometry=geometry,
            destination_crs=src.crs,
        )

        masked_image, out_transform = rio_mask(
            dataset=src,
            shapes=[geometry_native],
            crop=True,
            filled=False,
        )

        out_image = (
            masked_image
            .astype(np.float32)
            .filled(np.nan)
        )

        out_meta = src.meta.copy()

    out_meta.update(
        {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": np.nan,
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )

    return out_image, out_meta


def load_band_with_metadata(
    src_path: str,
    geometry: dict[str, Any],
    scale: float = 1e-4,
    zero_is_nodata: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load one raster band, clip it and apply an optional scale factor.

    Parameters
    ----------
    src_path:
        Local raster path.

    geometry:
        GeoJSON Polygon or MultiPolygon in EPSG:4326.

    scale:
        Multiplicative scale factor.

        Sentinel-2 reflectance bands normally use 1e-4.
        SCL should use 1.0.

    zero_is_nodata:
        If True, zero values are replaced with NaN.

        This should normally be True for reflectance bands and False
        for the scene classification layer.

    Returns
    -------
    tuple[np.ndarray, dict]
        Two-dimensional float32 array and its clipped metadata.
    """
    image, metadata = clip_to_geometry(
        src_path=src_path,
        geometry=geometry,
    )

    if image.shape[0] != 1:
        raise ValueError(
            f"Expected one raster band, found {image.shape[0]} "
            f"in {src_path}"
        )

    band = image[0].astype(np.float32)

    if zero_is_nodata:
        band[band == 0] = np.nan

    if scale != 1.0:
        band = band * np.float32(scale)

    return band, metadata


def load_band_array(
    src_path: str,
    geometry: dict[str, Any],
    scale: float = 1e-4,
    zero_is_nodata: bool = True,
) -> np.ndarray:
    """
    Load one clipped raster band and return only its array.

    This function is retained for compatibility with existing code.
    """
    band, _ = load_band_with_metadata(
        src_path=src_path,
        geometry=geometry,
        scale=scale,
        zero_is_nodata=zero_is_nodata,
    )

    return band


def align_array_to_reference(
    source_array: np.ndarray,
    source_metadata: dict[str, Any],
    reference_metadata: dict[str, Any],
    resampling: Resampling,
    source_nodata: float | int | None = np.nan,
    destination_nodata: float | int | None = np.nan,
) -> np.ndarray:
    """
    Reproject and resample an array onto a reference raster grid.

    Parameters
    ----------
    source_array:
        Two-dimensional source array.

    source_metadata:
        Metadata associated with the source array.

    reference_metadata:
        Metadata associated with the target/reference grid.

    resampling:
        Rasterio resampling method.

        Use bilinear for continuous reflectance bands.
        Use nearest for categorical SCL data.

    Returns
    -------
    np.ndarray
        Source array aligned to the reference raster grid.
    """
    if source_array.ndim != 2:
        raise ValueError(
            "source_array must be a two-dimensional array."
        )

    source_crs = source_metadata.get("crs")
    source_transform = source_metadata.get("transform")

    destination_crs = reference_metadata.get("crs")
    destination_transform = reference_metadata.get("transform")
    destination_height = reference_metadata.get("height")
    ipdb.set_trace()
    destination_width = reference_metadata.get("width")

    if source_crs is None or source_transform is None:
        raise ValueError(
            "Source raster metadata must contain crs and transform."
        )

    if destination_crs is None or destination_transform is None:
        raise ValueError(
            "Reference raster metadata must contain crs and transform."
        )

    if destination_height is None or destination_width is None:
        raise ValueError(
            "Reference raster metadata must contain height and width."
        )

    destination = np.full(
        shape=(int(destination_height), int(destination_width)),
        fill_value=destination_nodata,
        dtype=np.float32,
    )

    reproject(
        source=source_array.astype(np.float32),
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=source_nodata,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        dst_nodata=destination_nodata,
        resampling=resampling,
    )

    return destination


def grids_match(
    first_metadata: dict[str, Any],
    second_metadata: dict[str, Any],
) -> bool:
    """
    Check whether two raster grids have the same shape, CRS and transform.
    """
    first_transform = first_metadata.get("transform")
    second_transform = second_metadata.get("transform")

    return (
        first_metadata.get("crs") == second_metadata.get("crs")
        and first_metadata.get("height") == second_metadata.get("height")
        and first_metadata.get("width") == second_metadata.get("width")
        and isinstance(first_transform, Affine)
        and isinstance(second_transform, Affine)
        and first_transform.almost_equals(second_transform)
    )
# %%
