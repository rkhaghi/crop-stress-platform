import numpy as np
import pytest

from processing.cloud_mask import build_cloud_mask, apply_mask, valid_pixel_fraction
from processing.zonal_statistics import zonal_stats


class TestBuildCloudMask:
    def test_clear_pixels_are_valid(self):
        scl = np.array([[4, 5, 6], [4, 5, 6]], dtype=np.uint8)  # vegetation, bare, water
        mask = build_cloud_mask(scl)
        assert mask.all()

    def test_cloud_pixels_are_invalid(self):
        scl = np.array([[8, 9, 10]], dtype=np.uint8)  # cloud med, high, cirrus
        mask = build_cloud_mask(scl)
        assert not mask.any()

    def test_mixed_scene(self):
        scl = np.array([[4, 9, 5]], dtype=np.uint8)
        mask = build_cloud_mask(scl)
        assert mask[0, 0] and not mask[0, 1] and mask[0, 2]


class TestApplyMask:
    def test_invalid_pixels_become_nan(self):
        band = np.ones((3, 3), dtype=np.float32)
        mask = np.array([[True, False, True],
                          [True, True,  False],
                          [False, True, True]], dtype=bool)
        result = apply_mask(band, mask)
        assert np.isnan(result[mask == False]).all()
        assert np.all(result[mask] == 1.0)


class TestZonalStats:
    def test_basic_stats(self):
        arr = np.array([[0.5, 0.6, 0.7], [0.4, np.nan, 0.8]])
        stats = zonal_stats(arr, name="ndvi")
        assert "ndvi_mean" in stats
        assert abs(stats["ndvi_mean"] - np.nanmean(arr)) < 1e-5

    def test_all_nan_returns_nan_stats(self):
        arr = np.full((3, 3), np.nan)
        stats = zonal_stats(arr, name="ndvi")
        assert stats["ndvi_valid_frac"] == 0.0
        assert np.isnan(stats["ndvi_mean"])

    def test_valid_fraction(self):
        arr = np.array([[1.0, np.nan], [1.0, 1.0]])
        stats = zonal_stats(arr, name="x")
        assert abs(stats["x_valid_frac"] - 0.75) < 1e-5
