import numpy as np
import pytest

from processing.spectral_indices import ndvi, ndre, ndwi, lswi, compute_all


def _band(value: float, shape=(4, 4)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


class TestNDVI:
    def test_healthy_vegetation(self):
        result = ndvi(nir=_band(0.8), red=_band(0.1))
        expected = (0.8 - 0.1) / (0.8 + 0.1)
        assert np.allclose(result, expected, atol=1e-5)

    def test_bare_soil(self):
        result = ndvi(nir=_band(0.3), red=_band(0.3))
        assert np.allclose(result, 0.0, atol=1e-5)

    def test_zero_denominator_returns_nan(self):
        result = ndvi(nir=_band(0.0), red=_band(0.0))
        assert np.all(np.isnan(result))

    def test_output_range(self):
        nir = np.random.uniform(0, 1, (10, 10)).astype(np.float32)
        red = np.random.uniform(0, 1, (10, 10)).astype(np.float32)
        result = ndvi(nir, red)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1) and np.all(valid <= 1)


class TestComputeAll:
    def test_returns_all_indices(self):
        bands = {
            "nir":      _band(0.7),
            "red":      _band(0.1),
            "red_edge": _band(0.4),
            "swir1":    _band(0.2),
            "swir2":    _band(0.15),
        }
        result = compute_all(bands)
        assert set(result.keys()) == {"ndvi", "ndre", "ndwi", "lswi"}

    def test_values_are_float32(self):
        bands = {k: _band(0.5) for k in ["nir", "red", "red_edge", "swir1", "swir2"]}
        for arr in compute_all(bands).values():
            assert arr.dtype == np.float32
