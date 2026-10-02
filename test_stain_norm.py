from pathlib import Path

import numpy as np
from PIL import Image

from liver_histo_ai.preprocessing.stain_norm import (
    macenko_normalize,
    normalize_tile,
    reinhard_normalize,
)


def test_macenko_normalize_preserves_shape_and_dtype(sample_rgb_tile):
    out = macenko_normalize(sample_rgb_tile)
    assert out.shape == sample_rgb_tile.shape
    assert out.dtype == np.uint8


def test_reinhard_normalize_preserves_shape_and_dtype(sample_rgb_tile):
    out = reinhard_normalize(sample_rgb_tile)
    assert out.shape == sample_rgb_tile.shape
    assert out.dtype == np.uint8


def test_normalize_tile_none_is_identity(sample_rgb_tile):
    out = normalize_tile(sample_rgb_tile, method="none")
    assert np.array_equal(out, sample_rgb_tile)


def test_normalize_tile_dispatch_invalid_method(sample_rgb_tile):
    import pytest
    with pytest.raises(ValueError):
        normalize_tile(sample_rgb_tile, method="not-a-method")


def test_macenko_handles_near_blank_tile():
    blank = np.full((64, 64, 3), 250, dtype=np.uint8)
    out = macenko_normalize(blank)
    assert out.shape == blank.shape


def test_fit_macenko_reference_then_normalize_matches_reference_tile(sample_rgb_tile):
    from liver_histo_ai.preprocessing.stain_norm import fit_macenko_reference

    stain_matrix, max_conc = fit_macenko_reference(sample_rgb_tile)
    # normalizing the reference tile toward its own fitted reference should
    # be close to a no-op (up to the fixed 99th percentile rescaling)
    out = macenko_normalize(sample_rgb_tile, reference_matrix=stain_matrix, reference_max_conc=max_conc)
    assert out.shape == sample_rgb_tile.shape
    diff = np.abs(out.astype(int) - sample_rgb_tile.astype(int))
    assert diff.mean() < 15


def test_fit_macenko_reference_cross_slide_reduces_color_gap():
    """Fitting a reference from a real H&E slide and normalizing a DIFFERENT
    real H&E slide toward it should bring their mean colour closer together
    than they were unnormalized -- the actual cross slide harmonization use
    case, tested on genuine microscopy images rather than synthetic ones."""
    from liver_histo_ai.preprocessing.stain_norm import fit_macenko_reference

    real_dir = Path(__file__).resolve().parents[1] / "data" / "real_samples"
    a = np.array(Image.open(real_dir / "source.png").convert("RGB"))
    b = np.array(Image.open(real_dir / "target.png").convert("RGB"))

    stain_matrix, max_conc = fit_macenko_reference(b)
    a_normalized = macenko_normalize(a, reference_matrix=stain_matrix, reference_max_conc=max_conc)

    before = np.abs(a.reshape(-1, 3).mean(0) - b.reshape(-1, 3).mean(0)).mean()
    after = np.abs(a_normalized.reshape(-1, 3).mean(0) - b.reshape(-1, 3).mean(0)).mean()
    assert after < before


def test_fit_reinhard_reference_then_normalize_matches_reference_tile(sample_rgb_tile):
    from liver_histo_ai.preprocessing.stain_norm import fit_reinhard_reference

    mean, std = fit_reinhard_reference(sample_rgb_tile)
    out = reinhard_normalize(sample_rgb_tile, target_mean=mean, target_std=std)
    diff = np.abs(out.astype(int) - sample_rgb_tile.astype(int))
    assert diff.mean() < 10
