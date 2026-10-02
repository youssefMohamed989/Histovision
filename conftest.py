import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_rgb_tile(rng):
    """A synthetic 256x256 RGB tile that looks vaguely tissue-like (not
    pure noise) so stain normalization and texture code has non-degenerate
    input to work with."""
    base = rng.integers(150, 220, size=(256, 256, 3), dtype=np.uint8)
    # sprinkle in some darker "nuclei" blobs
    for _ in range(40):
        cy, cx = rng.integers(10, 246, size=2)
        r = rng.integers(3, 8)
        yy, xx = np.ogrid[:256, :256]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        base[mask] = rng.integers(40, 100, size=3, dtype=np.uint8)
    return base


@pytest.fixture
def sample_instance_mask(rng):
    """A synthetic instance mask with a handful of disjoint blob instances."""
    mask = np.zeros((128, 128), dtype=np.int32)
    next_id = 1
    for _ in range(8):
        cy, cx = rng.integers(15, 113, size=2)
        r = rng.integers(4, 10)
        yy, xx = np.ogrid[:128, :128]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        mask[blob & (mask == 0)] = next_id
        next_id += 1
    return mask
