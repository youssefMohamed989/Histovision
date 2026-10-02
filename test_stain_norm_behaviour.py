import numpy as np

from liver_histo_ai.preprocessing.stain_norm import macenko_normalize, reinhard_normalize
from liver_histo_ai.synthetic import generate_tile


def _jittered_tiles():
    return [
        generate_tile(128, np.random.default_rng(11), grade=1, weights=(0.6, 0.2, 0, 0.2), stain_jitter=j)["rgb"]
        for j in (0.0, 0.6, 1.0, 1.4)
    ]


def _colour_spread(tiles):
    return np.array([t.reshape(-1, 3).mean(0) for t in tiles]).std(axis=0).mean()


def test_normalization_reduces_between_tile_colour_variation():
    tiles = _jittered_tiles()
    raw = _colour_spread(tiles)
    assert _colour_spread([macenko_normalize(t) for t in tiles]) < raw
    assert _colour_spread([reinhard_normalize(t) for t in tiles]) < raw


def test_reinhard_outputs_valid_image():
    out = reinhard_normalize(_jittered_tiles()[1])
    assert out.dtype == np.uint8 and out.shape == (128, 128, 3)
