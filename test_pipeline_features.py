import numpy as np
import pandas as pd
import torch

from liver_histo_ai.config import PipelineConfig
from liver_histo_ai.pipeline import aggregate_slide_features, extract_tile_features
from liver_histo_ai.segmentation.nuclei_seg import HoVerNet
from liver_histo_ai.segmentation.tissue_unet import TissueUNet
from liver_histo_ai.synthetic import generate_tile


def _tile():
    return generate_tile(128, np.random.default_rng(0), grade=1)["rgb"]


def test_advanced_tile_features_in_pipeline():
    cfg = PipelineConfig()
    hover = HoVerNet(base_channels=8)
    unet = TissueUNet(base_channels=8)
    feats = extract_tile_features(_tile(), hover, torch.device("cpu"), cfg, tissue_model=unet)
    assert any(k.startswith("tex_") for k in feats)
    assert any(k.startswith("arch_") for k in feats)
    assert any(k.startswith("glcm_") for k in feats)


def test_basic_mode_still_available():
    cfg = PipelineConfig()
    cfg.features.advanced = False
    feats = extract_tile_features(_tile(), HoVerNet(base_channels=8), torch.device("cpu"), cfg)
    assert not any(k.startswith(("tex_", "graph_", "arch_")) for k in feats)
    assert any(k.startswith("glcm_") for k in feats)


def test_slide_aggregation_stats_and_habitats():
    rng = np.random.default_rng(0)
    n = 24
    tiles = [{"nuc_area_mean": float(a), "nuc_count": float(c), "tex_h_od_mean": float(h),
              "tex_h_positive_area": float(hp), "glcm_contrast_d1": float(g)}
             for a, c, h, hp, g in zip(np.r_[rng.normal(60, 3, 12), rng.normal(120, 3, 12)],
                                       rng.normal(50, 2, n), rng.normal(0.1, 0.01, n),
                                       np.r_[rng.normal(0.2, 0.01, 12), rng.normal(0.6, 0.01, 12)],
                                       rng.normal(5, 0.3, n), strict=True)]
    xs, ys = np.meshgrid(np.arange(6), np.arange(4))
    manifest = pd.DataFrame({"x": xs.ravel() * 512, "y": ys.ravel() * 512})
    out = aggregate_slide_features(tiles, stats=("mean", "std", "q90"), manifest=manifest, n_habitats=2)
    assert "slide_nuc_area_mean_q90" in out and "slide_nuc_area_mean_std" in out
    assert out["slide_habitat_shannon"] > 0.8  # two equally sized habitats
    assert "slide_habitat_boundary_frac" in out
    assert set(aggregate_slide_features(tiles)) == {f"slide_{k}_mean" for k in tiles[0]}
