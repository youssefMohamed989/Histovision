import numpy as np

from liver_histo_ai.features.morphology import (
    aggregate_morphology_features,
    extract_morphology_features,
)


def test_extract_morphology_features_counts_instances(sample_instance_mask):
    df = extract_morphology_features(sample_instance_mask)
    n_instances = len(np.unique(sample_instance_mask)) - 1  # exclude background
    assert len(df) == n_instances
    assert "area" in df.columns
    assert "circularity" in df.columns
    assert (df["area"] > 0).all()


def test_extract_morphology_features_empty_mask():
    empty_mask = np.zeros((32, 32), dtype=np.int32)
    df = extract_morphology_features(empty_mask)
    assert len(df) == 0


def test_aggregate_morphology_features_shapes(sample_instance_mask):
    df = extract_morphology_features(sample_instance_mask)
    feats = aggregate_morphology_features(df)
    assert feats["morph_nucleus_count"] == len(df)
    assert "morph_area_mean" in feats
    assert "morph_area_std" in feats


def test_aggregate_morphology_features_empty_df_returns_empty_dict():
    import pandas as pd
    empty_df = pd.DataFrame()
    assert aggregate_morphology_features(empty_df) == {}
