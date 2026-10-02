import pandas as pd

from liver_histo_ai.features.spatial import extract_spatial_features


def test_extract_spatial_features_basic():
    df = pd.DataFrame(
        {
            "instance_id": [1, 2, 3, 4],
            "centroid_row": [10, 20, 10, 100],
            "centroid_col": [10, 20, 30, 100],
        }
    )
    feats = extract_spatial_features(df, tile_area_px2=256 * 256, k_neighbors=2)
    assert "spatial_density_per_1000px2" in feats
    assert "spatial_nn_dist_mean" in feats
    assert feats["spatial_nn_dist_mean"] >= 0


def test_extract_spatial_features_single_point_is_safe():
    df = pd.DataFrame({"instance_id": [1], "centroid_row": [5], "centroid_col": [5]})
    feats = extract_spatial_features(df, tile_area_px2=100)
    assert feats["spatial_density_per_1000px2"] == 0.0


def test_extract_spatial_features_empty_df():
    df = pd.DataFrame(columns=["instance_id", "centroid_row", "centroid_col"])
    feats = extract_spatial_features(df, tile_area_px2=100)
    assert feats["spatial_density_per_1000px2"] == 0.0
