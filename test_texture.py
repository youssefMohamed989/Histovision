from liver_histo_ai.features.texture import extract_glcm_features


def test_extract_glcm_features_keys_and_ranges(sample_rgb_tile):
    feats = extract_glcm_features(sample_rgb_tile, distances=(1, 3), angles_deg=(0, 90))
    expected_keys = {
        "glcm_contrast_d1", "glcm_contrast_d3",
        "glcm_homogeneity_d1", "glcm_homogeneity_d3",
        "glcm_energy_d1", "glcm_energy_d3",
    }
    assert expected_keys.issubset(feats.keys())
    # homogeneity and energy are normalized in [0, 1]
    assert 0.0 <= feats["glcm_homogeneity_d1"] <= 1.0
    assert 0.0 <= feats["glcm_energy_d1"] <= 1.0


def test_extract_glcm_features_single_distance_angle(sample_rgb_tile):
    feats = extract_glcm_features(sample_rgb_tile, distances=(1,), angles_deg=(0,))
    assert "glcm_contrast_d1" in feats
    assert len(feats) == 6  # one per Haralick property
