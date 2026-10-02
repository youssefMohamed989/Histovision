import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from liver_histo_ai.features.advanced import (
    AdvancedFeatureConfig,
    AdvancedHistologyExtractor,
    feature_catalog,
    feature_group,
)
from liver_histo_ai.features.architecture import (
    architecture_features,
    interface_profile,
    signed_distance_to_tumor,
)
from liver_histo_ai.features.cell_graph import (
    clark_evans,
    delaunay_features,
    density_heterogeneity,
    graph_features,
    mixing_features,
    morans_i,
    mst_features,
    orientation_features,
    ripley_l_minus_r,
    voronoi_features,
)
from liver_histo_ai.features.chromatin import chromatin_descriptors, masked_glcm_features
from liver_histo_ai.features.habitats import (
    gini,
    habitat_clustering,
    slide_summary,
    spatial_habitat_metrics,
)
from liver_histo_ai.features.nuclear_shape import shape_descriptors
from liver_histo_ai.features.selection import drop_constant, prune_correlated, stability_selection
from liver_histo_ai.features.texture_advanced import (
    fractal_dimension,
    gabor_features,
    lacunarity,
    texture_features,
)
from liver_histo_ai.segmentation.classical import segment_nuclei_classical
from liver_histo_ai.synthetic import generate_tile

SHAPE = (256, 256)


# ---------------------------------------------------------------- point patterns
def regular_points(spacing=12, jitter=1.0, seed=0):
    rng = np.random.default_rng(seed)
    g = np.arange(10, 246, spacing)
    pts = np.array([(r, c) for r in g for c in g], dtype=float)
    return pts + rng.uniform(-jitter, jitter, pts.shape)


def poisson_points(n=441, seed=0):
    return np.random.default_rng(seed).uniform(0, 256, (n, 2))


def clustered_points(seed=0, parents=30, children=15, sd=6.0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(20, 236, (parents, 2))
    pts = np.concatenate([c + rng.normal(0, sd, (children, 2)) for c in centers])
    return np.clip(pts, 0, 255)


# ---------------------------------------------------------------- nuclear shape
def _grid():
    return np.mgrid[-24:25, -24:25]


def disk_mask(r=12):
    yy, xx = _grid()
    return yy**2 + xx**2 <= r * r


def star_mask(radius=12, amplitude=0.35, lobes=5):
    yy, xx = _grid()
    theta = np.arctan2(yy, xx)
    return np.hypot(yy, xx) <= radius * (1 + amplitude * np.cos(lobes * theta))


def ellipse_mask(a=16, b=8):
    yy, xx = _grid()
    return (xx / a) ** 2 + (yy / b) ** 2 <= 1


def test_shape_circle_is_smooth_and_round():
    d = shape_descriptors(disk_mask())
    assert d["radial_cv"] < 0.05
    assert d["roughness"] < 1.05
    assert d["n_concavities"] == 0
    assert 0.7 < d["bending_energy"] < 1.5


def test_shape_star_is_irregular():
    circle, star = shape_descriptors(disk_mask()), shape_descriptors(star_mask())
    assert star["radial_cv"] > 3 * circle["radial_cv"]
    assert star["n_concavities"] >= 3
    assert star["bending_energy"] > circle["bending_energy"]
    assert star["fd_harmonic5"] > circle["fd_harmonic5"]
    assert star["concavity_area_frac"] > circle["concavity_area_frac"]


def test_shape_fourier_ellipticity_orders_shapes():
    circle = shape_descriptors(disk_mask())["fd_ellipticity"]
    ellipse = shape_descriptors(ellipse_mask())["fd_ellipticity"]
    assert ellipse > circle + 0.1


def test_shape_descriptors_rotation_invariant():
    e = ellipse_mask()
    a, b = shape_descriptors(e), shape_descriptors(np.rot90(e))
    for k in ("fd_ellipticity", "radial_cv", "roughness"):
        assert a[k] == pytest.approx(b[k], abs=0.05)


def test_shape_tiny_mask_returns_empty():
    m = np.zeros((10, 10), dtype=bool)
    m[4:6, 4:6] = True
    assert shape_descriptors(m) == {}


# ---------------------------------------------------------------- chromatin
def test_masked_glcm_uniform_and_checkerboard():
    mask = np.ones((16, 16), dtype=bool)
    flat = masked_glcm_features(np.full((16, 16), 3), mask)
    assert flat["contrast"] == 0 and flat["energy"] == pytest.approx(1.0)
    yy, xx = np.mgrid[:16, :16]
    checker = ((yy + xx) % 2) * 7
    busy = masked_glcm_features(checker, mask)
    assert busy["contrast"] > 10 and busy["homogeneity"] < 0.5


def test_masked_glcm_ignores_pixels_outside_mask():
    q = np.zeros((16, 16), dtype=int)
    q[:, 8:] = 7  # bright half outside the mask
    mask = np.zeros((16, 16), dtype=bool)
    mask[:, :8] = True
    assert masked_glcm_features(q, mask)["contrast"] == 0


def test_chromatin_margination_detects_rim_loading():
    mask = disk_mask(12)
    yy, xx = _grid()
    r = np.hypot(yy, xx)
    rim_heavy = np.where(mask, 0.2 + 0.6 * (r / 12) ** 3, 0.02)
    core_heavy = np.where(mask, 0.8 - 0.6 * (r / 12) ** 3, 0.02)
    ring = ~mask & (r <= 15)
    rgb = np.full(mask.shape + (3,), 150, dtype=np.uint8)
    args = (np.zeros(mask.shape), rgb, mask, ring, (0.0, 1.0))
    rim = chromatin_descriptors(rim_heavy, *args)
    core = chromatin_descriptors(core_heavy, *args)
    assert rim["margination"] > 1.0 > core["margination"]
    assert rim["contrast_ring"] > 0


def test_chromatin_clumpy_nucleus_has_more_clumps():
    rng = np.random.default_rng(0)
    mask = disk_mask(12)
    smooth = np.where(mask, 0.5 + 0.005 * rng.normal(size=mask.shape), 0.0)
    clumpy = smooth.copy()
    for cy, cx in [(-5, -5), (5, 5), (-5, 6), (6, -6)]:
        clumpy[24 + cy - 1 : 24 + cy + 2, 24 + cx - 1 : 24 + cx + 2] = 1.0
    ring = np.zeros_like(mask)
    rgb = np.full(mask.shape + (3,), 120, dtype=np.uint8)
    a = chromatin_descriptors(smooth, np.zeros(mask.shape), rgb, mask, ring, (0.0, 1.0))
    b = chromatin_descriptors(clumpy, np.zeros(mask.shape), rgb, mask, ring, (0.0, 1.0))
    assert a["clump_count"] == 0
    assert b["clump_count"] == 4
    assert b["h_kurtosis"] > a["h_kurtosis"]


# ---------------------------------------------------------------- cell graph
def test_clark_evans_orders_patterns():
    area = 256.0 * 256.0
    reg = clark_evans(regular_points(), area)["clark_evans_R"]
    rnd = clark_evans(poisson_points(), area)["clark_evans_R"]
    clu = clark_evans(clustered_points(), area)["clark_evans_R"]
    assert reg > 1.6 > rnd > 0.8 > clu


def test_ripley_sign_matches_pattern():
    radii = (8, 16, 32)
    clustered = ripley_l_minus_r(clustered_points(), SHAPE, radii)
    regular = ripley_l_minus_r(regular_points(), SHAPE, radii)
    assert clustered[1] > 3.0
    assert regular[0] < -3.0
    poisson = ripley_l_minus_r(poisson_points(n=600), SHAPE, radii)
    assert abs(poisson[1]) < 3.0


def test_ripley_too_few_points_is_nan():
    assert np.isnan(ripley_l_minus_r(poisson_points(n=5), SHAPE)).all()


def test_delaunay_regular_more_regular_than_random():
    reg = delaunay_features(regular_points())
    rnd = delaunay_features(poisson_points())
    assert reg["delaunay_edge_cv"] < rnd["delaunay_edge_cv"]
    assert reg["delaunay_regularity_mean"] > rnd["delaunay_regularity_mean"]


def test_voronoi_and_mst_disorder():
    reg = voronoi_features(regular_points(), SHAPE)
    rnd = voronoi_features(poisson_points(), SHAPE)
    assert reg["voronoi_area_cv"] < rnd["voronoi_area_cv"]
    assert mst_features(regular_points())["mst_edge_cv"] < mst_features(poisson_points())["mst_edge_cv"]


def test_degenerate_points_do_not_crash():
    collinear = np.stack([np.arange(20.0), np.arange(20.0)], axis=1)
    assert delaunay_features(collinear) == {}
    assert delaunay_features(collinear[:3]) == {}
    assert voronoi_features(collinear, SHAPE) == {}


def test_orientation_order_aligned_vs_random():
    pts = poisson_points(200)
    aligned = orientation_features(pts, np.full(200, 0.3))
    rnd = orientation_features(pts, np.random.default_rng(1).uniform(-np.pi / 2, np.pi / 2, 200))
    assert aligned["orientation_order"] == pytest.approx(1.0, abs=1e-6)
    assert aligned["orientation_local_alignment"] == pytest.approx(1.0, abs=1e-6)
    assert rnd["orientation_order"] < 0.3


def test_morans_i_spatial_structure():
    pts = poisson_points(300)
    smooth = pts[:, 0] + pts[:, 1]
    noise = np.random.default_rng(2).normal(size=300)
    assert morans_i(pts, smooth) > 0.8
    assert abs(morans_i(pts, noise)) < 0.2


def test_mixing_excess_positive_for_structured_sizes():
    pts = poisson_points(300)
    structured = 50 + pts[:, 0]
    random_sizes = np.random.default_rng(3).uniform(20, 80, 300)
    assert mixing_features(pts, structured)["mixing_excess"] > 0.15
    assert abs(mixing_features(pts, random_sizes)["mixing_excess"]) < 0.1


def test_density_heterogeneity_clustered_vs_uniform():
    uniform = density_heterogeneity(regular_points(), SHAPE)
    clustered = density_heterogeneity(clustered_points(), SHAPE)
    assert clustered["morisita_index"] > uniform["morisita_index"]
    assert clustered["density_gini"] > uniform["density_gini"]


def test_graph_features_end_to_end():
    pts = poisson_points(120)
    df = pd.DataFrame({"centroid_row": pts[:, 0], "centroid_col": pts[:, 1],
                       "area": np.random.default_rng(0).uniform(30, 90, 120),
                       "orientation": np.random.default_rng(1).uniform(-1.5, 1.5, 120),
                       "chr_h_mean": np.random.default_rng(2).uniform(0.1, 0.4, 120)})
    feats = graph_features(df, SHAPE)
    assert all(k.startswith("graph_") for k in feats)
    assert "graph_ripley_L_minus_r_16" in feats and "graph_moran_hematoxylin" in feats
    assert graph_features(df.head(5), SHAPE) == {}


# ---------------------------------------------------------------- texture
def test_fractal_dimension_line_vs_filled():
    filled = np.ones((128, 128), dtype=bool)
    line = np.zeros((128, 128), dtype=bool)
    line[64, :] = True
    assert fractal_dimension(filled)["fractal_dimension"] == pytest.approx(2.0, abs=0.05)
    assert fractal_dimension(line)["fractal_dimension"] == pytest.approx(1.0, abs=0.1)


def test_lacunarity_gappy_higher_than_uniform():
    uniform = np.ones((128, 128), dtype=bool)
    rng = np.random.default_rng(0)
    gappy = np.zeros((128, 128), dtype=bool)
    for cy, cx in rng.integers(10, 118, (6, 2)):
        gappy[cy - 8 : cy + 8, cx - 8 : cx + 8] = True
    assert lacunarity(gappy)["lacunarity_r16"] > lacunarity(uniform)["lacunarity_r16"] + 0.5


def test_gabor_anisotropy_detects_oriented_texture():
    _yy, xx = np.mgrid[:96, :96]
    stripes = 0.5 + 0.5 * np.sin(2 * np.pi * 0.16 * xx)
    isotropic = np.random.default_rng(0).uniform(size=(96, 96))
    a = gabor_features(stripes)["gabor_f0.16_anisotropy"]
    b = gabor_features(isotropic)["gabor_f0.16_anisotropy"]
    assert a > b


def test_texture_features_keys():
    d = generate_tile(128, np.random.default_rng(0))
    feats = texture_features(d["rgb"], d["instances"] > 0)
    assert all(k.startswith("tex_") for k in feats)
    assert {"tex_fractal_dimension", "tex_h_positive_area", "tex_lbp_P8R1_entropy"} <= set(feats)


# ---------------------------------------------------------------- architecture
def _split_tissue(size=200):
    t = np.ones((size, size), dtype=int)  # stroma
    t[:, : size // 2] = 0  # tumor left half
    return t


def test_signed_distance_sign_convention():
    s = signed_distance_to_tumor(_split_tissue())
    assert s[100, 10] < 0 < s[100, 190]


def test_architecture_invasive_front_density():
    t = _split_tissue()
    rng = np.random.default_rng(0)
    tumor_pts = np.column_stack([rng.uniform(0, 200, 700), rng.uniform(0, 100, 700)])
    stroma_pts = np.column_stack([rng.uniform(0, 200, 50), rng.uniform(100, 200, 50)])
    pts = np.vstack([tumor_pts, stroma_pts])
    nuclei = pd.DataFrame({"centroid_row": pts[:, 0], "centroid_col": pts[:, 1], "area": 60.0})
    f = architecture_features(t, nuclei)
    assert f["arch_frac_tumor"] == pytest.approx(0.5)
    assert f["arch_interface_frac_stroma"] == pytest.approx(1.0)
    assert f["arch_dens_tumor_core"] > 5 * f["arch_dens_far"]
    assert f["arch_n_tumor_nests"] == 1
    centers, density, counts, _area = interface_profile(t, nuclei)
    assert np.nanmax(density[centers < 0]) > np.nanmax(density[centers > 0])
    assert counts.sum() <= len(nuclei)


def test_architecture_without_tumor_is_safe():
    t = np.ones((100, 100), dtype=int)
    f = architecture_features(t)
    assert f["arch_frac_tumor"] == 0 and "arch_n_tumor_nests" not in f


# ---------------------------------------------------------------- habitats / selection
def test_habitat_clustering_recovers_groups_and_orders():
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0, 0], 0.3, (60, 3))
    b = rng.normal([5, 5, 5], 0.3, (60, 3))
    df = pd.DataFrame(np.vstack([a, b]), columns=["x", "y", "z"])
    truth = np.r_[np.zeros(60), np.ones(60)]
    labels, model = habitat_clustering(df, n_habitats=2, n_components=2, order_by="x")
    assert adjusted_rand_score(truth, labels) == pytest.approx(1.0)
    assert labels[:60].mean() == 0  # lower x mean gets habitat 0
    assert (model.predict(df) == labels).all()


def test_spatial_habitat_metrics_block_vs_checkerboard():
    block = np.zeros((10, 10), dtype=int)
    block[:, 5:] = 1
    checker = (np.indices((10, 10)).sum(axis=0) % 2).astype(int)
    b = spatial_habitat_metrics(block, 2)
    c = spatial_habitat_metrics(checker, 2)
    assert b["habitat_boundary_frac"] < c["habitat_boundary_frac"]
    assert b["habitat_shannon"] == pytest.approx(1.0)
    assert b["habitat_largest_patch_frac"] > c["habitat_largest_patch_frac"]


def test_slide_summary_and_gini():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 5.0, 5.0, 5.0]})
    s = slide_summary(df)
    assert s["slide_a_mean"] == pytest.approx(2.5) and s["slide_b_std"] == 0
    assert gini(np.array([1, 1, 1, 1])) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 0, 10])) > 0.7


def test_feature_selection_tools():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 40)
    signal = y + rng.normal(0, 0.4, 120)
    df = pd.DataFrame({
        "signal": signal,
        "signal_copy": signal * 2 + 1,
        "noise": rng.normal(size=120),
        "const": 3.0,
    })
    assert "const" not in drop_constant(df)
    kept = prune_correlated(df[["signal", "signal_copy", "noise"]], threshold=0.95)
    assert len(kept) == 2 and "noise" in kept
    freq = stability_selection(df[["signal", "noise"]], y, n_boot=20, top_k=1)
    assert freq.index[0] == "signal" and freq["signal"] == 1.0


# ---------------------------------------------------------------- extractor
@pytest.fixture(scope="module")
def tile_and_features():
    d = generate_tile(192, np.random.default_rng(5), grade=2, weights=(0.6, 0.2, 0.05, 0.15))
    inst = segment_nuclei_classical(d["rgb"])
    feats = AdvancedHistologyExtractor().extract(d["rgb"], inst, d["tissue"])
    return d, inst, feats


def test_extractor_produces_all_groups(tile_and_features):
    _, _, feats = tile_and_features
    groups = {feature_group(k) for k in feats}
    assert groups == {"nuclear morphology", "nuclear shape", "chromatin", "cell graph",
                      "multiscale texture", "tissue architecture"}
    assert len(feats) > 150
    assert all(np.isfinite(v) for v in feats.values())


def test_extractor_is_deterministic(tile_and_features):
    d, inst, feats = tile_and_features
    again = AdvancedHistologyExtractor().extract(d["rgb"], inst, d["tissue"])
    assert again.keys() == feats.keys()
    assert all(again[k] == pytest.approx(feats[k], nan_ok=True) for k in feats)


def test_extractor_config_toggles(tile_and_features):
    d, inst, full = tile_and_features
    cfg = AdvancedFeatureConfig(shape=False, chromatin=False, graph=False, texture=False,
                                architecture=False, aggregations=("mean",))
    minimal = AdvancedHistologyExtractor(cfg).extract(d["rgb"], inst)
    assert len(minimal) < len(full) / 4
    assert not any(k.startswith(("graph_", "tex_", "arch_", "nuc_shp_", "nuc_chr_")) for k in minimal)


def test_extractor_handles_empty_tile():
    rgb = np.full((64, 64, 3), 235, dtype=np.uint8)
    feats = AdvancedHistologyExtractor().extract(rgb, np.zeros((64, 64), dtype=np.int32))
    assert isinstance(feats, dict)


def test_extractor_max_nuclei_cap(tile_and_features):
    d, inst, _ = tile_and_features
    ex = AdvancedHistologyExtractor(AdvancedFeatureConfig(max_nuclei=20))
    table = ex.nuclear_table(d["rgb"], inst)
    assert table["shp_roughness"].notna().sum() <= 20
    assert len(table) == inst.max()


def test_grade_increases_advanced_irregularity():
    """Higher grade tumors should read as larger and more pleomorphic through the new features."""
    out = {}
    for g in (0, 2):
        d = generate_tile(256, np.random.default_rng(40 + g), grade=g, weights=(1, 0, 0, 0))
        f = AdvancedHistologyExtractor(AdvancedFeatureConfig(architecture=False)).extract(
            d["rgb"], segment_nuclei_classical(d["rgb"]))
        out[g] = f
    assert out[2]["nuc_area_mean"] > out[0]["nuc_area_mean"]
    assert out[2]["nuc_shp_radial_cv_mean"] > out[0]["nuc_shp_radial_cv_mean"]


def test_feature_catalog_lists_every_feature(tile_and_features):
    _, _, feats = tile_and_features
    text = feature_catalog(feats)
    assert text.count("| `") == len(feats)
    assert "## Cell graph" in text and "## Tissue architecture" in text
