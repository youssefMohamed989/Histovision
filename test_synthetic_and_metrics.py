import numpy as np

from liver_histo_ai.metrics import confusion, dice_per_class, instance_metrics
from liver_histo_ai.segmentation.classical import segment_nuclei_classical
from liver_histo_ai.synthetic import generate_tile


def test_generate_tile_shapes_and_labels():
    d = generate_tile(128, np.random.default_rng(1), grade=2)
    assert d["rgb"].shape == (128, 128, 3) and d["rgb"].dtype == np.uint8
    assert d["tissue"].shape == (128, 128) and d["tissue"].max() <= 3
    assert d["instances"].max() == len(d["nucleus_region"]) - 1


def test_grade_increases_nuclear_size():
    areas = []
    for g in (0, 2):
        d = generate_tile(256, np.random.default_rng(3), grade=g, weights=(1, 0, 0, 0))
        ids, counts = np.unique(d["instances"], return_counts=True)
        areas.append(counts[ids > 0].mean())
    assert areas[1] > areas[0]


def test_classical_segmentation_finds_most_nuclei():
    d = generate_tile(256, np.random.default_rng(0), grade=1)
    pred = segment_nuclei_classical(d["rgb"])
    assert instance_metrics(pred, d["instances"])["f1"] > 0.6


def test_instance_metrics_perfect_and_empty():
    gt = np.zeros((20, 20), dtype=np.int32)
    gt[2:8, 2:8] = 1
    gt[12:18, 12:18] = 2
    perfect = instance_metrics(gt, gt)
    assert perfect["f1"] == 1.0 and perfect["aji"] == 1.0
    assert instance_metrics(np.zeros_like(gt), gt)["f1"] == 0.0


def test_dice_and_confusion():
    t = np.array([[0, 0, 1, 1]])
    assert np.allclose(dice_per_class(t, t, 2), [1.0, 1.0])
    cm = confusion(np.array([[0, 1, 1, 1]]), t, 2)
    assert np.allclose(cm, [[0.5, 0.5], [0.0, 1.0]])


def test_pale_stromal_nuclei_are_recovered():
    """Regression: a single global threshold used to miss almost all stromal nuclei."""
    from analysis_helpers import per_region_recall  # defined in tests/analysis_helpers.py

    d = generate_tile(384, np.random.default_rng(2024), grade=1, weights=(0.4, 0.3, 0.1, 0.2))
    pred = segment_nuclei_classical(d["rgb"])
    rec = per_region_recall(pred, d)
    assert rec[0] > 0.9  # tumor
    assert rec[1] > 0.6  # stroma
