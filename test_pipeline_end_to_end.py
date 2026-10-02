"""run_pipeline end to end with a mocked tiler (OpenSlide is not needed)."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from liver_histo_ai import pipeline
from liver_histo_ai.config import ClassifierConfig, PipelineConfig
from liver_histo_ai.models.classifier import TabularClassifier, save_feature_manifest
from liver_histo_ai.synthetic import generate_tile


@pytest.fixture()
def fake_tiler(tmp_path, monkeypatch):
    """Replace tile_slide with one that writes 12 synthetic tiles on a 4x3 grid."""
    def _tile_slide(slide_path, out_dir, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        rows = []
        for i in range(12):
            x, y = (i % 4) * 128, (i // 4) * 128
            path = out_dir / f"tile_x{x}_y{y}.png"
            Image.fromarray(generate_tile(128, rng, grade=i % 3)["rgb"]).save(path)
            rows.append({"x": x, "y": y, "path": str(path)})
        return pd.DataFrame(rows)

    monkeypatch.setattr(pipeline, "tile_slide", _tile_slide)


def _cfg(tmp_path) -> PipelineConfig:
    cfg = PipelineConfig(results_dir=str(tmp_path / "results"))
    cfg.train.device = "cpu"
    cfg.tissue_seg.checkpoint = None
    cfg.nuclei_seg.checkpoint = None
    cfg.classifier.checkpoint = str(tmp_path / "missing.json")
    return cfg


def test_runs_without_any_checkpoint_using_classical_segmenter(tmp_path, fake_tiler):
    result = pipeline.run_pipeline("slide.svs", _cfg(tmp_path))
    feats = result["histology_features"]
    assert result["num_tiles"] == 12
    assert feats["slide_nuc_count_mean"] > 0  # real nuclei were found, not random-weight noise
    assert "prediction" not in result


def test_hovernet_backend_requires_checkpoint(tmp_path, fake_tiler):
    cfg = _cfg(tmp_path)
    cfg.nuclei_seg.backend = "hovernet"
    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline("slide.svs", cfg)


def test_prediction_uses_training_feature_names_and_tolerates_missing_omics(tmp_path, fake_tiler):
    cfg = _cfg(tmp_path)
    names = list(pipeline.run_pipeline("slide.svs", cfg)["histology_features"])

    # train on a DIFFERENT column order, one column the slide lacks, and 3 omics columns
    train_names = names[:10][::-1] + ["feature_not_in_slide"]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, len(train_names) + 3)).astype(np.float32)
    y = rng.integers(0, 3, size=60)
    clf = TabularClassifier(ClassifierConfig(model_type="xgboost", num_classes=3))
    clf.fit(X, y)
    ckpt = tmp_path / "clf.json"
    clf.save(ckpt)
    save_feature_manifest(ckpt, train_names, omics_dim=3)

    cfg.classifier.checkpoint = str(ckpt)
    result = pipeline.run_pipeline("slide.svs", cfg, patient_id="NOT_IN_ANY_TABLE")
    probs = result["prediction"]["probabilities"]
    assert len(probs) == 3 and abs(sum(probs) - 1.0) < 1e-3
