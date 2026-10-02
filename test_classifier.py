import numpy as np
import pytest

from liver_histo_ai.config import ClassifierConfig
from liver_histo_ai.models.classifier import TabularClassifier


def _make_toy_dataset(n=120, in_dim=6, num_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, in_dim))
    # make the label a deterministic-ish function of the features so a
    # trained classifier should do meaningfully better than chance
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 1.0).astype(int)
    y = np.clip(y, 0, num_classes - 1)
    return X.astype(np.float32), y


@pytest.mark.parametrize("model_type", ["xgboost", "mlp"])
def test_classifier_fit_predict_roundtrip(model_type):
    X, y = _make_toy_dataset()
    cfg = ClassifierConfig(model_type=model_type, num_classes=3)
    clf = TabularClassifier(cfg, in_dim=X.shape[1])
    metrics = clf.fit(X, y, epochs=20)
    assert "val_accuracy" in metrics
    assert 0.0 <= metrics["val_accuracy"] <= 1.0

    preds = clf.predict(X)
    assert preds.shape == (len(X),)
    probs = clf.predict_proba(X)
    assert probs.shape == (len(X), 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)


@pytest.mark.parametrize("model_type", ["xgboost", "mlp"])
def test_classifier_save_load_roundtrip(tmp_path, model_type):
    X, y = _make_toy_dataset()
    cfg = ClassifierConfig(model_type=model_type, num_classes=3)
    clf = TabularClassifier(cfg, in_dim=X.shape[1])
    clf.fit(X, y, epochs=10)

    ext = ".joblib" if model_type == "xgboost" else ".pt"
    path = tmp_path / f"model{ext}"
    clf.save(path)

    clf2 = TabularClassifier(cfg, in_dim=X.shape[1])
    clf2.load(path)
    preds_original = clf.predict(X)
    preds_loaded = clf2.predict(X)
    assert np.array_equal(preds_original, preds_loaded)


def test_xgboost_native_format_roundtrip_and_manifest(tmp_path):
    from liver_histo_ai.models.classifier import load_feature_manifest, save_feature_manifest

    X, y = _make_toy_dataset()
    cfg = ClassifierConfig(model_type="xgboost", num_classes=3)
    clf = TabularClassifier(cfg, in_dim=X.shape[1])
    clf.fit(X, y)
    path = tmp_path / "clf.json"
    clf.save(path)
    save_feature_manifest(path, ["a", "b", "c", "d"], omics_dim=2)

    clf2 = TabularClassifier(cfg, in_dim=X.shape[1])
    clf2.load(path)
    assert np.array_equal(clf.predict(X), clf2.predict(X))
    assert clf2.predict_proba(X).shape == (len(X), 3)
    assert load_feature_manifest(path) == {"histo_features": ["a", "b", "c", "d"], "omics_dim": 2}
