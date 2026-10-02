import numpy as np
import pandas as pd
import torch

from liver_histo_ai.config import FusionConfig
from liver_histo_ai.omics.fusion import build_fusion_module, impute_missing_omics
from liver_histo_ai.omics.integration import (
    get_patient_omics,
    load_omics_tables,
    normalize_rnaseq,
    select_top_variable_genes,
)


def test_normalize_rnaseq_log1p():
    df = pd.DataFrame({"g1": [0, 10, 100], "g2": [5, 5, 5]})
    out = normalize_rnaseq(df, method="log1p_tpm")
    assert np.isclose(out.loc[0, "g1"], np.log1p(0))
    assert np.isclose(out.loc[2, "g1"], np.log1p(100))


def test_normalize_rnaseq_zscore():
    df = pd.DataFrame({"g1": [1.0, 2.0, 3.0]})
    out = normalize_rnaseq(df, method="zscore")
    assert abs(out["g1"].mean()) < 1e-6


def test_select_top_variable_genes():
    df = pd.DataFrame({"low_var": [1, 1, 1], "high_var": [1, 100, 5000]})
    top = select_top_variable_genes(df, top_k=1)
    assert list(top.columns) == ["high_var"]


def test_load_omics_tables_and_get_patient(tmp_path):
    rnaseq_path = tmp_path / "rnaseq.csv"
    pd.DataFrame({"patient_id": ["P1", "P2"], "g1": [1.0, 2.0], "g2": [3.0, 4.0]}).set_index(
        "patient_id"
    ).to_csv(rnaseq_path)

    tables = load_omics_tables(rnaseq_path=str(rnaseq_path), top_k_genes=2)
    patient = get_patient_omics("P1", tables)
    assert patient.rnaseq is not None
    assert patient.rnaseq.shape[0] == 2

    vector = patient.to_vector()
    assert vector.shape[0] == 2


def test_get_patient_omics_missing_patient_returns_none_fields(tmp_path):
    rnaseq_path = tmp_path / "rnaseq.csv"
    pd.DataFrame({"patient_id": ["P1"], "g1": [1.0]}).set_index("patient_id").to_csv(rnaseq_path)
    tables = load_omics_tables(rnaseq_path=str(rnaseq_path))
    patient = get_patient_omics("UNKNOWN", tables)
    assert patient.rnaseq is None
    assert patient.to_vector().shape[0] == 0


def test_impute_missing_omics_wrong_shape_returns_zeros():
    result = impute_missing_omics(np.array([1.0, 2.0]), expected_dim=5)
    assert result.shape == (5,)
    assert np.all(result == 0)


def test_impute_missing_omics_none_returns_zeros():
    result = impute_missing_omics(None, expected_dim=3)
    assert result.shape == (3,)


def test_build_fusion_module_late():
    cfg = FusionConfig(method="late", histo_dim=8, omics_dim=8, fused_dim=16)
    module = build_fusion_module(cfg, histo_in=10, omics_in=5)
    histo = torch.randn(4, 10)
    omics = torch.randn(4, 5)
    out = module(histo, omics)
    assert out.shape == (4, 16)


def test_build_fusion_module_early():
    cfg = FusionConfig(method="early", fused_dim=12)
    module = build_fusion_module(cfg, histo_in=10, omics_in=5)
    out = module(torch.randn(2, 10), torch.randn(2, 5))
    assert out.shape == (2, 12)


def test_build_fusion_module_attention():
    cfg = FusionConfig(method="attention", histo_dim=8, omics_dim=8, fused_dim=16)
    module = build_fusion_module(cfg, histo_in=10, omics_in=5)
    out = module(torch.randn(3, 10), torch.randn(3, 5))
    assert out.shape == (3, 16)


def test_build_fusion_module_invalid_method_raises():
    import pytest
    cfg = FusionConfig(method="not-a-method")
    with pytest.raises(ValueError):
        build_fusion_module(cfg, histo_in=10, omics_in=5)
