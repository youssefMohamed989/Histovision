from liver_histo_ai.config import PipelineConfig


def test_default_config_values():
    cfg = PipelineConfig()
    assert cfg.tiling.tile_size == 512
    assert cfg.tissue_seg.num_classes == 4
    assert cfg.classifier.model_type == "xgboost"


def test_config_yaml_roundtrip(tmp_path):
    cfg = PipelineConfig()
    cfg.tiling.tile_size = 256
    cfg.classifier.num_classes = 5
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)

    loaded = PipelineConfig.from_yaml(path)
    assert loaded.tiling.tile_size == 256
    assert loaded.classifier.num_classes == 5


def test_load_default_yaml_from_repo():
    cfg = PipelineConfig.from_yaml("configs/default.yaml")
    assert cfg.tiling.tile_size == 512
    assert cfg.omics.top_k_genes == 200
