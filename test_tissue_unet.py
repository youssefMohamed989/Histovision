import tempfile
from pathlib import Path

import torch

from liver_histo_ai.config import TissueSegConfig
from liver_histo_ai.segmentation.tissue_unet import TissueUNet


def test_forward_pass_output_shape():
    model = TissueUNet(in_channels=3, num_classes=4, base_channels=8)
    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 4, 64, 64)


def test_predict_mask_shape():
    model = TissueUNet(in_channels=3, num_classes=4, base_channels=8)
    x = torch.randn(1, 3, 64, 64)
    mask = model.predict_mask(x)
    assert mask.shape == (1, 64, 64)
    assert mask.max() < 4


def test_from_config_builds_matching_architecture():
    cfg = TissueSegConfig(num_classes=5, in_channels=3, base_channels=8)
    model = TissueUNet.from_config(cfg)
    x = torch.randn(1, 3, 32, 32)
    logits = model(x)
    assert logits.shape[1] == 5


def test_checkpoint_roundtrip():
    model = TissueUNet(in_channels=3, num_classes=4, base_channels=8)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "model.pt"
        model.save_checkpoint(ckpt_path, val_loss=0.5)
        assert ckpt_path.exists()

        model2 = TissueUNet(in_channels=3, num_classes=4, base_channels=8)
        model2.load_checkpoint(ckpt_path)
        x = torch.randn(1, 3, 32, 32)
        assert torch.allclose(model(x), model2(x), atol=1e-5)
