import torch

from liver_histo_ai.segmentation.losses import DiceCELoss, DiceLoss, FocalLoss, HoVerLoss


def test_dice_loss_perfect_prediction_near_zero():
    targets = torch.randint(0, 3, (2, 8, 8))
    logits = torch.zeros(2, 3, 8, 8)
    for b in range(2):
        for i in range(8):
            for j in range(8):
                logits[b, targets[b, i, j], i, j] = 10.0
    loss = DiceLoss(num_classes=3)(logits, targets)
    assert loss.item() < 0.05


def test_dice_loss_is_bounded():
    logits = torch.randn(2, 4, 16, 16)
    targets = torch.randint(0, 4, (2, 16, 16))
    loss = DiceLoss(num_classes=4)(logits, targets)
    assert 0.0 <= loss.item() <= 1.0


def test_focal_loss_runs():
    logits = torch.randn(2, 4, 16, 16)
    targets = torch.randint(0, 4, (2, 16, 16))
    loss = FocalLoss()(logits, targets)
    assert loss.item() >= 0


def test_dice_ce_loss_runs():
    logits = torch.randn(2, 4, 16, 16)
    targets = torch.randint(0, 4, (2, 16, 16))
    loss = DiceCELoss(num_classes=4)(logits, targets)
    assert loss.item() >= 0


def test_hover_loss_runs():
    np_logits = torch.randn(2, 2, 16, 16)
    np_target = torch.randint(0, 2, (2, 16, 16))
    hv_pred = torch.randn(2, 2, 16, 16)
    hv_target = torch.randn(2, 2, 16, 16)
    losses = HoVerLoss()(np_logits, np_target, hv_pred, hv_target)
    assert "total" in losses
    assert losses["total"].item() >= 0
