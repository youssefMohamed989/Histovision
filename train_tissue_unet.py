"""Train the tissue segmentation U-Net on a directory of tile/mask pairs.

Expects a dataset directory structured as:

    data/tissue_seg/
        images/*.png     # RGB tiles
        masks/*.png      # single-channel label maps, same filename as image

Each mask pixel value is a class index in [0, num_classes).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from liver_histo_ai.config import PipelineConfig
from liver_histo_ai.segmentation.losses import DiceCELoss
from liver_histo_ai.segmentation.tissue_unet import TissueUNet
from liver_histo_ai.utils.logging import get_logger
from liver_histo_ai.utils.seed import get_device, set_seed

logger = get_logger(__name__)


class TissueSegDataset(Dataset):
    def __init__(self, root: str | Path, input_size: int = 256):
        self.root = Path(root)
        self.image_paths = sorted((self.root / "images").glob("*.png"))
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.root / "masks" / img_path.name

        image = Image.open(img_path).convert("RGB").resize((self.input_size, self.input_size))
        mask = Image.open(mask_path).resize((self.input_size, self.input_size), Image.NEAREST)

        image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(np.array(mask)).long()
        return image_t, mask_t


def train(config_path: str, data_dir: str, epochs: int | None = None) -> None:
    cfg = PipelineConfig.from_yaml(config_path)
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    epochs = epochs or cfg.train.epochs

    dataset = TissueSegDataset(data_dir, input_size=cfg.tissue_seg.input_size)
    if len(dataset) == 0:
        raise RuntimeError(f"No training images found under {data_dir}/images")

    # NOTE: this is a random split by tile. If several tiles come from one slide/patient, split by
    # patient yourself (e.g. separate folders) to avoid optimistic validation loss.
    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                               num_workers=cfg.train.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                             num_workers=cfg.train.num_workers)

    model = TissueUNet.from_config(cfg.tissue_seg).to(device)
    criterion = DiceCELoss(num_classes=cfg.tissue_seg.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_checkpoint(cfg.tissue_seg.checkpoint, val_loss=val_loss, epoch=epoch)
            logger.info(f"New best model saved to {cfg.tissue_seg.checkpoint}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.early_stopping_patience:
                logger.info("Early stopping triggered")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/tissue_seg")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.data_dir, args.epochs)
