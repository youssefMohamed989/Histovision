"""Shared helpers for the demonstration analysis: plotting style, tissue
U-Net training on synthetic slides, and per tile feature extraction that
calls the real ``liver_histo_ai`` feature modules."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from liver_histo_ai.features.morphology import (
    aggregate_morphology_features,
    extract_morphology_features,
)
from liver_histo_ai.features.spatial import extract_spatial_features
from liver_histo_ai.features.texture import extract_glcm_features
from liver_histo_ai.metrics import dice_per_class
from liver_histo_ai.segmentation.classical import (
    hematoxylin_channel,
    segment_nuclei_classical,
)
from liver_histo_ai.segmentation.losses import DiceCELoss
from liver_histo_ai.segmentation.tissue_unet import TissueUNet
from liver_histo_ai.synthetic import REGION_COLORS, REGION_NAMES, generate_tile

FIG_DIR = Path("analysis/figures")
RES_DIR = Path("analysis/results")
GRADE_NAMES = ("well differentiated", "moderately differentiated", "poorly differentiated")
GRADE_COLORS = ("#2a9d8f", "#e9c46a", "#e76f51")
REGION_CMAP = ListedColormap([REGION_COLORS[n] for n in REGION_NAMES])


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)


def panel_label(ax, letter: str) -> None:
    ax.text(-0.05, 1.10, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="right")


def bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def make_patches(n_slides: int, rng: np.random.Generator, jitter: float, crop: int = 128):
    imgs, masks = [], []
    for _ in range(n_slides):
        d = generate_tile(
            256, rng, grade=int(rng.integers(0, 3)),
            weights=tuple(rng.dirichlet([2.5, 1.5, 0.8, 1.5])), stain_jitter=jitter,
        )
        for _ in range(4):
            y, x = rng.integers(0, 256 - crop + 1, size=2)
            imgs.append(d["rgb"][y:y + crop, x:x + crop])
            masks.append(d["tissue"][y:y + crop, x:x + crop])
    x_t = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
    y_t = torch.from_numpy(np.stack(masks)).long()
    return x_t, y_t


def train_tissue_unet(seed: int = 7, epochs: int = 16, log=print):
    """Train the package's TissueUNet on synthetic slides. Returns the model and history."""
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    xtr, ytr = make_patches(60, rng, jitter=0.6)
    xva, yva = make_patches(12, rng, jitter=0.6)
    model = TissueUNet(in_channels=3, num_classes=4, base_channels=12)
    crit = DiceCELoss(num_classes=4)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = {"train_loss": [], "val_loss": [], "val_dice": []}
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        running = 0.0
        for i in range(0, len(perm), 16):
            idx = perm[i:i + 16]
            xb, yb = xtr[idx], ytr[idx]
            if torch.rand(1) < 0.5:
                xb, yb = xb.flip(-1), yb.flip(-1)
            if torch.rand(1) < 0.5:
                xb, yb = xb.flip(-2), yb.flip(-2)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(xva)
            vloss = crit(logits, yva).item()
            dice = np.nanmean(dice_per_class(logits.argmax(1).numpy(), yva.numpy(), 4))
        hist["train_loss"].append(running / len(xtr))
        hist["val_loss"].append(vloss)
        hist["val_dice"].append(float(dice))
        log(f"  epoch {ep:2d}/{epochs} train {hist['train_loss'][-1]:.3f} val {vloss:.3f} dice {dice:.3f}")
    return model, hist


@torch.no_grad()
def predict_tissue(model: TissueUNet, rgb: np.ndarray) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return model(x).argmax(1)[0].numpy()


def per_gt_best_iou(pred: np.ndarray, gt: np.ndarray) -> dict[int, float]:
    """Best IoU achieved by any predicted nucleus for every ground truth nucleus."""
    p_area = np.bincount(pred.ravel())
    out: dict[int, float] = {}
    for gid, sl in enumerate(ndi.find_objects(gt), start=1):
        if sl is None:
            continue
        g = gt[sl] == gid
        overlap = pred[sl][g]
        overlap = overlap[overlap > 0]
        if overlap.size == 0:
            out[gid] = 0.0
            continue
        counts = np.bincount(overlap)
        pid = int(counts.argmax())
        inter = counts[pid]
        out[gid] = inter / (g.sum() + p_area[pid] - inter)
    return out


def tile_features(rgb: np.ndarray, unet: TissueUNet) -> dict[str, float]:
    """Full per tile histology feature vector using the package feature modules."""
    inst = segment_nuclei_classical(rgb)
    morph = extract_morphology_features(inst)
    if len(morph):
        h = hematoxylin_channel(rgb)
        morph["hema_mean"] = ndi.mean(h, inst, morph["instance_id"].values)
    feats = aggregate_morphology_features(morph)
    feats.update(extract_spatial_features(morph, rgb.shape[0] * rgb.shape[1]))
    feats.update(extract_glcm_features(rgb, distances=(1, 3), angles_deg=(0, 90)))
    tissue = predict_tissue(unet, rgb)
    frac = np.bincount(tissue.ravel(), minlength=4) / tissue.size
    for name, f in zip(REGION_NAMES, frac, strict=True):
        feats[f"tissue_frac_{name}"] = float(f)
    return {f"histo_{k}": v for k, v in feats.items()}


def outline(rgb: np.ndarray, inst: np.ndarray, color=(255, 215, 0)) -> np.ndarray:
    from skimage.segmentation import find_boundaries

    out = rgb.copy()
    out[find_boundaries(inst, mode="inner")] = color
    return out


def build_cohort(n: int, unet: TissueUNet, seed: int = 11, log=print):
    """Simulated liver cancer cohort: per patient tile features, RNA, mutations, survival.

    A latent aggressiveness score z = grade + noise drives BOTH the histology
    tile appearance and the omics profile, each with independent noise, so the
    two modalities carry complementary evidence about the same underlying state.
    """
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), n // 3)
    rng.shuffle(y)
    z = y + rng.normal(0, 0.55, size=len(y))

    rows = []
    for i, zi in enumerate(z):
        w = rng.dirichlet([9, 1.6, 0.4, 1.2])
        tile = generate_tile(256, rng, grade=int(np.clip(round(zi), 0, 2)),
                             weights=tuple(w), stain_jitter=0.4)
        rows.append(tile_features(tile["rgb"], unet))
        if (i + 1) % 30 == 0:
            log(f"  extracted histology features for {i + 1}/{len(z)} tiles")
    histo = pd.DataFrame(rows).fillna(0.0)

    up = ["AFP", "GPC3", "MKI67", "TOP2A", "EPCAM", "MYC", "VEGFA", "CCND1", "KRT19", "SALL4", "CDK1", "TERT"]
    down = ["ALB", "APOA1", "CYP3A4", "CYP2E1", "HNF4A", "SERPINA1", "TTR", "APOC3"]
    weak = ["CD8A", "PDCD1", "COL1A1", "ACTA2", "TGFB1"]
    noise = [f"GENE_{i:03d}" for i in range(1, 21)]
    genes = up + down + weak + noise
    effect = np.concatenate([
        rng.uniform(0.35, 0.7, len(up)), -rng.uniform(0.35, 0.7, len(down)),
        rng.uniform(-0.15, 0.15, len(weak)), np.zeros(len(noise)),
    ])
    base = rng.normal(6.0, 1.2, len(genes))
    expr = base[None, :] + z[:, None] * effect[None, :] + rng.normal(0, 1.1, (len(z), len(genes)))
    rna = pd.DataFrame(expr, columns=[f"rna_{g}" for g in genes])

    def sig(x):
        return 1 / (1 + np.exp(-x))

    mut = pd.DataFrame({
        "mut_TP53": rng.random(len(z)) < sig(-1.6 + 0.9 * z),
        "mut_CTNNB1": rng.random(len(z)) < sig(-0.7 - 0.5 * z),
        "mut_TERT": rng.random(len(z)) < sig(0.2 + 0.1 * z),
        "mut_ARID1A": rng.random(len(z)) < sig(-1.8 + 0.3 * z),
        "mut_AXIN1": rng.random(len(z)) < sig(-2.2 + 0.2 * z),
    }).astype(float)

    afp_z = (rna["rna_AFP"] - rna["rna_AFP"].mean()) / rna["rna_AFP"].std()
    hazard = np.exp(0.75 * z + 0.45 * mut["mut_TP53"].values + 0.2 * afp_z.values)
    t_event = 60.0 * rng.weibull(1.4, len(z)) / hazard
    t_cens = rng.uniform(15, 84, len(z))
    duration = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(int)

    ids = [f"SIM-{i:03d}" for i in range(len(z))]
    meta = pd.DataFrame({"patient_id": ids, "grade": y, "latent_z": z, "duration_months": duration, "event": event})
    return meta, histo, rna, mut
