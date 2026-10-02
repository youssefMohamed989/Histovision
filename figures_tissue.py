"""Tissue level analysis figures (segmentation, nuclei, texture, spatial, stain)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import (
    FIG_DIR,
    GRADE_COLORS,
    GRADE_NAMES,
    REGION_CMAP,
    bare,
    outline,
    panel_label,
    per_gt_best_iou,
    predict_tissue,
    tile_features,
)
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy import ndimage as ndi
from scipy.stats import kruskal

from liver_histo_ai.features.morphology import extract_morphology_features
from liver_histo_ai.features.texture import extract_glcm_features
from liver_histo_ai.metrics import confusion, dice_per_class, instance_metrics
from liver_histo_ai.preprocessing.stain_norm import macenko_normalize, reinhard_normalize
from liver_histo_ai.segmentation.classical import hematoxylin_channel, segment_nuclei_classical
from liver_histo_ai.synthetic import REGION_COLORS, REGION_NAMES, generate_tile

REGION_LEGEND = [Patch(color=REGION_COLORS[n], label=n) for n in REGION_NAMES]


def make_test_slide(seed: int = 2024) -> dict:
    """Find a 512 px test slide in which all four tissue regions are well represented."""
    for s in range(seed, seed + 200):
        d = generate_tile(512, np.random.default_rng(s), grade=1,
                          weights=(0.4, 0.25, 0.15, 0.2), stain_jitter=0.3)
        frac = np.bincount(d["tissue"].ravel(), minlength=4) / d["tissue"].size
        if frac.min() > 0.08:
            return d
    raise RuntimeError("no balanced test slide found")


def fig_training(hist: dict) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
    e = np.arange(1, len(hist["train_loss"]) + 1)
    ax[0].plot(e, hist["train_loss"], label="train", color="#264653", lw=2)
    ax[0].plot(e, hist["val_loss"], label="validation", color="#e76f51", lw=2)
    ax[0].set(xlabel="epoch", ylabel="Dice + cross entropy loss", title="Tissue U-Net training loss")
    ax[0].legend()
    ax[1].plot(e, hist["val_dice"], color="#2a9d8f", lw=2, marker="o", ms=3)
    ax[1].set(xlabel="epoch", ylabel="mean Dice (validation)", title="Validation segmentation quality", ylim=(0, 1))
    panel_label(ax[0], "a")
    panel_label(ax[1], "b")
    fig.savefig(FIG_DIR / "fig02_unet_training.png")
    plt.close(fig)


def fig_tissue_segmentation(slide: dict, pred: np.ndarray) -> dict:
    gt = slide["tissue"]
    dice = dice_per_class(pred, gt, 4)
    cm = confusion(pred, gt, 4)
    acc = float((pred == gt).mean())
    fig = plt.figure(figsize=(12, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0]); ax.imshow(slide["rgb"]); bare(ax)
    ax.set_title("Synthetic H&E slide (512 x 512 px)"); panel_label(ax, "a")
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(gt, cmap=REGION_CMAP, vmin=0, vmax=3, interpolation="nearest"); bare(ax)
    ax.set_title("Ground truth tissue regions"); panel_label(ax, "b")
    ax.legend(handles=REGION_LEGEND, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.09))
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(pred, cmap=REGION_CMAP, vmin=0, vmax=3, interpolation="nearest"); bare(ax)
    ax.set_title(f"U-Net prediction (pixel accuracy {acc:.1%})"); panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(slide["rgb"]); err = np.ma.masked_where(pred == gt, np.ones_like(pred))
    ax.imshow(err, cmap=ListedRed(), alpha=0.75); bare(ax)
    ax.set_title("Misclassified pixels (red)"); panel_label(ax, "d")

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    ax.set_xticks(range(4), REGION_NAMES, rotation=30); ax.set_yticks(range(4), REGION_NAMES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
    ax.set(xlabel="predicted", ylabel="ground truth", title="Row normalised confusion matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.06); panel_label(ax, "e")

    ax = fig.add_subplot(gs[1, 2])
    ax.bar(REGION_NAMES, dice, color=[REGION_COLORS[n] for n in REGION_NAMES])
    for i, d in enumerate(dice):
        ax.text(i, d + 0.01, f"{d:.2f}", ha="center")
    ax.set(ylim=(0, 1.08), ylabel="Dice coefficient", title="Per class Dice"); panel_label(ax, "f")
    fig.savefig(FIG_DIR / "fig01_tissue_segmentation.png")
    plt.close(fig)
    return {"pixel_accuracy": acc, "dice": dict(zip(REGION_NAMES, map(float, dice), strict=True)),
            "mean_dice": float(np.nanmean(dice))}


def ListedRed():
    from matplotlib.colors import ListedColormap
    return ListedColormap(["#ff1744"])


def fig_nuclei(slide: dict, inst: np.ndarray) -> dict:
    gt = slide["instances"]
    m = instance_metrics(inst, gt)
    best = per_gt_best_iou(inst, gt)
    region_of = slide["nucleus_region"]
    rec = {}
    for r, name in enumerate(REGION_NAMES):
        ids = [g for g in best if region_of[g] == r]
        rec[name] = float(np.mean([best[g] >= 0.5 for g in ids])) if ids else float("nan")
    h = hematoxylin_channel(slide["rgb"])
    sl = (slice(0, 200), slice(0, 200))

    fig, ax = plt.subplots(2, 3, figsize=(12, 7.6))
    ax = ax.ravel()
    ax[0].imshow(slide["rgb"][sl]); ax[0].set_title("Input tile crop (200 x 200 px)")
    ax[1].imshow(h[sl], cmap="magma"); ax[1].set_title("Hematoxylin channel (colour deconvolution)")
    ov = outline(outline(slide["rgb"], gt, (0, 200, 90)), inst, (255, 230, 0))
    ax[2].imshow(ov[sl]); ax[2].set_title("Contours: ground truth (green), predicted (yellow)")
    rng = np.random.default_rng(1)
    colors = rng.random((inst.max() + 1, 3)) * 0.8 + 0.2
    colors[0] = 0
    ax[3].imshow(colors[inst][sl]); ax[3].set_title(f"Predicted instances ({inst.max()} nuclei on full slide)")
    for a in ax[:4]:
        bare(a)
    a = ax[4]
    names = [n for n in REGION_NAMES if not np.isnan(rec[n])]
    a.bar(names, [rec[n] for n in names], color=[REGION_COLORS[n] for n in names])
    for i, n in enumerate(names):
        a.text(i, rec[n] + 0.015, f"{rec[n]:.2f}", ha="center")
    a.set(ylim=(0, 1.1), ylabel="recall at IoU 0.5", title="Nucleus detection recall by tissue region")
    a = ax[5]
    keys = ["precision", "recall", "f1", "mean_iou", "aji"]
    a.barh(["precision", "recall", "F1", "mean IoU", "AJI"][::-1], [m[k] for k in keys][::-1], color="#264653")
    for i, k in enumerate(keys[::-1]):
        a.text(m[k] + 0.01, i, f"{m[k]:.2f}", va="center")
    a.set(xlim=(0, 1.1), title="Instance segmentation accuracy (full 512 px slide)")
    for a, letter in zip(ax, "abcdef", strict=True):
        panel_label(a, letter)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_nuclei_segmentation.png")
    plt.close(fig)
    return {**m, "recall_by_region": rec, "n_gt": int(gt.max()), "n_pred": int(inst.max())}


def nuclei_table(inst: np.ndarray, tissue_pred: np.ndarray, rgb: np.ndarray) -> pd.DataFrame:
    morph = extract_morphology_features(inst)
    h = hematoxylin_channel(rgb)
    morph["hema_mean"] = ndi.mean(h, inst, morph["instance_id"].values)
    r = np.clip(morph["centroid_row"].round().astype(int), 0, inst.shape[0] - 1)
    c = np.clip(morph["centroid_col"].round().astype(int), 0, inst.shape[1] - 1)
    morph["region"] = [REGION_NAMES[k] for k in tissue_pred[r, c]]
    return morph


def fig_morphology(df: pd.DataFrame) -> dict:
    feats = [("area", "Nuclear area (px$^2$)"), ("eccentricity", "Eccentricity"), ("circularity", "Circularity"),
             ("solidity", "Solidity"), ("axis_ratio", "Axis ratio"), ("hema_mean", "Mean hematoxylin OD")]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.6))
    stats = {}
    for a, (col, title), letter in zip(ax.ravel(), feats, "abcdef", strict=True):
        data, labels, cols = [], [], []
        for r in REGION_NAMES:
            v = df.loc[df.region == r, col].dropna().values
            if len(v) >= 8:
                data.append(v); labels.append(f"{r}\n(n={len(v)})"); cols.append(REGION_COLORS[r])
        parts = a.violinplot(data, showmedians=True, showextrema=False)
        for body, col_ in zip(parts["bodies"], cols, strict=True):
            body.set_facecolor(col_); body.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        a.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
        p = kruskal(*data).pvalue if len(data) > 1 else float("nan")
        a.set_title(title); a.text(0.98, 0.97, f"Kruskal Wallis p = {p:.1e}", transform=a.transAxes,
                                   ha="right", va="top", fontsize=8)
        panel_label(a, letter)
        stats[col] = {r: float(df.loc[df.region == r, col].median()) for r in REGION_NAMES
                      if (df.region == r).sum() >= 8}
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_nuclear_morphology.png")
    plt.close(fig)
    return stats


def window_grid(slide_rgb: np.ndarray, tissue_pred: np.ndarray, inst: np.ndarray, win: int = 32):
    n = slide_rgb.shape[0] // win
    rows = []
    for i in range(n):
        for j in range(n):
            sl = (slice(i * win, (i + 1) * win), slice(j * win, (j + 1) * win))
            g = extract_glcm_features(slide_rgb[sl], distances=(1, 3), angles_deg=(0, 90))
            region = np.bincount(tissue_pred[sl].ravel(), minlength=4).argmax()
            ids = np.unique(inst[sl])[1:]
            areas = [(inst == k).sum() for k in ids] if len(ids) else []
            rows.append({"i": i, "j": j, "region": REGION_NAMES[region], "n_nuclei": len(ids),
                         "mean_area": float(np.mean(areas)) if areas else np.nan, **g})
    return pd.DataFrame(rows), n


def fig_texture_spatial(slide: dict, tissue_pred: np.ndarray, inst: np.ndarray) -> dict:
    win = 32
    grid, n = window_grid(slide["rgb"], tissue_pred, inst, win)
    cols = ["glcm_contrast_d1", "glcm_contrast_d3", "glcm_dissimilarity_d3", "glcm_homogeneity_d1",
            "glcm_homogeneity_d3", "glcm_energy_d1", "glcm_correlation_d1", "glcm_correlation_d3"]
    z = (grid[cols] - grid[cols].mean()) / (grid[cols].std() + 1e-9)
    z["region"] = grid["region"]
    hm = z.groupby("region").mean().reindex(list(REGION_NAMES)).dropna(how="all")

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    a = ax[0, 0]
    im = a.imshow(hm.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto"); a.grid(False)
    a.set_xticks(range(len(cols)), [c.replace("glcm_", "") for c in cols], rotation=40, ha="right")
    a.set_yticks(range(len(hm)), hm.index)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            a.text(j, i, f"{hm.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    a.set_title("GLCM texture signature by tissue region (z score)"); plt.colorbar(im, ax=a, fraction=0.04)

    dens = grid["n_nuclei"].values.reshape(n, n) / (win * win) * 1000
    a = ax[0, 1]
    im = a.imshow(dens, cmap="viridis", extent=(0, 512, 512, 0)); bare(a)
    a.contour(np.linspace(0, 512, 512), np.linspace(0, 512, 512), tissue_pred, levels=[0.5, 1.5, 2.5],
              colors="white", linewidths=0.7, alpha=0.8)
    a.set_title("Nuclear density (per 1000 px$^2$) with tissue borders"); plt.colorbar(im, ax=a, fraction=0.046)

    area = grid["mean_area"].values.reshape(n, n)
    a = ax[1, 0]
    im = a.imshow(area, cmap="magma", extent=(0, 512, 512, 0)); bare(a)
    a.contour(np.linspace(0, 512, 512), np.linspace(0, 512, 512), tissue_pred, levels=[0.5, 1.5, 2.5],
              colors="white", linewidths=0.7, alpha=0.8)
    a.set_title("Mean nuclear area per window (px$^2$)"); plt.colorbar(im, ax=a, fraction=0.046)

    from scipy.spatial import cKDTree
    morph = extract_morphology_features(inst)
    pts = morph[["centroid_row", "centroid_col"]].values
    nn = cKDTree(pts).query(pts, k=2)[0][:, 1]
    r = np.clip(pts[:, 0].round().astype(int), 0, 511); c = np.clip(pts[:, 1].round().astype(int), 0, 511)
    reg = np.array(REGION_NAMES)[tissue_pred[r, c]]
    a = ax[1, 1]
    data = [nn[reg == n_] for n_ in REGION_NAMES if (reg == n_).sum() >= 8]
    names = [n_ for n_ in REGION_NAMES if (reg == n_).sum() >= 8]
    bp = a.boxplot(data, tick_labels=names, patch_artist=True, showfliers=False)
    for patch, n_ in zip(bp["boxes"], names, strict=True):
        patch.set_facecolor(REGION_COLORS[n_]); patch.set_alpha(0.8)
    a.set(ylabel="nearest neighbour distance (px)", title="Nuclear spacing by tissue region")
    for a_, letter in zip(ax.ravel(), "abcd", strict=True):
        panel_label(a_, letter)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_texture_and_spatial.png")
    plt.close(fig)
    return {"n_windows": len(grid), "median_nn_dist_by_region": {n_: float(np.median(nn[reg == n_]))
                                                                for n_ in names}}


def fig_stain_norm() -> dict:
    jit = [0.0, 0.6, 1.0, 1.4]
    tiles = [generate_tile(256, np.random.default_rng(11), grade=1, weights=(0.6, 0.2, 0.0, 0.2),
                           stain_jitter=j) for j in jit]
    ref_gt = tiles[0]["instances"]
    variants = {"raw": [t["rgb"] for t in tiles],
                "Macenko": [macenko_normalize(t["rgb"]) for t in tiles],
                "Reinhard": [reinhard_normalize(t["rgb"]) for t in tiles]}
    fig, ax = plt.subplots(3, 4, figsize=(11, 8.6))
    stats = {}
    for r, (name, imgs) in enumerate(variants.items()):
        means = np.array([im.reshape(-1, 3).mean(0) for im in imgs])
        disp = float(means.std(axis=0).mean())
        f1 = float(np.mean([instance_metrics(segment_nuclei_classical(im), ref_gt)["f1"] for im in imgs]))
        stats[name] = {"between_tile_color_std": disp, "mean_nuclei_f1": f1}
        for c, im in enumerate(imgs):
            ax[r, c].imshow(im); bare(ax[r, c])
            if c == 0:
                ax[r, c].set_ylabel(name, rotation=90, fontsize=11, fontweight="bold")
                ax[r, c].set_title(f"{name}: colour std {disp:.1f}, nuclei F1 {f1:.2f}", loc="left", fontsize=9)
    fig.suptitle("Stain normalization across simulated staining variability (same tissue, 4 colour profiles)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_stain_normalization.png")
    plt.close(fig)
    return stats


def fig_grade_gallery(unet) -> dict:
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 7.8))
    out = {}
    for g in range(3):
        d = generate_tile(256, np.random.default_rng(100 + g), grade=g, weights=(1, 0, 0, 0))
        inst = segment_nuclei_classical(d["rgb"])
        areas = np.bincount(inst.ravel()).astype(float)
        area_img = areas[inst]
        ax[0, g].imshow(d["rgb"]); ax[0, g].set_title(GRADE_NAMES[g], color=GRADE_COLORS[g])
        ax[1, g].imshow(np.clip(d["rgb"] * 0.55 + 90, 0, 255).astype(np.uint8))
        im = ax[1, g].imshow(np.ma.masked_where(inst == 0, area_img), cmap="plasma", vmin=20, vmax=300)
        for a in (ax[0, g], ax[1, g]):
            bare(a)
        feats = tile_features(d["rgb"], unet)
        out[GRADE_NAMES[g]] = {"nuclei": int(inst.max()),
                               "mean_area": float(feats["histo_morph_area_mean"]),
                               "area_cv": float(feats["histo_morph_area_std"] / feats["histo_morph_area_mean"])}
        ax[1, g].text(0.02, 0.02, f"n={inst.max()}  mean area={out[GRADE_NAMES[g]]['mean_area']:.0f}  "
                                  f"CV={out[GRADE_NAMES[g]]['area_cv']:.2f}", transform=ax[1, g].transAxes,
                      fontsize=8, color="white", backgroundcolor="#00000088")
    fig.colorbar(im, ax=ax[1, :], fraction=0.02, label="nuclear area (px$^2$)")
    fig.suptitle("Tumor differentiation gallery: nuclear size and pleomorphism increase with grade",
                 fontweight="bold")
    fig.savefig(FIG_DIR / "fig12_grade_gallery.png")
    plt.close(fig)
    return out


__all__ = [
    "fig_grade_gallery",
    "fig_morphology",
    "fig_nuclei",
    "fig_stain_norm",
    "fig_texture_spatial",
    "fig_tissue_segmentation",
    "fig_training",
    "make_test_slide",
    "nuclei_table",
    "predict_tissue",
    "to_rgb",
]
