"""Validate the pipeline on REAL histology images, not synthetic ones.

    python analysis/run_real_data_validation.py

Every other analysis script in this repo (``run_analysis.py``,
``run_advanced.py``, ``run_graph_mil.py``) uses the synthetic tile generator
in ``liver_histo_ai.synthetic`` so that segmentation accuracy can be scored
against exact ground truth. This script instead runs the same production
code -- stain normalization, classical nuclei segmentation, the full
``AdvancedHistologyExtractor`` feature set, and the tissue U-Net trained
earlier on synthetic data -- on real, genuine microscopy images, with no
ground truth and no cherry picking of what to show.

Image provenance (see ``data/real_samples/SOURCES.md`` for full detail):

- ``target.png`` / ``source.png``: two different real H&E stained tumor
  slides used as the standard stain-normalization test pair in the
  torchstain library (MIT licensed), fetched from
  ``github.com/EIDOSLAB/torchstain``.
- ``skimage_ihc.png``: a real DAB/hematoxylin immunohistochemistry image of
  human colon tissue, bundled with scikit-image
  (``skimage.data.immunohistochemistry``), no network fetch required.

None of these are liver tissue and none come with nuclei or tissue ground
truth -- the sandboxed network this ran in cannot reach TCGA/GDC, so a real
TCGA-LIHC slide could not be pulled in for this run (``scripts/
download_tcga_lihc.py`` does that once network access to the GDC API is
available). What this script DOES honestly establish: the segmentation,
stain normalization and feature extraction code runs correctly end to end on
genuine microscopy images of real tissue, not just on the synthetic
generator, and it reports where the synthetic-trained tissue model fails to
generalize rather than hiding it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from common import FIG_DIR, RES_DIR, bare, panel_label, setup_style
from PIL import Image

from liver_histo_ai.features.advanced import (
    AdvancedFeatureConfig,
    AdvancedHistologyExtractor,
)
from liver_histo_ai.features.morphology import extract_morphology_features
from liver_histo_ai.preprocessing.stain_norm import (
    fit_macenko_reference,
    fit_reinhard_reference,
    macenko_normalize,
    reinhard_normalize,
)
from liver_histo_ai.segmentation.classical import (
    hematoxylin_channel,
    segment_nuclei_classical,
)
from liver_histo_ai.segmentation.tissue_unet import TissueUNet
from liver_histo_ai.synthetic import REGION_NAMES

REAL_DIR = Path("data/real_samples")
IMAGES = {
    "torchstain target": "target.png",
    "torchstain source": "source.png",
    "skimage IHC (colon)": "skimage_ihc.png",
}


def load(name: str) -> np.ndarray:
    return np.array(Image.open(REAL_DIR / IMAGES[name]).convert("RGB"))


def outline(rgb: np.ndarray, inst: np.ndarray, color=(255, 215, 0)) -> np.ndarray:
    from skimage.segmentation import find_boundaries

    out = rgb.copy()
    out[find_boundaries(inst, mode="inner")] = color
    return out


def fig_real_nuclei_segmentation(images: dict[str, np.ndarray]) -> dict:
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.4))
    metrics = {}
    for col, (name, rgb) in enumerate(images.items()):
        inst = segment_nuclei_classical(rgb)
        table = extract_morphology_features(inst)
        ax[0, col].imshow(rgb)
        ax[0, col].set_title(f"{name}\n({rgb.shape[1]} x {rgb.shape[0]} px, real image)", fontsize=9)
        ov = outline(rgb, inst)
        ax[1, col].imshow(ov)
        ax[1, col].set_title(f"{len(table)} nuclei detected, no training required", fontsize=9)
        for a in ax[:, col]:
            bare(a)
        metrics[name] = {
            "n_nuclei_detected": len(table),
            "mean_area_px2": float(table["area"].mean()) if len(table) else float("nan"),
            "mean_circularity": float(table["circularity"].mean()) if len(table) else float("nan"),
        }
    for a_, letter in zip(ax.ravel(), "abcdef", strict=True):
        panel_label(a_, letter)
    fig.suptitle("Real image test 1: classical (weight-free) nuclei segmentation on genuine H&E/IHC slides",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig19_real_nuclei_segmentation.png")
    plt.close(fig)
    return metrics


def fig_real_stain_normalization(source: np.ndarray, target: np.ndarray) -> dict:
    stain_matrix, max_conc = fit_macenko_reference(target)
    macenko_matched = macenko_normalize(source, reference_matrix=stain_matrix, reference_max_conc=max_conc)
    mean, std = fit_reinhard_reference(target)
    reinhard_matched = reinhard_normalize(source, target_mean=mean, target_std=std)

    def gap(a, b):
        return float(np.abs(a.reshape(-1, 3).astype(float).mean(0) - b.reshape(-1, 3).astype(float).mean(0)).mean())

    before = gap(source, target)
    after_macenko = gap(macenko_matched, target)
    after_reinhard = gap(reinhard_matched, target)

    fig, ax = plt.subplots(2, 3, figsize=(12.5, 8.2))
    panels = [
        ("Real source slide (unnormalized)", source),
        ("Real target slide (reference)", target),
        (f"Source -> Macenko toward target\nmean colour gap {before:.1f} -> {after_macenko:.1f}", macenko_matched),
        ("Target (reference, repeated)", target),
        (f"Source -> Reinhard toward target\nmean colour gap {before:.1f} -> {after_reinhard:.1f}", reinhard_matched),
        ("Hematoxylin channel of matched source\n(Macenko)", None),
    ]
    for a, (title, img) in zip(ax.ravel(), panels, strict=True):
        if img is not None:
            a.imshow(img)
        else:
            a.imshow(hematoxylin_channel(macenko_matched), cmap="magma")
        a.set_title(title, fontsize=9)
        bare(a)
    for a_, letter in zip(ax.ravel(), "abcdef", strict=True):
        panel_label(a_, letter)
    fig.suptitle("Real image test 2: cross slide stain normalization, fit on one real slide, applied to another",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig20_real_stain_normalization.png")
    plt.close(fig)
    return {"mean_colour_gap_before": before, "mean_colour_gap_after_macenko": after_macenko,
            "mean_colour_gap_after_reinhard": after_reinhard}


def fig_real_features_and_domain_gap(images: dict[str, np.ndarray], unet: TissueUNet) -> dict:
    names = list(images)
    ex = AdvancedHistologyExtractor(AdvancedFeatureConfig(architecture=False))
    feats = {}
    for name, rgb in images.items():
        inst = segment_nuclei_classical(rgb)
        feats[name] = ex.extract(rgb, inst)

    compare_cols = ["nuc_area_mean", "nuc_circularity_mean", "nuc_eccentricity_mean",
                    "nuc_shp_roughness_mean", "nuc_chr_h_entropy_mean", "tex_h_positive_area",
                    "tex_edge_density", "graph_delaunay_regularity_mean"]
    labels = ["nuclear area", "circularity", "eccentricity", "contour roughness",
              "chromatin entropy", "hematoxylin+ area", "edge density", "Delaunay regularity"]

    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.4, wspace=0.35)

    a = fig.add_subplot(gs[0, :2])
    mat = np.array([[feats[n].get(c, np.nan) for n in names] for c in compare_cols])
    z = (mat - np.nanmean(mat, axis=1, keepdims=True)) / (np.nanstd(mat, axis=1, keepdims=True) + 1e-9)
    im = a.imshow(z, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    a.set_xticks(range(len(names)), names, fontsize=8)
    a.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(names)):
            a.text(j, i, f"{mat[i, j]:.2g}", ha="center", va="center", fontsize=7)
    a.set_title("Real feature values across 3 independent real images (colour = z score across images)")
    plt.colorbar(im, ax=a, fraction=0.03)
    panel_label(a, "a")

    a = fig.add_subplot(gs[0, 2])
    n_feats = [len(feats[n]) for n in names]
    a.bar(names, n_feats, color="#264653")
    for i, v in enumerate(n_feats):
        a.text(i, v + 2, str(v), ha="center", fontsize=8)
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    a.set(title="Features extracted per real image\n(nuclear morphology, shape, chromatin, texture, cell graph)")
    panel_label(a, "b")

    domain = {}
    for col, (name, rgb) in enumerate(images.items()):
        with torch.no_grad():
            x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            probs = torch.softmax(unet(x), dim=1)[0]
            pred = probs.argmax(0).numpy()
        confidence = float(probs.max(0).values.mean())
        frac = np.bincount(pred.ravel(), minlength=4) / pred.size
        domain[name] = {"mean_confidence": confidence,
                        **{f"frac_{r}": float(f) for r, f in zip(REGION_NAMES, frac, strict=True)}}
        a = fig.add_subplot(gs[1, col])
        a.imshow(rgb)
        a.imshow(pred, cmap="tab10", vmin=0, vmax=3, alpha=0.45, interpolation="nearest")
        a.set_title(f"{name}\nU-Net (trained on SYNTHETIC data only)\nconfidently WRONG: {confidence:.2f} softmax "
                    f"conf. on near-uniform blocks", fontsize=8)
        bare(a)
        panel_label(a, "cde"[col])
    fig.suptitle("Real image test 3: feature extraction works on real tissue; the synthetic-only tissue "
                 "classifier does not transfer (honest domain-gap check)", fontweight="bold")
    fig.savefig(FIG_DIR / "fig21_real_features_domain_gap.png")
    plt.close(fig)
    return {"feature_summary": {n: {c: feats[n].get(c) for c in compare_cols} for n in names},
            "n_features": dict(zip(names, n_feats, strict=True)), "tissue_unet_domain_gap": domain}


def main() -> None:
    setup_style()
    if not (REAL_DIR / "target.png").exists():
        raise SystemExit(
            f"Real sample images not found under {REAL_DIR}/. See data/real_samples/SOURCES.md "
            "for how they were obtained (small MIT/BSD licensed test fixtures + a scikit-image "
            "bundled sample); re-run that fetch before this script."
        )
    images = {name: load(name) for name in IMAGES}
    metrics: dict = {
        "note": "Genuine real microscopy images (see data/real_samples/SOURCES.md), no synthetic data.",
        "images": {n: {"shape": list(im.shape)} for n, im in images.items()},
    }

    print("real image test 1: nuclei segmentation")
    metrics["nuclei_segmentation"] = fig_real_nuclei_segmentation(images)

    print("real image test 2: cross slide stain normalization")
    metrics["stain_normalization"] = fig_real_stain_normalization(images["torchstain source"], images["torchstain target"])

    print("real image test 3: advanced features + tissue U-Net domain gap")
    unet = TissueUNet(in_channels=3, num_classes=4, base_channels=12)
    unet.load_checkpoint("checkpoints/demo_tissue_unet.pt")
    unet.eval()
    metrics["features_and_domain_gap"] = fig_real_features_and_domain_gap(images, unet)

    RES_DIR.mkdir(parents=True, exist_ok=True)
    with open(RES_DIR / "real_data_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    write_report(metrics)
    print("done")


def write_report(m: dict) -> None:
    n = m["nuclei_segmentation"]
    s = m["stain_normalization"]
    d = m["features_and_domain_gap"]["tissue_unet_domain_gap"]
    lines = [
        "# Real data validation",
        "",
        "> This is the ONLY analysis in this repository that runs on genuine microscopy",
        "> images rather than the synthetic generator. See `data/real_samples/SOURCES.md`",
        "> for exactly where each image came from and its licence.",
        "",
        "## Test 1: nuclei segmentation on real slides (no training required)",
        *[f"- {name}: {v['n_nuclei_detected']} nuclei detected, mean area {v['mean_area_px2']:.0f} px^2, "
          f"mean circularity {v['mean_circularity']:.2f}" for name, v in n.items()],
        "",
        "## Test 2: cross slide stain normalization (fit on one real slide, apply to another)",
        f"- Mean colour gap before normalization: {s['mean_colour_gap_before']:.2f}",
        f"- After Macenko: {s['mean_colour_gap_after_macenko']:.2f}",
        f"- After Reinhard: {s['mean_colour_gap_after_reinhard']:.2f}",
        "",
        "## Test 3: advanced feature extraction + honest domain-gap check",
        f"- Features extracted per real image: {m['features_and_domain_gap']['n_features']}",
        "- Tissue U-Net (trained ONLY on synthetic data) applied to real images:",
        *[f"  - {name}: mean softmax confidence {v['mean_confidence']:.2f} (confidently WRONG, not uncertain), "
          f"predicted composition {({k[5:]: round(val, 2) for k, val in v.items() if k.startswith('frac_')})}"
          for name, v in d.items()],
        "",
        "The model is not merely uncertain on real tissue -- it is confident and wrong, predicting",
        "large near-uniform blocks that do not correspond to any real structure. This is the expected,",
        "well known failure mode of a classifier evaluated far outside its training distribution, and is",
        "reported here rather than hidden, and is exactly why `scripts/train_tissue_unet.py` and",
        "`docs/data_preparation.md` describe training on real annotated data (e.g. PAIP 2019) before",
        "using the tissue segmentation stage on real slides.",
        "",
        "## Figures",
        "",
        *[f"![{p.stem}](figures/{p.name})" for p in sorted(Path("analysis/figures").glob("fig19*.png"))
          + sorted(Path("analysis/figures").glob("fig2[01]*.png"))],
    ]
    Path("analysis/REAL_DATA_REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
