"""Advanced histology feature analysis on the synthetic cohort.

    python analysis/run_advanced.py            # about 8 to 10 minutes on one CPU core
    python analysis/run_advanced.py --quick    # small cohort smoke run
    python analysis/run_advanced.py --reuse    # reuse cached cohort feature table

Requires the tissue U-Net checkpoint and cohort table written by
``python analysis/run_analysis.py``. All data is synthetic (see README).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import adv_figures_a as fa
import adv_figures_b as fb
import figures_tissue as ft
import numpy as np
import pandas as pd
from common import RES_DIR, setup_style

from liver_histo_ai.features.advanced import (
    AdvancedHistologyExtractor,
    feature_catalog,
)
from liver_histo_ai.features.morphology import extract_morphology_features
from liver_histo_ai.segmentation.classical import segment_nuclei_classical
from liver_histo_ai.segmentation.tissue_unet import TissueUNet
from liver_histo_ai.synthetic import generate_tile

CACHE = RES_DIR / "advanced_cohort_features.csv"


def cohort_tiles(n: int, seed: int = 11):
    """Regenerate the exact tiles of ``common.build_cohort`` (same RNG call order)."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), n // 3)
    rng.shuffle(y)
    z = y + rng.normal(0, 0.55, size=len(y))
    for zi in z:
        w = rng.dirichlet([9, 1.6, 0.4, 1.2])
        yield generate_tile(256, rng, grade=int(np.clip(round(zi), 0, 2)), weights=tuple(w), stain_jitter=0.4)


def extract_cohort(n: int, unet: TissueUNet, log) -> pd.DataFrame:
    import torch

    ex = AdvancedHistologyExtractor()
    rows = []
    for i, tile in enumerate(cohort_tiles(n)):
        rgb = tile["rgb"]
        with torch.no_grad():
            tissue = unet(torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0).argmax(1)[0].numpy()
        inst = segment_nuclei_classical(rgb)
        feats = ex.extract(rgb, inst, tissue)
        pts = extract_morphology_features(inst)[["centroid_row", "centroid_col"]].to_numpy()
        curve = fa.ripley_l_minus_r(pts, rgb.shape[:2], fa.RIPLEY_RADII)
        feats.update({f"ripleycurve_r{r}": float(v) for r, v in zip(fa.RIPLEY_RADII, curve, strict=True)})
        rows.append(feats)
        if (i + 1) % 25 == 0:
            log(f"  advanced features for {i + 1}/{n} tiles ({len(feats)} features each)")
    return pd.DataFrame(rows)


def main(quick: bool, reuse: bool) -> None:
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    setup_style()
    unet = TissueUNet(in_channels=3, num_classes=4, base_channels=12)
    unet.load_checkpoint("checkpoints/demo_tissue_unet.pt")
    unet.eval()

    base = pd.read_csv(RES_DIR / "simulated_cohort_features.csv")
    n = 60 if quick else len(base)
    base = base.iloc[:n]
    y = base["grade"].to_numpy()

    if reuse and CACHE.exists() and len(pd.read_csv(CACHE)) >= n:
        adv = pd.read_csv(CACHE).iloc[:n]
        log("reusing cached advanced features")
    else:
        log(f"extracting advanced features for {n} tiles")
        adv = extract_cohort(n, unet, log)
        if not quick:
            adv.to_csv(CACHE, index=False)
    adv = adv.iloc[: len(y)]
    # the regenerated tiles must be the same patients as the saved cohort table
    fresh_y = np.repeat(np.arange(3), 150 // 3)
    np.random.default_rng(11).shuffle(fresh_y)
    if not quick:
        assert (fresh_y == y).all(), "cohort regeneration drifted from the saved table"

    basic = base[[c for c in base.columns if c.startswith("histo_")]]
    omics = base[[c for c in base.columns if c.startswith(("rna_", "mut_"))]]
    metrics: dict = {"note": "All results are computed on synthetic data with exact ground truth."}

    log("fig13 cell graph")
    metrics["cell_graph_examples"] = fa.fig_cell_graph()
    log("fig14 point patterns")
    metrics["point_patterns"] = fa.fig_point_patterns(adv, y)
    log("fig15 nuclear shape and chromatin")
    metrics["nuclear_shape_chromatin"] = fa.fig_nuclear_shape_chromatin()
    slide = ft.make_test_slide()
    log("fig16 tumor interface")
    metrics["interface"] = fa.fig_interface(slide, unet)
    log("fig17 feature landscape and ablation")
    metrics["landscape"] = fb.fig_feature_landscape(adv.drop(columns=[c for c in adv if c.startswith("ripleycurve_")]),
                                                    basic, omics, y)
    log("fig18 habitats")
    metrics["habitats"] = fb.fig_habitats(slide, unet)

    feature_cols = [c for c in adv.columns if not c.startswith("ripleycurve_")]
    Path("docs").mkdir(exist_ok=True)
    Path("docs/FEATURE_CATALOG.md").write_text(feature_catalog(feature_cols) + "\n")
    with open(RES_DIR / "advanced_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    write_report(metrics, n)
    log("done")


def write_report(m: dict, n: int) -> None:
    ls = m["landscape"]
    lines = [
        "# Advanced feature analysis",
        "",
        f"> Synthetic data, simulated cohort of {n} patients. Validates code paths, not clinical performance.",
        "",
        f"- Advanced features per tile: {ls['n_advanced_features']} ({ls['n_after_pruning']} after |r| < 0.95 pruning)",
        "- Family sizes: " + ", ".join(f"{k} {v}" for k, v in ls["family_sizes"].items()),
        "",
        "## Single family predictive power (macro AUC, grade)",
        *[f"- {k}: {v:.3f}" for k, v in ls["family_auc"].items()],
        "",
        "## Basic vs advanced (accuracy, macro AUC)",
        *[f"- {k}: acc {v['acc'][0]:.3f} +/- {v['acc'][1]:.3f}, AUC {v['auc'][0]:.3f}" for k, v in ls["ablation"].items()],
        "",
        "## Spatial habitats vs true tissue (512 px slide)",
        (f"- ARI {m['habitats']['ari_vs_tissue']:.3f}, NMI {m['habitats']['nmi_vs_tissue']:.3f}, "
         f"Shannon {m['habitats']['habitat_shannon']:.3f}, "
         f"boundary fraction {m['habitats']['habitat_boundary_frac']:.3f}"),
        "",
        "## Figures",
        "",
        *[f"![{p.stem}](figures/{p.name})" for p in sorted(Path("analysis/figures").glob("fig1[3-8]*.png"))],
    ]
    Path("analysis/ADVANCED_REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    main(a.quick, a.reuse)
