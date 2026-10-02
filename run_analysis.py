"""Run the full demonstration analysis and write figures, tables and a report.

    python analysis/run_analysis.py            # ~5 to 10 minutes on CPU
    python analysis/run_analysis.py --quick    # smaller cohort for a fast check

All data here is SYNTHETIC (see src/liver_histo_ai/synthetic.py and
analysis/common.py). The point is to exercise and validate the real
pipeline code paths end to end with exact ground truth. Numbers are not
clinical results; rerun the same modules on TCGA-LIHC slides for that.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import figures_cohort as fc
import figures_tissue as ft
import numpy as np
from common import (
    RES_DIR,
    build_cohort,
    predict_tissue,
    setup_style,
    train_tissue_unet,
)

from liver_histo_ai.config import ClassifierConfig
from liver_histo_ai.models.classifier import TabularClassifier
from liver_histo_ai.segmentation.classical import segment_nuclei_classical


def main(quick: bool) -> None:
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    setup_style()
    metrics: dict = {"note": "All results are computed on synthetic data with exact ground truth."}

    log("training tissue U-Net on synthetic slides")
    unet, hist = train_tissue_unet(epochs=6 if quick else 16, log=log)
    unet.save_checkpoint("checkpoints/demo_tissue_unet.pt", history=hist)
    ft.fig_training(hist)

    log("evaluating on held out 512 px slide")
    slide = ft.make_test_slide()
    tissue_pred = predict_tissue(unet, slide["rgb"])
    metrics["tissue_segmentation"] = ft.fig_tissue_segmentation(slide, tissue_pred)
    inst = segment_nuclei_classical(slide["rgb"])
    metrics["nuclei_segmentation"] = ft.fig_nuclei(slide, inst)
    log("nuclear morphology, texture and spatial analysis")
    nuc = ft.nuclei_table(inst, tissue_pred, slide["rgb"])
    metrics["morphology_medians"] = ft.fig_morphology(nuc)
    metrics["texture_spatial"] = ft.fig_texture_spatial(slide, tissue_pred, inst)
    log("stain normalization experiment")
    metrics["stain_normalization"] = ft.fig_stain_norm()
    log("grade gallery")
    metrics["grade_gallery"] = ft.fig_grade_gallery(unet)

    n = 60 if quick else 150
    log(f"building simulated cohort of {n} patients (histology features via real pipeline code)")
    meta, histo, rna, mut = build_cohort(n, unet, log=log)
    omics = rna.join(mut)
    fused = histo.join(omics)
    y = meta["grade"].values
    meta.join(histo).join(omics).to_csv(RES_DIR / "simulated_cohort_features.csv", index=False)

    log("embeddings")
    metrics["tsne_silhouette"] = fc.fig_embedding(
        {"Histology features": histo, "Omics features": omics, "Fused features": fused}, y)
    log("classification with modality ablation")
    metrics["classification"] = fc.fig_classification(
        {"Histology only": histo, "Omics only": omics, "Histology + omics": fused}, y)
    log("feature importance")
    gain, share = fc.fig_importance(fused, y)
    metrics["importance_share_by_modality"] = share
    metrics["top_features"] = {k: float(v) for k, v in gain.head(10).items()}
    log("survival analysis")
    metrics["survival"] = fc.fig_survival(
        meta, {"Histology only": histo, "Omics only": omics, "Histology + omics": fused}, fused)
    fc.fig_cohort_overview(fused, y)

    log("saving reusable classifier through the package API")
    clf = TabularClassifier(ClassifierConfig(model_type="xgboost", num_classes=3), in_dim=fused.shape[1])
    metrics["packaged_classifier_holdout"] = clf.fit(fused.values.astype(np.float32), y)
    clf.save("checkpoints/demo_classifier.joblib")

    with open(RES_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    write_report(metrics, n)
    log("done")


def write_report(m: dict, n: int) -> None:
    t, nu, c, s = m["tissue_segmentation"], m["nuclei_segmentation"], m["classification"], m["survival"]
    lines = [
        "# Demonstration analysis results",
        "",
        (f"> All numbers below come from **synthetic** liver histology and a **simulated** multi-omics cohort "
         f"(n = {n}) with exact ground truth. They validate that the pipeline code works end to end. "
         f"They are not clinical findings."),
        "",
        "## Tissue segmentation (U-Net, held out 512 px slide)",
        f"- Pixel accuracy {t['pixel_accuracy']:.3f}, mean Dice {t['mean_dice']:.3f}",
        "- Dice per class: " + ", ".join(f"{k} {v:.2f}" for k, v in t["dice"].items()),
        "",
        "## Nuclei instance segmentation (color deconvolution + watershed)",
        (f"- Precision {nu['precision']:.3f}, recall {nu['recall']:.3f}, F1 {nu['f1']:.3f}, AJI {nu['aji']:.3f} "
         f"({nu['n_pred']} predicted vs {nu['n_gt']} true nuclei)"),
        "- Recall by region: " + ", ".join(f"{k} {v:.2f}" for k, v in nu["recall_by_region"].items()),
        "",
        "## Stain normalization",
    ]
    for k, v in m["stain_normalization"].items():
        lines.append(f"- {k}: between tile colour std {v['between_tile_color_std']:.1f}, nuclei F1 {v['mean_nuclei_f1']:.3f}")
    lines += ["", "## Grade classification (5 fold x 3 repeat CV, XGBoost)"]
    for k, v in c.items():
        lines.append(f"- {k}: accuracy {v['acc'][0]:.3f} +/- {v['acc'][1]:.3f}, "
                     f"macro AUC {v['auc'][0]:.3f}, macro F1 {v['f1'][0]:.3f}")
    lines += ["", "## Survival (out of fold C-index)"]
    for k, v in s["cindex"].items():
        lines.append(f"- {k}: {v:.3f}")
    lines.append(f"- Log rank p across predicted risk tertiles: {s['logrank_p']:.2e}")
    lines += ["", "## Figures", ""]
    for p in sorted(Path("analysis/figures").glob("*.png")):
        lines.append(f"![{p.stem}](figures/{p.name})")
    Path("analysis/REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
