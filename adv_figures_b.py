"""Advanced feature figures, part B: feature landscape, ablation and spatial habitats."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import FIG_DIR, bare, panel_label, predict_tissue
from figures_cohort import _xgb, cv_eval
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import spearmanr
from sklearn.feature_selection import f_classif
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from liver_histo_ai.features.advanced import GROUPS, feature_group
from liver_histo_ai.features.habitats import habitat_clustering, spatial_habitat_metrics
from liver_histo_ai.features.selection import prune_correlated, stability_selection
from liver_histo_ai.features.texture import extract_glcm_features
from liver_histo_ai.features.texture_advanced import colour_features
from liver_histo_ai.synthetic import REGION_COLORS, REGION_NAMES

FAMILY_COLORS = {
    "nuclear morphology": "#264653", "nuclear shape": "#2a9d8f", "chromatin": "#b5179e",
    "cell graph": "#e76f51", "multiscale texture": "#e9c46a", "tissue architecture": "#457b9d",
    "RNA-seq": "#8d99ae", "mutation": "#bcbd22",
}
SHORT = {"nuclear morphology": "morphology", "nuclear shape": "shape", "chromatin": "chromatin",
         "cell graph": "cell graph", "multiscale texture": "texture", "tissue architecture": "architecture"}


def family_of(col: str) -> str:
    if col.startswith("rna_"):
        return "RNA-seq"
    if col.startswith("mut_"):
        return "mutation"
    return feature_group(col)


def fig_feature_landscape(adv: pd.DataFrame, basic: pd.DataFrame, omics: pd.DataFrame, y: np.ndarray) -> dict:
    adv_cols = [c for c in adv.columns if not c.startswith("ripleycurve_") and c not in ("patient_id", "grade")]
    adv_x = adv[adv_cols].fillna(adv[adv_cols].median())
    families = {f: [c for c in adv_cols if feature_group(c) == f] for f, _ in GROUPS}

    # single family ablation
    fam_auc = {}
    for f, cols in families.items():
        if len(cols) >= 3:
            _, r = cv_eval(adv_x[cols], y, repeats=2)
            fam_auc[f] = (float(np.mean(r["auc"])), float(np.std(r["auc"])), len(cols))

    pruned = prune_correlated(adv_x, 0.95)
    sets = {
        "Basic histology": basic,
        "Advanced histology": adv_x,
        "Advanced pruned\n(|r| < 0.95)": adv_x[pruned],
        "Basic histology\n+ omics": basic.join(omics),
        "Advanced histology\n+ omics": adv_x.join(omics),
    }
    ablation = {}
    for name, x in sets.items():
        _, r = cv_eval(x, y, repeats=3)
        ablation[name.replace("\n", " ")] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in r.items()}

    fused = adv_x.join(omics)
    clf = _xgb(0).fit(fused, y)
    gain = pd.Series(clf.get_booster().get_score(importance_type="total_gain")).reindex(fused.columns).fillna(0)
    gain /= gain.sum()
    share = gain.groupby([family_of(c) for c in gain.index]).sum().sort_values()

    stab = stability_selection(adv_x, y, n_boot=80, top_k=15).head(15)[::-1]

    top_per_family = []
    F = pd.Series(np.nan_to_num(f_classif(adv_x.to_numpy(), y)[0]), index=adv_cols)
    for f, cols in families.items():
        top_per_family += list(F[cols].sort_values(ascending=False).head(7).index)
    corr = pd.DataFrame(spearmanr(adv_x[top_per_family].to_numpy())[0], index=top_per_family, columns=top_per_family)
    order = leaves_list(linkage(1 - corr.abs().to_numpy()[np.triu_indices(len(corr), 1)], "average"))
    corr = corr.iloc[order, order]

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.15], hspace=0.5, wspace=1.4)
    a = fig.add_subplot(gs[0, 0:2])
    names = list(fam_auc)
    vals = [fam_auc[n][0] for n in names]
    left = min(0.4, min(vals) - 0.08)
    widths = [v - left for v in vals]
    a.barh([SHORT[n] for n in names][::-1], widths[::-1], xerr=[fam_auc[n][1] for n in names][::-1],
           color=[FAMILY_COLORS[n] for n in names][::-1], capsize=2, left=left, height=0.6)
    for i, (n, v) in enumerate(zip(names[::-1], vals[::-1], strict=True)):
        a.text(v + 0.01, i, f"n={fam_auc[n][2]}", va="center", fontsize=7)
    a.axvline(0.5, color="k", ls="--", lw=1, label="chance")
    a.set(xlim=(left, 1.0), xlabel="macro AUC (grade), one family alone", title="Predictive power per feature family")
    a.legend(fontsize=7, loc="lower right")
    panel_label(a, "a")

    a = fig.add_subplot(gs[0, 2:4])
    w = 0.38
    keys = list(ablation)
    a.bar(np.arange(len(keys)) - w / 2, [ablation[k]["acc"][0] for k in keys], w, yerr=[ablation[k]["acc"][1] for k in keys],
          capsize=2, color="#264653", label="accuracy")
    a.bar(np.arange(len(keys)) + w / 2, [ablation[k]["auc"][0] for k in keys], w, yerr=[ablation[k]["auc"][1] for k in keys],
          capsize=2, color="#2a9d8f", label="macro AUC")
    a.set_xticks(range(len(keys)), [k.replace(" ", "\n", 1) if "\n" not in k else k for k in sets], fontsize=7)
    a.set(ylim=(0.4, 1.02), title="Basic vs advanced features (5 fold x 3 repeats)")
    a.legend(fontsize=7, ncol=2, loc="upper left")
    panel_label(a, "b")

    a = fig.add_subplot(gs[0, 4:6])
    a.barh([SHORT.get(n, n) for n in share.index], share.values, color=[FAMILY_COLORS[n] for n in share.index])
    a.set(xlabel="share of total gain", title="Importance by family (advanced histology + omics)")
    panel_label(a, "c")

    a = fig.add_subplot(gs[1, 0:2])
    a.barh([c.replace("nuc_", "").replace("graph_", "g:").replace("tex_", "t:").replace("arch_", "a:") for c in stab.index],
           stab.values, color=[FAMILY_COLORS[family_of(c)] for c in stab.index])
    a.set(xlabel="selection frequency (80 subsamples)", title="Stable features (ANOVA top 15)")
    a.tick_params(axis="y", labelsize=6.5)
    panel_label(a, "d")

    a = fig.add_subplot(gs[1, 2:6])
    im = a.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    a.grid(False)
    a.set_xticks([])
    a.set_yticks(range(len(corr)), [c.replace("nuc_", "").replace("graph_", "").replace("tex_", "").replace("arch_", "")[:26]
                                    for c in corr.index], fontsize=5.5)
    for i, c in enumerate(corr.index):
        a.add_patch(plt.Rectangle((-1.6, i - 0.5), 1.0, 1.0, color=FAMILY_COLORS[family_of(c)], clip_on=False))
    a.set_title("Spearman correlation of the 7 most discriminative features per family (clustered; colour bar = family)")
    plt.colorbar(im, ax=a, fraction=0.025)
    panel_label(a, "e")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in FAMILY_COLORS if f not in ("RNA-seq", "mutation")]
    fig.legend(handles, [SHORT[f] for f in FAMILY_COLORS if f not in ("RNA-seq", "mutation")], loc="lower left",
               ncol=6, fontsize=8, bbox_to_anchor=(0.09, 0.03))
    fig.savefig(FIG_DIR / "fig17_feature_landscape.png")
    plt.close(fig)
    return {"n_advanced_features": len(adv_cols), "n_after_pruning": len(pruned),
            "family_auc": {k: v[0] for k, v in fam_auc.items()}, "family_sizes": {k: len(v) for k, v in families.items()},
            "ablation": ablation, "importance_share": {k: float(v) for k, v in share.items()},
            "top_stable": list(stab.index[::-1][:8])}


def fig_habitats(slide: dict, unet, win: int = 64, step: int = 32) -> dict:
    rgb, gt = slide["rgb"], slide["tissue"]
    from scipy import ndimage as ndi

    from liver_histo_ai.features.morphology import extract_morphology_features
    from liver_histo_ai.segmentation.classical import segment_nuclei_classical

    inst = segment_nuclei_classical(rgb)
    table = extract_morphology_features(inst)
    n = rgb.shape[0] // step
    rows, gt_cell = [], np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            r0, r1 = max(i * step - (win - step) // 2, 0), min(i * step + step + (win - step) // 2, rgb.shape[0])
            c0, c1 = max(j * step - (win - step) // 2, 0), min(j * step + step + (win - step) // 2, rgb.shape[1])
            patch = rgb[r0:r1, c0:c1]
            inside = table[table.centroid_row.between(r0, r1) & table.centroid_col.between(c0, c1)]
            cf, gl = colour_features(patch), extract_glcm_features(patch, distances=(1,), angles_deg=(0, 90))
            rows.append({
                "nuc_area_mean": inside["area"].mean() if len(inside) else 0.0,
                "nuc_count": float(len(inside)),
                "nuc_area_cv": inside["area"].std() / (inside["area"].mean() + 1e-9) if len(inside) > 2 else 0.0,
                "h_od_mean": cf["h_od_mean"], "h_positive_area": cf["h_positive_area"], "e_od_mean": cf["e_od_mean"],
                "intensity_entropy": cf["intensity_entropy"], "edge_density": cf["edge_density"],
                "glcm_contrast": gl["glcm_contrast_d1"], "glcm_homogeneity": gl["glcm_homogeneity_d1"],
            })
            gt_cell[i, j] = np.bincount(gt[i * step:(i + 1) * step, j * step:(j + 1) * step].ravel(), minlength=4).argmax()
    feats = pd.DataFrame(rows)
    labels, _model = habitat_clustering(feats, n_habitats=4, n_components=5, order_by="h_od_mean")
    grid = labels.reshape(n, n)
    ari = float(adjusted_rand_score(gt_cell.ravel(), labels))
    nmi = float(normalized_mutual_info_score(gt_cell.ravel(), labels))
    spatial = spatial_habitat_metrics(grid, 4)
    z = (feats - feats.mean()) / (feats.std() + 1e-9)
    signature = np.array([z[labels == k].mean().to_numpy() for k in range(4)])
    cont = np.array([[np.sum((labels == k) & (gt_cell.ravel() == t)) for t in range(4)] for k in range(4)], dtype=float)
    cont_n = cont / np.maximum(cont.sum(axis=1, keepdims=True), 1)

    hab_cmap = ListedColormap(["#264653", "#2a9d8f", "#e9c46a", "#e76f51"])
    tis_cmap = ListedColormap([REGION_COLORS[k] for k in REGION_NAMES])
    fig, ax = plt.subplots(2, 3, figsize=(14, 9))
    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title("Slide (512 x 512 px)")
    ax[0, 1].imshow(rgb)
    ax[0, 1].imshow(np.kron(grid, np.ones((step, step))), cmap=hab_cmap, vmin=0, vmax=3, alpha=0.6, interpolation="nearest")
    ax[0, 1].set_title(f"Unsupervised habitats (k=4, {win}px windows)")
    ax[0, 2].imshow(np.kron(gt_cell, np.ones((step, step))), cmap=tis_cmap, vmin=0, vmax=3, interpolation="nearest")
    ax[0, 2].set_title("Ground truth tissue (majority per cell)")
    for a in ax[0]:
        bare(a)
    a = ax[1, 0]
    im = a.imshow(signature.T, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    a.grid(False)
    a.set_xticks(range(4), [f"H{k}" for k in range(4)])
    a.set_yticks(range(feats.shape[1]), feats.columns, fontsize=7)
    plt.colorbar(im, ax=a, fraction=0.05, label="z score")
    a.set_title("Habitat feature signatures")
    a = ax[1, 1]
    im = a.imshow(cont_n, cmap="Blues", vmin=0, vmax=1)
    a.grid(False)
    a.set_xticks(range(4), REGION_NAMES, rotation=25)
    a.set_yticks(range(4), [f"H{k}" for k in range(4)])
    for i in range(4):
        for j in range(4):
            a.text(j, i, f"{cont_n[i, j]:.2f}", ha="center", va="center", color="white" if cont_n[i, j] > 0.5 else "black")
    a.set_title(f"Habitat vs true tissue  (ARI {ari:.2f}, NMI {nmi:.2f})")
    a = ax[1, 2]
    bars = {"Shannon\ndiversity": spatial["habitat_shannon"], "boundary\nfraction": spatial["habitat_boundary_frac"],
            "largest patch\nfraction": spatial["habitat_largest_patch_frac"],
            "patches per\nhabitat / 10": spatial["habitat_patches_per_class"] / 10}
    a.bar(list(bars), list(bars.values()), color="#264653")
    for i, v in enumerate(bars.values()):
        a.text(i, v + 0.01, f"{v:.2f}", ha="center")
    a.set(ylim=(0, 1.05), title="Spatial organization of habitats")
    for a_, letter in zip(ax.ravel(), "abcdef", strict=True):
        panel_label(a_, letter)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig18_habitats.png")
    plt.close(fig)
    _ = (ndi, predict_tissue, unet)
    return {"ari_vs_tissue": ari, "nmi_vs_tissue": nmi, **{k: float(v) for k, v in spatial.items()}}
