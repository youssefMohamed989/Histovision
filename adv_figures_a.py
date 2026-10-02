"""Advanced feature figures, part A: cell graph, point patterns, nuclear shape and chromatin, interface."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import FIG_DIR, GRADE_COLORS, GRADE_NAMES, bare, panel_label, predict_tissue
from matplotlib.collections import LineCollection, PolyCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import ConvexHull, Voronoi
from scipy.stats import kruskal

from liver_histo_ai.features.advanced import AdvancedHistologyExtractor
from liver_histo_ai.features.architecture import (
    architecture_features,
    interface_profile,
    signed_distance_to_tumor,
)
from liver_histo_ai.features.cell_graph import delaunay_edges, graph_features, ripley_l_minus_r
from liver_histo_ai.features.morphology import extract_morphology_features
from liver_histo_ai.segmentation.classical import segment_nuclei_classical
from liver_histo_ai.synthetic import REGION_COLORS, REGION_NAMES, generate_tile

RIPLEY_RADII = tuple(range(8, 68, 4))


def _faded(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb * 0.35 + 165, 0, 255).astype(np.uint8)


def fig_cell_graph() -> dict:
    fig, ax = plt.subplots(2, 3, figsize=(12.5, 8.6))
    out = {}
    for g in range(3):
        d = generate_tile(256, np.random.default_rng(100 + g), grade=g, weights=(1, 0, 0, 0))
        inst = segment_nuclei_classical(d["rgb"])
        table = extract_morphology_features(inst)
        pts = table[["centroid_row", "centroid_col"]].to_numpy()
        edges = delaunay_edges(pts)
        length = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
        keep = length <= 3 * np.median(length)
        segs = [[(pts[i, 1], pts[i, 0]), (pts[j, 1], pts[j, 0])] for i, j in edges[keep]]
        a = ax[0, g]
        a.imshow(_faded(d["rgb"]))
        lc = LineCollection(segs, array=length[keep], cmap="viridis", linewidths=1.1, clim=(6, 30))
        a.add_collection(lc)
        n = len(pts)
        mst = minimum_spanning_tree(coo_matrix((length[keep], (edges[keep, 0], edges[keep, 1])), shape=(n, n)).tocsr())
        mi, mj = mst.nonzero()
        a.add_collection(LineCollection([[(pts[i, 1], pts[i, 0]), (pts[j, 1], pts[j, 0])] for i, j in zip(mi, mj, strict=True)],
                                        colors="#e63946", linewidths=1.6, alpha=0.9))
        a.scatter(pts[:, 1], pts[:, 0], s=5, color="k", zorder=3)
        bare(a)
        a.set_title(GRADE_NAMES[g], color=GRADE_COLORS[g])

        gf = graph_features(table, d["rgb"].shape[:2])
        out[GRADE_NAMES[g]] = {k: gf[k] for k in ("graph_delaunay_edge_cv", "graph_delaunay_regularity_mean",
                                                    "graph_voronoi_area_cv", "graph_clark_evans_R",
                                                    "graph_mst_branch_frac", "graph_moran_area")}
        a.text(0.02, 0.02, f"edge CV {gf['graph_delaunay_edge_cv']:.2f}  regularity {gf['graph_delaunay_regularity_mean']:.2f}\n"
                           f"MST branch {gf['graph_mst_branch_frac']:.2f}", transform=a.transAxes, fontsize=8,
               color="white", backgroundcolor="#000000aa", va="bottom")

        b = ax[1, g]
        b.imshow(_faded(d["rgb"]))
        vor = Voronoi(pts)
        polys, areas = [], []
        for i in range(len(pts)):
            reg = vor.regions[vor.point_region[i]]
            if not reg or -1 in reg:
                continue
            v = vor.vertices[reg]
            if v.min() < 0 or v.max() > 256:
                continue
            hull = ConvexHull(v)
            polys.append(v[hull.vertices][:, ::-1])
            areas.append(hull.volume)
        pc = PolyCollection(polys, array=np.asarray(areas), cmap="magma_r", edgecolors="white", linewidths=0.4,
                            alpha=0.85, clim=(40, 400))
        b.add_collection(pc)
        b.scatter(pts[:, 1], pts[:, 0], s=3, color="k")
        bare(b)
        b.text(0.02, 0.02, f"Voronoi area CV {gf['graph_voronoi_area_cv']:.2f}   Clark-Evans R {gf['graph_clark_evans_R']:.2f}",
               transform=b.transAxes, fontsize=8, color="white", backgroundcolor="#000000aa", va="bottom")
    fig.colorbar(lc, ax=ax[0, :], fraction=0.018, pad=0.01, label="Delaunay edge length (px)")
    fig.colorbar(pc, ax=ax[1, :], fraction=0.018, pad=0.01, label="Voronoi cell area (px$^2$)")
    fig.suptitle("Cell graph topology: Delaunay edges (top, MST in red) and Voronoi cells (bottom)", fontweight="bold")
    for a_, letter in zip(ax.ravel(), "abcdef", strict=True):
        panel_label(a_, letter)
    fig.savefig(FIG_DIR / "fig13_cell_graph.png")
    plt.close(fig)
    return out


def fig_point_patterns(adv: pd.DataFrame, y: np.ndarray) -> dict:
    curve_cols = [f"ripleycurve_r{r}" for r in RIPLEY_RADII]
    n_typ = int(adv["nuc_count"].median())
    rng = np.random.default_rng(0)
    sims = np.array([ripley_l_minus_r(rng.uniform(0, 256, (n_typ, 2)), (256, 256), RIPLEY_RADII) for _ in range(99)])
    lo, hi = np.nanpercentile(sims, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
    a = ax[0, 0]
    a.fill_between(RIPLEY_RADII, lo, hi, color="#adb5bd", alpha=0.5, label=f"CSR envelope (99 sims, n={n_typ})")
    a.axhline(0, color="k", lw=0.8)
    for g in range(3):
        m = adv.loc[y == g, curve_cols].to_numpy(float)
        mu, se = np.nanmean(m, 0), np.nanstd(m, 0) / np.sqrt(len(m))
        a.plot(RIPLEY_RADII, mu, color=GRADE_COLORS[g], lw=2, label=GRADE_NAMES[g])
        a.fill_between(RIPLEY_RADII, mu - 1.96 * se, mu + 1.96 * se, color=GRADE_COLORS[g], alpha=0.25)
    a.set(xlabel="radius r (px)", ylabel="L(r) - r", title="Ripley's L function (below 0 = regular spacing)")
    a.text(0.02, 0.03, "dip near r=8-12 is real: nuclei have a hard-core exclusion radius\n"
           "(they cannot overlap), which suppresses close neighbour pairs", transform=a.transAxes,
           fontsize=7, style="italic", va="bottom")
    a.legend(fontsize=7, loc="lower left")

    stats = {}
    for a, col, title, letter in [(ax[0, 1], "graph_clark_evans_R", "Clark-Evans index R", "b"),
                                  (ax[1, 0], "graph_voronoi_area_cv", "Voronoi cell area CV (disorder)", "c"),
                                  (ax[1, 1], "graph_delaunay_regularity_mean", "Delaunay triangle regularity", "d")]:
        data = [adv.loc[y == g, col].dropna().to_numpy() for g in range(3)]
        bp = a.boxplot(data, tick_labels=["well", "moderate", "poor"], patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], GRADE_COLORS, strict=True):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        for g, v in enumerate(data):
            a.scatter(np.random.default_rng(g).normal(g + 1, 0.06, len(v)), v, s=6, color="k", alpha=0.3)
        p = kruskal(*data).pvalue
        a.set_title(f"{title}\nKruskal-Wallis p = {p:.1e}")
        stats[col] = {"kruskal_p": float(p), "medians": [float(np.median(v)) for v in data]}
    for a_, letter in zip(ax.ravel(), "abcd", strict=True):
        panel_label(a_, letter)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig14_point_patterns.png")
    plt.close(fig)
    return stats


def _nuclear_tables(n_tiles: int = 5):
    ex = AdvancedHistologyExtractor()
    tables, tiles = [], {}
    for g in range(3):
        for k in range(n_tiles):
            d = generate_tile(256, np.random.default_rng(500 + 10 * g + k), grade=g, weights=(1, 0, 0, 0))
            inst = segment_nuclei_classical(d["rgb"])
            t = ex.nuclear_table(d["rgb"], inst)
            t["grade"], t["tile_g"], t["tile_k"] = g, g, k
            tiles[(g, k)] = (d["rgb"], inst)
            tables.append(t)
    return pd.concat(tables, ignore_index=True), tiles


def fig_nuclear_shape_chromatin() -> dict:
    df, tiles = _nuclear_tables()
    feature_cols = [c for c in df.columns if c.startswith(("shp_", "chr_")) or c in
                    ("area", "eccentricity", "solidity", "circularity", "axis_ratio")]
    fig = plt.figure(figsize=(16.5, 8.6))
    outer = fig.add_gridspec(2, 4, wspace=0.32, hspace=0.34)
    inner = outer[:, 0:2].subgridspec(3, 4, wspace=0.08, hspace=0.32)

    for g in range(3):
        sub = df[(df.grade == g) & df["shp_radial_cv"].notna()
                 & df["centroid_row"].between(20, 236) & df["centroid_col"].between(20, 236)]
        qs = np.quantile(sub["shp_radial_cv"], [0.1, 0.4, 0.7, 0.95])
        for j, q in enumerate(qs):
            row = sub.iloc[(sub["shp_radial_cv"] - q).abs().argmin()]
            rgb, inst = tiles[(int(row["tile_g"]), int(row["tile_k"]))]
            r0, c0 = int(row["centroid_row"]), int(row["centroid_col"])
            sl = (slice(r0 - 16, r0 + 16), slice(c0 - 16, c0 + 16))
            a = fig.add_subplot(inner[g, j])
            a.imshow(rgb[sl], interpolation="bicubic")
            a.contour(inst[sl] == int(row["instance_id"]), [0.5], colors="#ffd60a", linewidths=1.4)
            bare(a)
            a.set_title(f"rough {row['shp_roughness']:.2f}  rCV {row['shp_radial_cv']:.2f}\nH ent {row['chr_h_entropy']:.1f}  clumps {int(row['chr_clump_count'])}",
                        fontsize=6.5, color=GRADE_COLORS[g])
            if j == 0:
                a.set_ylabel(GRADE_NAMES[g].split()[0], color=GRADE_COLORS[g], fontsize=9, fontweight="bold")
    panel_label(fig.axes[0], "a")

    a = fig.add_subplot(outer[0, 2])
    ks = range(2, 7)
    for g in range(3):
        m = np.array([df.loc[df.grade == g, f"shp_fd_harmonic{k}"].dropna().mean() for k in ks])
        s = np.array([df.loc[df.grade == g, f"shp_fd_harmonic{k}"].dropna().sem() for k in ks])
        a.errorbar(list(ks), m, yerr=1.96 * s, color=GRADE_COLORS[g], marker="o", lw=2, capsize=3, label=GRADE_NAMES[g].split()[0])
    a.set(xlabel="Fourier harmonic k", ylabel="normalized amplitude", title="Contour Fourier spectrum by grade")
    a.legend(fontsize=7)
    panel_label(a, "b")

    a = fig.add_subplot(outer[0, 3])
    data = [df.loc[df.grade == g, "shp_radial_cv"].dropna().to_numpy() for g in range(3)]
    parts = a.violinplot(data, showmedians=True, showextrema=False)
    for body, c in zip(parts["bodies"], GRADE_COLORS, strict=True):
        body.set_facecolor(c)
        body.set_alpha(0.8)
    parts["cmedians"].set_color("k")
    a.set_xticks([1, 2, 3], ["well", "moderate", "poor"])
    a.set(title=f"Radial distance variability\nKruskal-Wallis p = {kruskal(*data).pvalue:.1e}", ylabel="radial CV")
    panel_label(a, "c")

    a = fig.add_subplot(outer[1, 2])
    data = [df.loc[df.grade == g, "chr_glcm_entropy"].dropna().to_numpy() for g in range(3)]
    bp = a.boxplot(data, tick_labels=["well", "moderate", "poor"], patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], GRADE_COLORS, strict=True):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    a.set(title=f"Intranuclear chromatin texture entropy\nKruskal-Wallis p = {kruskal(*data).pvalue:.1e}", ylabel="masked GLCM entropy")
    panel_label(a, "d")

    a = fig.add_subplot(outer[1, 3])
    lo, hi = df[df.grade == 0], df[df.grade == 2]
    d_eff = {}
    for c in feature_cols:
        x, z = lo[c].dropna(), hi[c].dropna()
        if len(x) > 20 and len(z) > 20:
            pooled = np.sqrt((x.var() + z.var()) / 2) + 1e-12
            d_eff[c] = (z.mean() - x.mean()) / pooled
    top = pd.Series(d_eff).reindex(pd.Series(d_eff).abs().sort_values(ascending=False).index).head(14)[::-1]
    a.barh([n.replace("shp_", "shape ").replace("chr_", "chrom ") for n in top.index], top.values,
           color=["#2a9d8f" if n.startswith("shp_") else "#b5179e" if n.startswith("chr_") else "#264653" for n in top.index])
    a.axvline(0, color="k", lw=0.8)
    a.set(xlabel="Cohen's d (poor vs well)", title="Most discriminative nuclear features")
    a.tick_params(axis="y", labelsize=7)
    panel_label(a, "e")
    fig.suptitle("Nuclear shape and chromatin descriptors across tumor grade (yellow = nucleus contour)", fontweight="bold")
    fig.savefig(FIG_DIR / "fig15_nuclear_shape_chromatin.png")
    plt.close(fig)
    return {"n_nuclei": len(df), "top_cohens_d": {k: float(v) for k, v in top[::-1].items()}}


def fig_interface(slide: dict, unet) -> dict:
    rgb, gt_tissue = slide["rgb"], slide["tissue"]
    pred = predict_tissue(unet, rgb)
    inst = segment_nuclei_classical(rgb)
    table = extract_morphology_features(inst)
    feats_pred = architecture_features(pred, table)
    signed = signed_distance_to_tumor(pred)

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    a = ax[0, 0]
    a.imshow(rgb)
    lo, hi = float(signed.min()) - 1, float(signed.max()) + 1
    cs = a.contourf(signed, levels=[lo, -40, -20, 0, 20, 40, 80, hi], cmap="RdYlBu_r", alpha=0.32)
    a.contour(signed, [0], colors="k", linewidths=1.6)
    r = np.clip(table["centroid_row"].to_numpy().round().astype(int), 0, 511)
    c = np.clip(table["centroid_col"].to_numpy().round().astype(int), 0, 511)
    a.scatter(table["centroid_col"], table["centroid_row"], s=3, c=signed[r, c], cmap="RdYlBu_r", vmin=-80, vmax=80,
              edgecolors="none")
    bare(a)
    a.set_title("Signed distance to tumor boundary (black = boundary)")
    plt.colorbar(cs, ax=a, fraction=0.04, label="px (negative = inside tumor)")

    a = ax[0, 1]
    for tissue_map, label, color in [(gt_tissue, "ground truth tissue map", "#264653"), (pred, "U-Net predicted map", "#e76f51")]:
        centers, density, _counts, _area = interface_profile(tissue_map, table)
        a.plot(centers, density, marker="o", lw=2, color=color, label=label)
    a.axvline(0, color="k", ls="--", lw=1)
    a.set(xlabel="signed distance from tumor boundary (px)", ylabel="nuclear density (per 1000 px$^2$)",
          title="Nuclear density across the tumor boundary")
    a.legend(fontsize=8)

    a = ax[1, 0]
    keys = [("arch_frac_tumor", "tumor"), ("arch_frac_stroma", "stroma"), ("arch_frac_necrosis", "necrosis"),
            ("arch_frac_normal", "normal"), ("arch_interface_frac_stroma", "rim: stroma"),
            ("arch_interface_frac_necrosis", "rim: necrosis"), ("arch_interface_frac_normal", "rim: normal")]
    vals = [feats_pred.get(k, 0.0) for k, _ in keys]
    cols = [REGION_COLORS[n] for n in ("tumor", "stroma", "necrosis", "normal")] + [REGION_COLORS[n] for n in ("stroma", "necrosis", "normal")]
    a.barh([n for _, n in keys][::-1], vals[::-1], color=cols[::-1])
    for i, v in enumerate(vals[::-1]):
        a.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    a.set(xlim=(0, 1.1), title=f"Composition and what borders the tumor\n{int(feats_pred.get('arch_n_tumor_nests', 0))} tumor nest(s), largest = {feats_pred.get('arch_nest_largest_frac', 0):.0%} of tumor")

    a = ax[1, 1]
    names = [n for n in REGION_NAMES if f"arch_nuc_density_{n}" in feats_pred]
    dens = [feats_pred[f"arch_nuc_density_{n}"] for n in names]
    area = [feats_pred.get(f"arch_nuc_area_{n}", np.nan) for n in names]
    x = np.arange(len(names))
    a.bar(x - 0.2, dens, 0.4, color=[REGION_COLORS[n] for n in names], label="density (per 1000 px$^2$)")
    a.set_xticks(x, names)
    a.set_ylabel("nuclear density")
    a2 = a.twinx()
    a2.bar(x + 0.2, area, 0.4, color=[REGION_COLORS[n] for n in names], alpha=0.45, hatch="//", label="mean area")
    a2.set_ylabel("mean nuclear area (px$^2$), hatched")
    a2.grid(False)
    a.set_title("Region specific nuclear density and size")
    for a_, letter in zip(ax.ravel(), "abcd", strict=True):
        panel_label(a_, letter)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig16_tumor_interface.png")
    plt.close(fig)
    return {k: float(v) for k, v in feats_pred.items() if np.isfinite(v)}
