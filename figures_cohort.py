"""Cohort level multi-omics + histology analysis figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from common import FIG_DIR, GRADE_COLORS, GRADE_NAMES, panel_label
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from liver_histo_ai.models.survival import CoxPHWrapper, DeepSurv, train_deepsurv

MODALITY_COLORS = {"histology": "#b5179e", "RNA-seq": "#2a9d8f", "mutation": "#e9c46a"}


def modality(col: str) -> str:
    return "histology" if col.startswith("histo_") else "RNA-seq" if col.startswith("rna_") else "mutation"


def pretty(col: str) -> str:
    return col.replace("histo_", "").replace("morph_", "nuc_").replace("rna_", "").replace("mut_", "")


def _xgb(seed: int) -> XGBClassifier:
    return XGBClassifier(n_estimators=160, max_depth=3, learning_rate=0.07, subsample=0.85,
                         colsample_bytree=0.7, objective="multi:softprob", num_class=3,
                         eval_metric="mlogloss", random_state=seed, n_jobs=2, verbosity=0)


def cv_eval(X: pd.DataFrame, y: np.ndarray, seed: int = 0, repeats: int = 3):
    """Repeated stratified 5 fold CV. Returns out of fold probabilities (averaged over repeats) and fold metrics."""
    oof = np.zeros((len(y), 3))
    accs, aucs, f1s = [], [], []
    for rep in range(repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(X, y):
            clf = _xgb(seed + rep).fit(X.iloc[tr], y[tr])
            p = clf.predict_proba(X.iloc[te])
            oof[te] += p / repeats
            pred = p.argmax(1)
            accs.append(accuracy_score(y[te], pred))
            aucs.append(roc_auc_score(y[te], p, multi_class="ovr"))
            f1s.append(f1_score(y[te], pred, average="macro"))
    return oof, {"acc": accs, "auc": aucs, "f1": f1s}


def fig_embedding(sets: dict[str, pd.DataFrame], y: np.ndarray) -> dict:
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    sil = {}
    for a, (name, X), letter in zip(ax, sets.items(), "abc", strict=True):
        Z = StandardScaler().fit_transform(X.values)
        emb = TSNE(2, perplexity=25, init="pca", random_state=0).fit_transform(Z)
        sil[name] = float(silhouette_score(emb, y))
        for g in range(3):
            a.scatter(*emb[y == g].T, s=22, color=GRADE_COLORS[g], alpha=0.85, label=GRADE_NAMES[g],
                      edgecolor="white", lw=0.3)
        a.set_title(f"{name}\nsilhouette = {sil[name]:.2f}"); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        panel_label(a, letter)
    ax[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("t-SNE of patients by feature modality (colour = tumor differentiation)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_embedding.png")
    plt.close(fig)
    return sil


def fig_classification(X_by: dict[str, pd.DataFrame], y: np.ndarray) -> dict:
    res, oofs = {}, {}
    for name, X in X_by.items():
        oofs[name], res[name] = cv_eval(X, y)
    fused = oofs["Histology + omics"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    a = ax[0]
    for g in range(3):
        fpr, tpr, _ = roc_curve(y == g, fused[:, g])
        a.plot(fpr, tpr, color=GRADE_COLORS[g], lw=2,
               label=f"{GRADE_NAMES[g]} (AUC {roc_auc_score(y == g, fused[:, g]):.2f})")
    a.plot([0, 1], [0, 1], "k--", lw=1)
    a.set(xlabel="false positive rate", ylabel="true positive rate", title="One vs rest ROC, fused model")
    a.legend(fontsize=7, loc="lower right")
    a = ax[1]
    cm = confusion_matrix(y, fused.argmax(1), normalize="true")
    im = a.imshow(cm, cmap="Blues", vmin=0, vmax=1); a.grid(False)
    short = ["well", "moderate", "poor"]
    a.set_xticks(range(3), short); a.set_yticks(range(3), short)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
    a.set(xlabel="predicted", ylabel="true", title="Out of fold confusion matrix"); plt.colorbar(im, ax=a, fraction=0.046)
    a = ax[2]
    w = 0.26
    for k, (metric, label) in enumerate([("acc", "accuracy"), ("auc", "macro AUC"), ("f1", "macro F1")]):
        m = [np.mean(res[n][metric]) for n in res]
        s = [np.std(res[n][metric]) for n in res]
        a.bar(np.arange(len(res)) + (k - 1) * w, m, w, yerr=s, capsize=2, label=label,
              color=["#264653", "#2a9d8f", "#e9c46a"][k])
    a.set_xticks(range(len(res)), [n.replace(" + ", "\n+ ") for n in res]); a.set_ylim(0.4, 1.02)
    a.set(title="Modality ablation (5 fold x 3 repeats)"); a.legend(fontsize=7, ncol=3, loc="upper left")
    for a_, letter in zip(ax, "abc", strict=True):
        panel_label(a_, letter)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig08_classification.png")
    plt.close(fig)
    return {n: {k: [float(np.mean(v)), float(np.std(v))] for k, v in r.items()} for n, r in res.items()}


def fig_importance(X: pd.DataFrame, y: np.ndarray, top: int = 18) -> pd.DataFrame:
    clf = _xgb(0).fit(X, y)
    gain = pd.Series(clf.get_booster().get_score(importance_type="total_gain")).reindex(X.columns).fillna(0)
    gain = (gain / gain.sum()).sort_values(ascending=False)
    top_s = gain.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh([pretty(c) for c in top_s.index], top_s.values, color=[MODALITY_COLORS[modality(c)] for c in top_s.index])
    ax.set(xlabel="share of total gain", title="Top predictive features of the fused model")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in MODALITY_COLORS.values()],
              labels=list(MODALITY_COLORS), loc="lower right")
    fig.savefig(FIG_DIR / "fig09_feature_importance.png")
    plt.close(fig)
    share = {m: float(sum(v for c, v in gain.items() if modality(c) == m)) for m in MODALITY_COLORS}
    return gain.head(top), share


def oof_risk(X: np.ndarray, dur: np.ndarray, ev: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    risk = np.zeros(len(dur))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        sc = StandardScaler().fit(X[tr])
        pca = PCA(8, random_state=0).fit(sc.transform(X[tr]))
        ztr, zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
        if kind == "cox":
            m = CoxPHWrapper(penalizer=0.5).fit(ztr, dur[tr], ev[tr])
            risk[te] = np.log(m.predict_risk(zte) + 1e-12)
        else:
            torch.manual_seed(seed)
            net = DeepSurv(8, (16,), 0.2)
            train_deepsurv(net, ztr.astype(np.float32), dur[tr].astype(np.float32), ev[tr].astype(np.float32),
                           epochs=150, lr=5e-3)
            net.eval()
            with torch.no_grad():
                risk[te] = net(torch.tensor(zte, dtype=torch.float32)).numpy()
    return risk


def fig_survival(meta: pd.DataFrame, sets: dict[str, pd.DataFrame], fused_all: pd.DataFrame) -> dict:
    dur, ev = meta["duration_months"].values, meta["event"].values
    cidx = {}
    risks = {}
    for name, X in sets.items():
        risks[name] = oof_risk(X.values, dur, ev, "cox")
        cidx[f"{name} (Cox)"] = float(concordance_index(dur, -risks[name], ev))
    ds_risk = oof_risk(fused_all.values, dur, ev, "deepsurv")
    cidx["Histology + omics (DeepSurv)"] = float(concordance_index(dur, -ds_risk, ev))
    cidx["Grade label only"] = float(concordance_index(dur, -meta["grade"].values, ev))

    risk = risks["Histology + omics"]
    q = np.quantile(risk, [1 / 3, 2 / 3])
    grp = np.digitize(risk, q)
    lr = multivariate_logrank_test(dur, grp, ev)
    fig = plt.figure(figsize=(14, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.9, 1.05], wspace=0.32)
    a = fig.add_subplot(gs[0])
    med = {}
    for g, (lab, col) in enumerate(zip(["low risk", "intermediate risk", "high risk"], GRADE_COLORS, strict=True)):
        km = KaplanMeierFitter().fit(dur[grp == g], ev[grp == g], label=f"{lab} (n={int((grp == g).sum())})")
        km.plot_survival_function(ax=a, ci_show=True, color=col, lw=2, ci_alpha=0.15)
        med[lab] = float(km.median_survival_time_)
    a.set(xlabel="months", ylabel="overall survival probability",
          title=f"Kaplan Meier by predicted risk tertile\nlog rank p = {lr.p_value:.1e}", ylim=(0, 1.02))
    a = fig.add_subplot(gs[1])
    names = list(cidx)
    vals = [cidx[n] for n in names]
    a.barh(names[::-1], vals[::-1], color=["#264653" if "omics" in n else "#9aa5b1" for n in names[::-1]])
    for i, v in enumerate(vals[::-1]):
        a.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    a.axvline(0.5, color="k", ls="--", lw=1); a.set(xlim=(0.45, 0.82), xlabel="out of fold C-index",
                                                      title="Prognostic performance")
    a.tick_params(axis="y", labelsize=7.5)

    cov = pd.DataFrame({
        "Nuclear area (mean)": fused_all["histo_morph_area_mean"],
        "Nuclear spacing (NN dist)": fused_all["histo_spatial_nn_dist_mean"],
        "Hematoxylin OD (mean)": fused_all["histo_morph_hema_mean_mean"],
        "AFP expression": fused_all["rna_AFP"],
        "MKI67 expression": fused_all["rna_MKI67"],
        "ALB expression": fused_all["rna_ALB"],
        "TP53 mutation": fused_all["mut_TP53"],
    })
    binary = {"TP53 mutation"}
    for c in cov.columns:
        if c not in binary:
            cov[c] = (cov[c] - cov[c].mean()) / (cov[c].std() + 1e-9)
    cov["duration"], cov["event"] = dur, ev
    cph = CoxPHFitter(penalizer=0.05).fit(cov, "duration", "event")
    s = cph.summary.iloc[::-1]
    a = fig.add_subplot(gs[2])
    yy = np.arange(len(s))
    hr, lo, hi = s["exp(coef)"].values, s["exp(coef) lower 95%"].values, s["exp(coef) upper 95%"].values
    a.errorbar(hr, yy, xerr=[hr - lo, hi - hr], fmt="o", color="#264653", ecolor="#264653", capsize=3)
    a.axvline(1, color="k", ls="--", lw=1)
    a.set_yticks(yy, s.index, fontsize=8); a.set_xscale("log")
    for i, p in enumerate(s["p"].values):
        a.text(1.02, i, f"p={p:.2g}", transform=a.get_yaxis_transform(), fontsize=7, va="center")
    a.set(xlabel="hazard ratio per 1 SD (95% CI)", title="Multivariable Cox model")
    for a_, letter in zip(fig.axes, "abc", strict=True):
        panel_label(a_, letter)
    fig.savefig(FIG_DIR / "fig10_survival.png")
    plt.close(fig)
    return {"cindex": cidx, "logrank_p": float(lr.p_value), "median_survival_months": med,
            "cox_hazard_ratios": {k: float(v) for k, v in s["exp(coef)"].items()}}


def fig_cohort_overview(fused: pd.DataFrame, y: np.ndarray) -> None:
    F, _ = f_classif(fused.values, y)
    top = pd.Series(F, index=fused.columns).sort_values(ascending=False).head(16).index
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.42, wspace=0.32)
    picks = [("histo_morph_area_mean", "Mean nuclear area (px$^2$)"), ("histo_spatial_density_per_1000px2", "Nuclear density"),
             ("rna_AFP", "AFP expression (log)"), ("rna_ALB", "ALB expression (log)")]
    for k, (col, title) in enumerate(picks):
        a = fig.add_subplot(gs[0, k])
        data = [fused.loc[y == g, col].values for g in range(3)]
        bp = a.boxplot(data, tick_labels=["well", "mod.", "poor"], patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], GRADE_COLORS, strict=True):
            patch.set_facecolor(c); patch.set_alpha(0.85)
        for g, d in enumerate(data):
            a.scatter(np.random.default_rng(g).normal(g + 1, 0.06, len(d)), d, s=6, color="k", alpha=0.35)
        a.set_title(title, fontsize=9); panel_label(a, "abcd"[k])
    a = fig.add_subplot(gs[1, :])
    Z = (fused[top] - fused[top].mean()) / (fused[top].std() + 1e-9)
    hm = np.array([Z.loc[y == g].mean().values for g in range(3)])
    im = a.imshow(hm, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto"); a.grid(False)
    a.set_xticks(range(len(top)), [f"{pretty(c)}\n[{modality(c)[:4]}]" for c in top], rotation=40, ha="right", fontsize=7)
    a.set_yticks(range(3), ["well", "moderate", "poor"])
    for i in range(3):
        for j in range(len(top)):
            a.text(j, i, f"{hm[i, j]:.1f}", ha="center", va="center", fontsize=7)
    a.set_title("Class mean z scores of the 16 most discriminative features (ANOVA F)")
    plt.colorbar(im, ax=a, fraction=0.02); panel_label(a, "e")
    fig.savefig(FIG_DIR / "fig11_cohort_overview.png")
    plt.close(fig)
