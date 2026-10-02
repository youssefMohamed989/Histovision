"""Test the epoch-budget hypothesis from GRAPH_MIL_LEARNING_CURVE.md directly.

That report observed GraphMIL accuracy getting WORSE as the synthetic cohort
grew from 90 to 300 patients at a fixed 30 epochs, and proposed that this is
an artifact of full batch gradient descent with a fixed epoch count (same
number of optimization steps regardless of cohort size) rather than evidence
that the architecture fails to scale. This script tests that hypothesis
directly: build the n=300 bags ONCE, then cross validate at epochs=30 (the
original setting) and more epochs on the identical bags/folds, so any
difference is attributable to training budget alone.

    python analysis/run_graph_mil_epoch_test.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from common import RES_DIR
from run_graph_mil import (
    baseline_features,
    build_bag_cohort,
    cross_validate,
    tiles_to_graphs,
)
from sklearn.metrics import accuracy_score, roc_auc_score


def main() -> None:
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    n_patients, tiles_per_patient, folds, seed = 300, 4, 3, 0
    log(f"building {n_patients} patient bags once, reused for every epoch setting")
    patients = build_bag_cohort(n_patients, tiles_per_patient, seed=seed)
    bags, tile_meta, kept_idx = tiles_to_graphs(patients, log)
    y = np.array([patients[i]["base_grade"] for i in kept_idx])
    baseline_x = baseline_features(tile_meta)

    results = {}
    for epochs in (30, 90):  # kept small to finish well within one polling window
        log(f"cross validating at epochs={epochs}")
        oof_graphmil, oof_baseline, _ = cross_validate(bags, y, baseline_x, epochs=epochs, folds=folds, seed=seed)
        results[epochs] = {
            "graphmil_acc": float(accuracy_score(y, oof_graphmil.argmax(1))),
            "graphmil_auc": float(roc_auc_score(y, oof_graphmil, multi_class="ovr")),
            "baseline_acc": float(accuracy_score(y, oof_baseline.argmax(1))),
        }
        log(f"  -> GraphMIL acc {results[epochs]['graphmil_acc']:.3f}")

    RES_DIR.mkdir(parents=True, exist_ok=True)
    with open(RES_DIR / "graph_mil_epoch_test.json", "w") as f:
        json.dump(results, f, indent=2)
    write_report(results, n_patients)
    log("done")


def write_report(results: dict, n_patients: int) -> None:
    accs = {e: r["graphmil_acc"] for e, r in results.items()}
    baseline_acc = next(iter(results.values()))["baseline_acc"]
    best_epoch = max(accs, key=accs.get)
    improved = accs[best_epoch] > accs[30] + 0.02
    confirmed = improved and accs[best_epoch] >= accs[30]
    lines = [
        "# Epoch budget test: is the learning-curve degradation a training artifact?",
        "",
        f"> Identical {n_patients} patient bags and cross validation folds at every epoch setting below;",
        "> only the training budget changes. Tests the hypothesis from `GRAPH_MIL_LEARNING_CURVE.md`.",
        "",
        "| epochs | GraphMIL accuracy | GraphMIL macro AUC |",
        "| --- | --- | --- |",
        *[f"| {e} | {r['graphmil_acc']:.3f} | {r['graphmil_auc']:.3f} |" for e, r in results.items()],
        "",
        (f"(Baseline accuracy at this cohort size, for reference: {baseline_acc:.3f}, "
         f"unaffected by GraphMIL's epoch count.)"),
        "",
    ]
    if confirmed:
        lines += [
            (f"**Hypothesis confirmed.** More epochs ({best_epoch} vs the original 30) recovered accuracy "
             f"({accs[30]:.3f} -> {accs[best_epoch]:.3f}), supporting that the n=300 result in the learning "
             f"curve was undertrained, not a genuine architecture scaling failure. The fixed-epoch comparison "
             f"there should be rerun with epochs tuned per cohort size (or minibatch SGD) for a fair scaling claim."),
        ]
    elif improved:
        lines += [
            (f"**Partially confirmed.** More epochs helped ({accs[30]:.3f} -> {accs[best_epoch]:.3f} at "
             f"{best_epoch} epochs) but did not close the gap to the baseline ({baseline_acc:.3f}). Training "
             f"budget is A factor, not the whole explanation; some combination of architecture capacity, "
             f"regularization, or the attention mechanisms themselves likely also contributes at this cohort size."),
        ]
    else:
        lines += [
            (f"**Hypothesis NOT confirmed.** More epochs ({best_epoch}) did not meaningfully recover accuracy "
             f"over the original 30 ({accs[30]:.3f} vs {accs[best_epoch]:.3f}). The epoch-budget explanation "
             f"proposed in the learning curve report does not hold up under direct test; the degradation with "
             f"cohort size has some other cause (optimization difficulty on more diverse bags, a capacity "
             f"mismatch, or an actual architectural limitation) that this repository has not yet identified."),
        ]
    Path("analysis/GRAPH_MIL_EPOCH_TEST.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
