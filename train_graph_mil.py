"""Train the cell graph GNN + gated attention MIL model on a bag of tiles per patient.

Expects a manifest CSV with columns:

    patient_id, tile_path, label

Multiple rows share a ``patient_id`` (one row per tile belonging to that
patient's bag); ``label`` must be identical within a patient and is the bag
level target (e.g. tumor grade, subtype). Tiles are read as RGB images,
nuclei are segmented with the classical (weight free) segmenter, and each
tile becomes one :class:`CellGraphData` fed to :class:`GraphMIL`.

    python scripts/train_graph_mil.py --manifest data/bags/manifest.csv \\
        --config configs/default.yaml --out checkpoints/graph_mil.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from liver_histo_ai.config import PipelineConfig
from liver_histo_ai.features.morphology import extract_morphology_features
from liver_histo_ai.models.cell_gnn import NODE_FEATURE_COLUMNS, build_cell_graph
from liver_histo_ai.models.graph_mil import GraphMIL, train_graph_mil
from liver_histo_ai.segmentation.classical import segment_nuclei_classical
from liver_histo_ai.utils.logging import get_logger
from liver_histo_ai.utils.seed import set_seed

logger = get_logger(__name__)


def build_bags(manifest: pd.DataFrame, min_tile_nuclei: int = 5):
    bags, labels, patient_ids = [], [], []
    for pid, group in manifest.groupby("patient_id"):
        tiles = []
        for _, row in group.iterrows():
            rgb = np.array(Image.open(row["tile_path"]).convert("RGB"))
            inst = segment_nuclei_classical(rgb)
            table = extract_morphology_features(inst)
            if len(table) < min_tile_nuclei:
                continue
            g = build_cell_graph(table)
            if g is not None:
                tiles.append(g)
        if len(tiles) >= 2:
            bags.append(tiles)
            labels.append(group["label"].iloc[0])
            patient_ids.append(pid)
        else:
            logger.warning(f"patient {pid}: fewer than 2 usable tiles, skipped")
    return bags, np.asarray(labels), patient_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="checkpoints/graph_mil.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--val-split", type=float, default=0.2)
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    set_seed(cfg.train.seed)

    manifest = pd.read_csv(args.manifest)
    logger.info(f"building nuclei graphs for {manifest['patient_id'].nunique()} patients")
    bags, labels, _patient_ids = build_bags(manifest)
    classes = sorted(set(labels.tolist()))
    y = np.array([classes.index(v) for v in labels])

    idx_train, idx_val = train_test_split(np.arange(len(bags)), test_size=args.val_split,
                                          random_state=cfg.train.seed, stratify=y)
    train_bags, val_bags = [bags[i] for i in idx_train], [bags[i] for i in idx_val]

    model = GraphMIL(node_in_dim=len(NODE_FEATURE_COLUMNS), num_classes=len(classes))
    logger.info(f"training on {len(train_bags)} patients, validating on {len(val_bags)}")
    history = train_graph_mil(model, train_bags, y[idx_train], val_bags, y[idx_val], epochs=args.epochs)
    logger.info(f"final train acc {history['train_acc'][-1]:.3f}, val acc {history['val_acc'][-1]:.3f}")

    import torch

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "classes": classes}, args.out)
    logger.info(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
