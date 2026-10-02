"""Train the fused histology+omics classifier.

Expects a feature table CSV with one row per patient:

    patient_id, <histology feature columns...>, <omics_* feature columns...>, label

Omics columns must be named with the ``--omics-prefix`` (default ``omics_``) and come after
the histology columns. The histology column names, their order and the omics width are saved
next to the checkpoint (``<checkpoint>.features.json``) so prediction rebuilds the same layout.

The label column name and the split between histology/omics columns are
configurable via CLI flags so this works whether features came from
``pipeline.run_pipeline`` output or a custom extraction run.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from liver_histo_ai.config import PipelineConfig
from liver_histo_ai.models.classifier import TabularClassifier, save_feature_manifest
from liver_histo_ai.utils.logging import get_logger

logger = get_logger(__name__)


def train(
    config_path: str,
    features_csv: str,
    label_column: str = "label",
    id_column: str = "patient_id",
    omics_prefix: str = "omics_",
) -> None:
    cfg = PipelineConfig.from_yaml(config_path)
    df = pd.read_csv(features_csv)

    feature_cols = [c for c in df.columns if c not in (label_column, id_column)]
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df[label_column].values.astype(int)

    logger.info(f"Training on {len(df)} patients, {len(feature_cols)} features, "
                f"{cfg.classifier.num_classes} classes")

    clf = TabularClassifier(cfg.classifier, in_dim=X.shape[1])
    metrics = clf.fit(X, y)
    logger.info(f"Validation metrics: {metrics}")

    clf.save(cfg.classifier.checkpoint)
    histo_cols = [c for c in feature_cols if not c.startswith(omics_prefix)]
    omics_cols = [c for c in feature_cols if c.startswith(omics_prefix)]
    # histology columns must come first in the CSV, then omics, matching how the pipeline fuses them
    if feature_cols != histo_cols + omics_cols:
        raise SystemExit(f"Put all histology columns before the '{omics_prefix}*' omics columns in the CSV")
    save_feature_manifest(cfg.classifier.checkpoint, histo_cols, len(omics_cols))
    logger.info(f"Saved classifier to {cfg.classifier.checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--id-column", default="patient_id")
    parser.add_argument("--omics-prefix", default="omics_",
                        help="Columns starting with this prefix are omics features (must come last)")
    args = parser.parse_args()
    train(args.config, args.features_csv, args.label_column, args.id_column, args.omics_prefix)
