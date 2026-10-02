"""CLI entrypoint: run the full pipeline on a single slide.

    python scripts/run_pipeline.py \
        --slide data/raw/sample.svs \
        --patient-id TCGA-XX-YYYY \
        --config configs/default.yaml \
        --out results/patient_123
"""
from __future__ import annotations

import argparse
import json

from liver_histo_ai.config import PipelineConfig
from liver_histo_ai.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the liver-histo-ai pipeline end to end")
    parser.add_argument("--slide", required=True, help="Path to the WSI file")
    parser.add_argument("--patient-id", default=None, help="Patient id to join with omics tables")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default=None, help="Output directory (default: results/<slide_stem>)")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    result = run_pipeline(args.slide, cfg, patient_id=args.patient_id, out_dir=args.out)

    print(json.dumps(
        {k: v for k, v in result.items() if k != "histology_features"},
        indent=2,
    ))


if __name__ == "__main__":
    main()
