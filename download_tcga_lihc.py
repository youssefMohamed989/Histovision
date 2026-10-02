"""Download TCGA-LIHC (liver hepatocellular carcinoma) data via the GDC API.

Fetches, for a small number of cases by default:

- Diagnostic H&E whole slide images (.svs)
- RNA-seq gene expression quantification (STAR - Counts)
- Somatic mutation calls (MAF)
- Clinical / demographic metadata

This only wraps the public GDC REST API (https://api.gdc.cancer.gov);
no authentication is required for open-access TCGA data. Large
downloads (WSIs are often 0.5-2 GB each) are not run by default -
pass ``--max-cases`` to control how many cases to pull.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from tqdm import tqdm

from liver_histo_ai.utils.logging import get_logger

logger = get_logger(__name__)

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

PROJECT_ID = "TCGA-LIHC"


def _query_files(data_category: str, data_format: str | None, max_cases: int) -> list[dict]:
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [PROJECT_ID]}},
            {"op": "in", "content": {"field": "files.data_category", "value": [data_category]}},
        ],
    }
    if data_format:
        filters["content"].append(
            {"op": "in", "content": {"field": "files.data_format", "value": [data_format]}}
        )

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id,data_category,data_format,file_size",
        "format": "JSON",
        "size": str(max_cases),
    }
    resp = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()["data"]["hits"]


def _download_file(file_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(f"{GDC_DATA_ENDPOINT}/{file_id}", stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))


def download_cohort(out_dir: str = "data/raw", max_cases: int = 5, include_slides: bool = True) -> None:
    out_dir = Path(out_dir)

    logger.info("Querying GDC for TCGA-LIHC diagnostic slides, RNA-seq, mutations, and clinical data")

    manifests = {}
    if include_slides:
        manifests["slides"] = _query_files("Biospecimen", "SVS", max_cases)
    manifests["rnaseq"] = _query_files("Transcriptome Profiling", "TSV", max_cases)
    manifests["mutations"] = _query_files("Simple Nucleotide Variation", "MAF", max_cases)
    manifests["clinical"] = _query_files("Clinical", "BCR XML", max_cases)

    for category, hits in manifests.items():
        logger.info(f"{category}: {len(hits)} files found")
        for hit in hits:
            file_id = hit["file_id"]
            file_name = hit["file_name"]
            _download_file(file_id, out_dir / category / file_name)

    logger.info(f"Download complete. Files written under {out_dir}")
    logger.info(
        "NOTE: RNA-seq / mutation / clinical files still need to be merged into the "
        "patients x genes tables expected by liver_histo_ai.omics.integration - see "
        "docs/data_preparation.md for the merge script."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--no-slides", action="store_true", help="Skip downloading WSIs (large files)")
    args = parser.parse_args()
    download_cohort(args.out, args.max_cases, include_slides=not args.no_slides)
