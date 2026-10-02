# Data preparation

## Whole slide images

`scripts/download_tcga_lihc.py` pulls diagnostic `.svs` slides for
TCGA-LIHC from the GDC. Slides are large (0.5-2 GB); start with a small
`--max-cases` for local development.

Once downloaded, tile a slide with:

```bash
python -m liver_histo_ai.preprocessing.wsi_tiling \
    --slide data/raw/slides/<file>.svs \
    --out data/tiles/<case_id> \
    --tile-size 512 --level 0
```

## Omics tables

The pipeline expects three CSVs, each indexed by patient/case barcode
(e.g. `TCGA-XX-YYYY`):

- `rnaseq_tcga_lihc.csv`: patients x genes, raw counts or TPM
- `mutations_tcga_lihc.csv`: patients x genes, 0/1 mutation indicator
- `clinical_tcga_lihc.csv`: patients x clinical variables (age, stage, etc.)

GDC delivers RNA-seq as one file per case (STAR - Counts TSVs) and
mutations as pooled MAF files. Merge these into the wide patient x gene
matrices above with a small pandas script, for example:

```python
import pandas as pd
from pathlib import Path

frames = []
for f in Path("data/raw/rnaseq").glob("*.tsv"):
    case_id = f.stem  # map back to the case barcode via the GDC manifest
    df = pd.read_csv(f, sep="\t", skiprows=1, index_col=0)
    frames.append(df["tpm_unstranded"].rename(case_id))

wide = pd.concat(frames, axis=1).T
wide.to_csv("data/omics/rnaseq_tcga_lihc.csv")
```

A minimal example table is provided at `data/omics/example_omics.csv`
for tests and demos.

## Tissue / nuclei segmentation training data

Public annotated liver histology datasets (e.g. PAIP2019 liver cancer
segmentation challenge) can be used to train `TissueUNet` and
`HoVerNet`. Format them into `images/` + `masks/` pairs as described in
`scripts/train_tissue_unet.py`.
