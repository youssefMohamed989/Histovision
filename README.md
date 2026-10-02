[README.md](https://github.com/user-attachments/files/32941400/README.md)
# Histovision
# Liver-Histo-AI

**Tile-based histopathology feature extraction, graph deep learning, and multi-omics fusion for H&E whole-slide images of hepatocellular carcinoma (HCC) and other primary liver cancers.**
<img width="2494" height="1508" alt="fig20_real_stain_normalization" src="https://github.com/user-attachments/assets/e02ef2ef-4def-4fe2-828a-a19f053c5b6d" />
<img width="2545" height="1659" alt="fig19_real_nuclei_segmentation" src="https://github.com/user-attachments/assets/c2ce284d-009f-44d9-8240-06a6d65c8be6" />
<img width="2628" height="1536" alt="fig21_real_features_domain_gap" src="https://github.com/user-attachments/assets/8ad4c18d-e19f-4cc8-a330-0f10d3751df9" />
<img width="2078" height="1202" alt="fig22_graph_mil_attention" src="https://github.com/user-attachments/assets/dae7ac23-7c71-49fb-bdcb-8999e13d7293" />
<img width="1590" height="1081" alt="fig09_feature_importance" src="https://github.com/user-attachments/assets/7cd68b71-926b-4877-9768-43c48027b8d4" />
<img width="1924" height="1397" alt="fig12_grade_gallery" src="https://github.com/user-attachments/assets/d733efc6-8549-4ef2-82c8-3ecfd2211285" />
<img width="2138" height="1536" alt="fig13_cell_graph" src="https://github.com/user-attachments/assets/55112519-56a1-4694-af8f-15d6d61acd92" />
<img width="2668" height="1614" alt="fig15_nuclear_shape_chromatin" src="https://github.com/user-attachments/assets/d450e664-1cb1-424b-adc5-9fd275705de2" />
<img width="2173" height="1698" alt="fig06_stain_normalization" src="https://github.com/user-attachments/assets/159684de-f4bc-4016-af03-1593e9f2cc57" />
<img width="2379" height="1995" alt="fig16_tumor_interface" src="https://github.com/user-attachments/assets/99989e2c-4710-4d7d-bd5a-a6ac41ec0a3b" />
<img width="3009" height="1999" alt="fig17_feature_landscape" src="https://github.com/user-attachments/assets/96b311dc-a7ad-40d9-8240-3deb14291efd" />



[![CI](https://github.com/<you>/liver-histo-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/<you>/liver-histo-ai/actions)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

> ⚠️ **Research software, not a medical device.** Every quantitative result in this
> repository comes from **synthetic** tissue and a **simulated** omics cohort, plus
> three small non-liver real images used for sanity checks. Nothing here is a
> clinical finding. See [Limitations](#limitations-and-honest-caveats).

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Repository layout](#repository-layout)
4. [Installation](#installation)
5. [Quickstart](#quickstart)
6. [Configuration](#configuration)
7. [Feature extraction](#feature-extraction)
8. [Cell-graph GNN + attention MIL](#cell-graph-gnn--attention-mil)
9. [Omics fusion, classification, survival](#omics-fusion-classification-survival)
10. [REST API](#rest-api)
11. [Results](#results)
12. [Real-data validation](#real-data-validation)
13. [Training on your own data](#training-on-your-own-data)
14. [Testing and CI](#testing-and-ci)
15. [Docker](#docker)
16. [Limitations and honest caveats](#limitations-and-honest-caveats)
17. [Roadmap](#roadmap)
18. [Citation, license, acknowledgements](#citation)

---

## What it does

Given a whole-slide image (`.svs`, `.ndpi`, `.tiff`) and optional patient-level omics
(RNA-seq, mutations, clinical variables), Liver-Histo-AI produces:

1. **Tiling and stain normalization** (Macenko / Reinhard) with tissue masking
2. **Tissue segmentation** into tumor, stroma, necrosis, and normal parenchyma (U-Net)
3. **Nuclei instance segmentation**, either a HoVer-Net-style network or a weight-free classical baseline
4. **~205 interpretable features per tile** across six families (nuclear shape, chromatin, cell-graph topology, multiscale texture, tissue architecture, basic morphology)
5. **Slide-level aggregation** with heterogeneity statistics and unsupervised spatial "habitats"
6. **Multi-omics fusion** (late, early, and gated-attention fusion)
7. **Prediction heads** for grade / subtype (XGBoost, MLP) and survival (Cox, DeepSurv)
8. **A second deep-learning route**: a from-scratch cell-graph attention network (CellGAT) feeding gated attention MIL, with two-level interpretability (which nuclei, which tiles)
9. **A FastAPI service and CLI** that share one orchestration function

Design principles: every stage is independently testable, wired together in exactly one
place (`pipeline.py`), and every analysis reports negative results rather than hiding them.

## Architecture

```
raw WSI (.svs/.ndpi/.tiff)
        │
        ▼
 [1] Tiling + stain normalization        preprocessing/
        │
        ▼
 [2] Tissue segmentation (U-Net)         segmentation/tissue_unet.py
        │
        ▼
 [3] Nuclei segmentation                 segmentation/nuclei_seg.py | classical.py
        │
        ├──────────────► [3b] Cell graph → CellGAT → Gated attention MIL   models/
        ▼
 [4] Tile features (~205)                features/
        │
        ▼
 [5] Slide aggregation + habitats        pipeline.py, features/habitats.py
        │                                   │
        ▼                                   ▼
 [6] Omics fusion  ◄────────────────  patient omics tables   omics/
        │
        ▼
 [7] Classifier (grade/subtype) / Survival (Cox, DeepSurv)   models/
        │
        ▼
   Prediction + feature report (JSON)
```

## Repository layout

```
liver-histo-ai/
├── src/liver_histo_ai/
│   ├── config.py                 dataclass config + YAML loading
│   ├── pipeline.py               end-to-end orchestration (CLI + API share this)
│   ├── synthetic.py              synthetic H&E tiles with exact ground truth
│   ├── metrics.py                Dice, confusion, instance F1 / AJI
│   ├── preprocessing/
│   │   ├── wsi_tiling.py         OpenSlide tiling, thumbnail tissue masking
│   │   └── stain_norm.py         Macenko / Reinhard
│   ├── segmentation/
│   │   ├── tissue_unet.py        5-level U-Net, 4 tissue classes
│   │   ├── nuclei_seg.py         HoVer-Net-style instance segmentation
│   │   ├── classical.py          weight-free nuclei segmentation baseline
│   │   └── losses.py             Dice / focal / weighted CE
│   ├── features/
│   │   ├── morphology.py         per-nucleus regionprops features
│   │   ├── texture.py            GLCM / Haralick
│   │   ├── spatial.py            density, nearest-neighbour statistics
│   │   ├── nuclear_shape.py      Fourier descriptors, curvature, Hu moments
│   │   ├── chromatin.py          intranuclear OD, clumping, margination
│   │   ├── cell_graph.py         Delaunay / Voronoi / MST / Ripley / Clark-Evans
│   │   ├── texture_advanced.py   Gabor, LBP, fractal dimension, lacunarity
│   │   ├── architecture.py       tumor nests, invasive-front profiles
│   │   ├── habitats.py           slide summaries + unsupervised habitats
│   │   ├── advanced.py           AdvancedHistologyExtractor (combines all)
│   │   └── selection.py          correlation pruning, stability selection
│   ├── models/
│   │   ├── cell_gnn.py           CellGAT (multi-head graph attention, pure PyTorch)
│   │   ├── attention_mil.py      gated attention MIL
│   │   ├── graph_mil.py          GraphMIL = CellGAT + MIL, trained jointly
│   │   ├── classifier.py         XGBoost + MLP behind one API
│   │   └── survival.py           penalized Cox, DeepSurv
│   ├── omics/
│   │   ├── integration.py        load / normalize RNA-seq, mutations, clinical
│   │   └── fusion.py             late / early / gated-attention fusion
│   ├── api/                      FastAPI app + pydantic schemas
│   └── utils/                    io, logging, seeding
├── scripts/                      train_tissue_unet, train_classifier,
│                                 train_graph_mil, run_pipeline, download_tcga_lihc
├── analysis/                     reproducible analyses, 24 figures, reports, JSON metrics
├── configs/default.yaml
├── data/                         example omics CSV + 3 real sample images (see SOURCES.md)
├── checkpoints/                  small demo checkpoints (U-Net, classifier)
├── docs/                         methods.md, data_preparation.md, FEATURE_CATALOG.md
├── tests/                        ~120 pytest tests
├── docker/                       Dockerfile, docker-compose.yml
└── .github/workflows/ci.yml      ruff + mypy + pytest + docker build
```

## Installation

```bash
git clone https://github.com/<you>/liver-histo-ai.git
cd liver-histo-ai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

OpenSlide's C library is required at the OS level:

```bash
# Debian / Ubuntu
sudo apt-get install openslide-tools libgl1 libglib2.0-0
# macOS
brew install openslide
```

Requires Python ≥ 3.10. A GPU is optional (`train.device: cuda` in the config; set `cpu` if you have none).

## Quickstart

```bash
# 1. Tile + stain-normalize a slide
python -m liver_histo_ai.preprocessing.wsi_tiling \
    --slide data/raw/sample.svs --out data/tiles/sample \
    --tile-size 512 --stain-method macenko

# 2. Run the full pipeline on one slide
python scripts/run_pipeline.py \
    --slide data/raw/sample.svs \
    --patient-id TCGA-XX-YYYY \
    --config configs/default.yaml \
    --out results/TCGA-XX-YYYY

# 3. Serve the REST API
uvicorn liver_histo_ai.api.main:app --reload
```

Outputs written to `--out`: `tiles/` (images + manifest CSV), `histology_features.json`
(slide-level feature vector), and `pipeline_result.json` (including prediction if a
classifier checkpoint exists and `--patient-id` is given).

Reproduce the demo analyses (synthetic data, CPU-friendly):

```bash
python analysis/run_analysis.py --quick     # smoke run
python analysis/run_analysis.py             # ~5 min, figures 01-12
python analysis/run_advanced.py             # ~8 min, figures 13-18
python analysis/run_real_data_validation.py # figures 19-21
python analysis/run_graph_mil.py            # figures 22-23
```

## Configuration

All hyperparameters live in `configs/default.yaml` and map onto dataclasses in `config.py`.

| Section | Key options |
| --- | --- |
| `tiling` | `tile_size: 512`, `stride`, `tissue_threshold: 0.10`, `stain_normalize`, `stain_method: macenko \| reinhard \| none` |
| `tissue_seg` | 4 classes `[tumor, stroma, necrosis, normal]`, `base_channels: 32`, `checkpoint` |
| `nuclei_seg` | `checkpoint`, `min_nucleus_area: 10`, `max_nucleus_area: 4000` |
| `features` | `advanced: true`, `advanced_aggregations`, `slide_stats: [mean, std, q90]`, `habitats: 4`, GLCM distances/angles |
| `omics` | table paths, `top_k_genes: 200`, `normalization: log1p_tpm` |
| `fusion` | `method: late \| early \| gated`, embedding dims |
| `classifier` | `task`, `model_type: xgboost \| mlp`, `num_classes`, `checkpoint` |
| `train` | epochs, batch size, lr, weight decay, seed, device, early-stopping patience |

## Feature extraction

`AdvancedHistologyExtractor` (`features/advanced.py`) computes ~205 features per tile
(`features.advanced: true`, the default):

| Family | # | Highlights |
| --- | --- | --- |
| Nuclear morphology | 23 | area, perimeter, eccentricity, solidity, circularity, axis ratio (mean/std/median/p90) |
| Nuclear shape | 40 | contour Fourier descriptors, radial-distance variability, curvature and bending energy, hull roughness, concavity count/depth, log Hu moments |
| Chromatin | 48 | intranuclear hematoxylin OD stats, clumping, envelope margination, masked GLCM (pixel pairs inside the nucleus only), multiscale LBP, colour ratios |
| Cell graph | 36 | Delaunay regularity, Voronoi area disorder, MST branching, Ripley's L(r)−r, Clark-Evans index, DBSCAN clustering, Moran's I, orientation order, Morisita / Gini density heterogeneity |
| Multiscale texture | 30 | Gabor bank, multi-radius LBP, fractal dimension, lacunarity, H&E OD statistics |
| Tissue architecture | 28 | tumor-nest geometry, what borders the tumor, signed-distance density profile across the invasive front |

Architecture features require a tissue-segmentation checkpoint; without one they are skipped
(the pipeline logs a warning). That is why real-image runs report 177 features rather than 205.

Slide-level aggregation keeps heterogeneity (mean / std / q90 …) instead of just averaging,
and optionally clusters tiles into **spatial habitats** and reports diversity, patchiness,
and boundary fraction.

Regenerate the full catalogue (every feature name + one-line description):

```bash
python -m liver_histo_ai.features.advanced   # writes docs/FEATURE_CATALOG.md
```

`features/selection.py` provides `prune_correlated` and `stability_selection`. Fit both
**inside CV folds only**, otherwise you leak labels.

![cell graph](analysis/figures/fig13_cell_graph.png)
![nuclear shape and chromatin](analysis/figures/fig15_nuclear_shape_chromatin.png)
![feature landscape](analysis/figures/fig17_feature_landscape.png)
![habitats](analysis/figures/fig18_habitats.png)

## Cell-graph GNN + attention MIL

A second, end-to-end deep-learning route to patient-level prediction, implemented in plain
PyTorch (**no torch-geometric dependency**):

- **CellGAT**: multi-head graph attention (Veličković et al., 2018) on each tile's nuclei
  graph (nodes = nuclei, edges = pruned Delaunay adjacency with length/orientation features),
  with a from-scratch scatter-softmax and block-diagonal graph batching. A learned node-level
  attention readout produces one embedding per tile.
- **GatedAttentionMIL**: gated attention multiple-instance learning (Ilse, Tomczak & Welling,
  2018) pools a patient's bag of tile embeddings into one prediction.
- **GraphMIL**: chains both with a single loss, giving two-level attention: *which nuclei
  mattered within a tile* and *which tiles mattered within a patient*.

```bash
# manifest columns: patient_id, tile_path, label
python scripts/train_graph_mil.py --manifest data/bags/manifest.csv \
    --config configs/default.yaml --out checkpoints/graph_mil.pt \
    --epochs 90 --val-split 0.2
```

The test suite covers this module specifically: batched vs. looped GAT forward passes agree
numerically, gradients reach every parameter, node attention is permutation-equivariant, and
attention sums to 1 and concentrates on a bag's informative instance after training.

### What the synthetic experiments actually showed

(90 synthetic patients × 4 tiles; see `analysis/GRAPH_MIL_REPORT.md`)

| Question | Result |
| --- | --- |
| Does node attention find atypical nuclei? | Partially. Spearman ρ = 0.12 vs. area, −0.34 vs. circularity. Attention maps show it mostly highlights elongated spindle-shaped stromal nuclei (geometric outliers), not tumor-specific atypia. |
| Does tile attention find the most informative tile? | **No.** ρ = 0.007, a negative result. |
| Patient-level accuracy (5-fold OOF) | GraphMIL 0.700 acc / 0.866 AUC vs. mean-pooled logistic-regression baseline **0.822 / 0.926**. The baseline wins at n = 90. |

**Diagnosing the gap.** A learning curve (n = 90/180/300) showed GraphMIL getting *worse* with
more data (0.633 → 0.557 accuracy) while the baseline held steady, which is backwards. The
cause was the training loop: `train_graph_mil` takes one full-batch gradient step per epoch, so
a fixed epoch count gives larger cohorts the same number of updates. Testing it directly on 300
patients (only the epoch count varied): 30 epochs → 0.607 acc, 90 epochs → **0.813 acc / 0.932
AUC** (baseline: 0.840). The remaining gap at n = 90 is plausibly a genuine data-efficiency
difference; that is inference from two cohort sizes, not proof. Full arc:
`GRAPH_MIL_REPORT.md`, `GRAPH_MIL_LEARNING_CURVE.md`, `GRAPH_MIL_EPOCH_TEST.md`.

![multi level attention](analysis/figures/fig22_graph_mil_attention.png)
![learning curve](analysis/figures/fig24_graph_mil_learning_curve.png)

## Omics fusion, classification, survival

- `omics/integration.py`: loads RNA-seq, mutation, and clinical tables; normalizes expression
  (log1p / z-score / quantile); keeps the top-k most variable genes.
- `omics/fusion.py`: **late**, **early**, and **gated-attention** fusion as PyTorch modules;
  missing-omics imputation for histology-only patients.
- `models/classifier.py`: XGBoost and MLP behind one API.
- `models/survival.py`: penalized Cox regression and DeepSurv.

Expected omics inputs (indexed by TCGA barcode, e.g. `TCGA-XX-YYYY`): `rnaseq_tcga_lihc.csv`
(patients × genes), `mutations_tcga_lihc.csv`, `clinical_tcga_lihc.csv`. A tiny example is in
`data/omics/example_omics.csv`. Details: `docs/data_preparation.md`.

## REST API

```bash
uvicorn liver_histo_ai.api.main:app --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | liveness + compute device |
| GET | `/pipeline/config` | currently loaded default configuration |
| POST | `/pipeline/run` | run the full pipeline on a **server-side** slide path |

```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"slide_path": "data/raw/sample.svs", "patient_id": "TCGA-XX-YYYY",
       "config_path": "configs/default.yaml"}'
```

Response: `slide_path`, `num_tiles`, `histology_features` (dict), optional `omics_vector_dim`,
and optional `prediction` (`predicted_class`, `class_probabilities`). Interactive docs at `/docs`.

> The service reads slides from the server's filesystem and ships with permissive CORS
> (`*`). Put it behind authentication and restrict origins before exposing it anywhere.

## Results

All results below: **synthetic** tissue + **simulated** cohort (n = 150) with exact ground
truth. They show the code works end to end; they are not clinical performance.

### Segmentation (held-out 512 px synthetic slide)

| Task | Metric | Value |
| --- | --- | --- |
| Tissue U-Net | pixel accuracy / mean Dice | 0.992 / 0.990 |
| Nuclei (color deconvolution + watershed) | precision / recall / F1 / AJI | 0.984 / 0.953 / 0.968 / 0.742 |
| Nuclei recall by region | tumor / stroma / normal / necrosis | 0.99 / 0.99 / 0.93 / 0.59 |

### Stain normalization (simulated stain variability)

| Method | Between-tile colour std | Nuclei F1 |
| --- | --- | --- |
| Raw | 5.0 | 0.940 |
| Macenko | 1.6 | 0.899 |
| Reinhard | 0.1 | 0.832 |

Normalization reduces colour variance but **costs** segmentation F1 on this data; it is a
trade-off, not a free win.

### Grade classification (5-fold × 3 repeats, XGBoost)

| Inputs | Accuracy | Macro AUC |
| --- | --- | --- |
| Histology (basic) | 0.784 ± 0.059 | 0.871 |
| Omics only | 0.624 ± 0.079 | 0.821 |
| Histology + omics | 0.820 ± 0.044 | 0.901 |
| Histology (advanced, 205 features) | 0.822 ± 0.045 | 0.889 |
| Advanced + omics | 0.827 ± 0.039 | 0.897 |

Single-family AUCs for grade: chromatin 0.898, cell graph 0.884, nuclear morphology 0.881,
tissue architecture 0.873, nuclear shape 0.864, multiscale texture 0.859. Unsupervised
habitat clustering recovers the true tissue map with ARI 0.78 / NMI 0.74. Delaunay
triangle regularity separates grades (Kruskal-Wallis p = 1.1e-05).

### Survival (out-of-fold C-index)

| Model | C-index |
| --- | --- |
| Histology Cox | 0.746 |
| Omics Cox | 0.778 |
| Histology + omics Cox | 0.767 |
| Histology + omics DeepSurv | 0.741 |
| Grade label only | 0.747 |

### Figure index (`analysis/figures/`)

| Figures | Content |
| --- | --- |
| 01-03 | tissue segmentation, U-Net training curves, nuclei segmentation |
| 04-06 | nuclear morphology, texture/spatial maps, stain normalization |
| 07-09 | t-SNE embeddings, classification ROC/ablation, feature importance |
| 10-12 | survival (KM, C-index, Cox forest), cohort overview, grade gallery |
| 13-18 | cell graphs, point patterns, shape/chromatin, invasive front, feature landscape, habitats |
| 19-21 | real-image nuclei segmentation, cross-slide stain norm, real-image domain gap |
| 22-24 | GraphMIL attention, attention validity, learning curve |

![tissue segmentation](analysis/figures/fig01_tissue_segmentation.png)
![nuclei segmentation](analysis/figures/fig03_nuclei_segmentation.png)
![survival](analysis/figures/fig10_survival.png)

## Real-data validation

`analysis/run_real_data_validation.py` runs production code on three **genuine** microscopy
images (two real H&E tumor slides from the torchstain repo, one real colon IHC image from
scikit-image; provenance and licenses in `data/real_samples/SOURCES.md`). None are liver tissue
and none have ground-truth annotations; the build environment could not reach GDC/TCIA.

| Test | Result |
| --- | --- |
| Classical nuclei segmentation, no training | 780 / 1264 nuclei on the two H&E images, 536 on IHC |
| Cross-slide stain normalization (mean colour gap) | 22.2 → 8.2 (Macenko) → **0.44** (Reinhard) |
| Advanced features on real images | 177 extracted per image (architecture family needs a tissue checkpoint) |
| Synthetic-trained tissue U-Net on real tissue | **Fails, confidently** (mean softmax confidence 0.83–0.92, predicted composition meaningless) |

**It caught a real bug.** The classical segmenter's original "redraw each blob at half its own
peak" heuristic collapsed dense touching nuclei into single detections (19 found on a slide with
hundreds visible). Fixing it with a sliding-window local peak raised real detections from 19 to
780 and *improved* synthetic F1/AJI too. The domain-gap result is exactly why you must train the
tissue U-Net on real annotated data (e.g. PAIP 2019) before using it on real slides.

![real nuclei segmentation](analysis/figures/fig19_real_nuclei_segmentation.png)
![real stain normalization](analysis/figures/fig20_real_stain_normalization.png)

## Training on your own data

**1. Get data.** `scripts/download_tcga_lihc.py` wraps the public GDC API:

```bash
python scripts/download_tcga_lihc.py --out data/raw --max-cases 5          # start small
python scripts/download_tcga_lihc.py --out data/raw --no-slides            # omics/clinical only
```

Diagnostic slides are 0.5-2 GB each. No data is bundled with the repo.

**2. Tissue U-Net**: needs tile/mask pairs (`images/*.png`, `masks/*.png`, pixel value = class index):

```bash
python scripts/train_tissue_unet.py --config configs/default.yaml --data-dir data/tissue_seg --epochs 50
```

**3. Tabular classifier**: from a feature CSV:

```bash
python scripts/train_classifier.py --config configs/default.yaml \
    --features-csv features.csv --label-column label --id-column patient_id
```

**4. GraphMIL**: see [above](#cell-graph-gnn--attention-mil). Use enough epochs: full-batch
training needs more steps for larger cohorts, so monitor validation loss.

**Checkpoint note.** `checkpoints/` ships small *demo* models (`demo_tissue_unet.pt`,
`demo_classifier.joblib`) trained on synthetic data. The default config expects
`checkpoints/tissue_unet.pt` and `checkpoints/classifier.pt`, so either rename/point the config
at the demo files for a smoke test or train your own. Without a nuclei checkpoint,
`run_pipeline` falls back to a randomly initialized HoVer-Net (it logs a warning); use real
trained weights, or the classical segmenter, for meaningful output.

## Testing and CI

```bash
pytest -v --cov=liver_histo_ai
ruff check src tests scripts
```

~120 tests cover segmentation, losses, stain normalization, texture/spatial/morphology features,
the advanced feature stack, omics, survival, classifier, config, API, synthetic generator, and
GraphMIL. GitHub Actions (`.github/workflows/ci.yml`) runs ruff, mypy (non-blocking), and pytest
with coverage on Python 3.10 and 3.11, then builds the Docker image.

## Docker

```bash
docker compose -f docker/Dockerfile up --build   # see below for the correct path
docker compose -f docker/docker-compose.yml up --build
```

Serves the API on port 8000 with `data/`, `results/`, and `checkpoints/` mounted as volumes
and a `/health` healthcheck.

## Limitations and honest caveats

- **No real liver results.** Performance numbers are synthetic-data code-validation numbers.
  Real-world performance is unmeasured.
- **Synthetic-to-real gap.** The tissue U-Net trained on synthetic data is confidently wrong on
  real tissue. Retrain on annotated real data.
- **Simulated omics.** Histology and omics share a latent variable by construction, so the
  fusion gain is partly baked in.
- **GraphMIL does not beat a mean-pooled baseline at n = 90**, and tile attention did not learn
  to select informative tiles. Node attention mainly flags elongated stromal nuclei.
- **Stain normalization can hurt** nuclei segmentation (F1 0.940 raw → 0.899 Macenko → 0.832
  Reinhard on simulated variation).
- **Nuclei recall in necrosis is low** (0.59).
- **~205 features on small cohorts** invites overfitting; use the selection utilities inside CV.
- **API security**: open CORS and server-side file paths; not production-hardened.
- **Not validated for clinical use** and must not inform diagnosis or treatment.

## Roadmap

- Run the full pipeline on TCGA-LIHC and report cross-validated results with external validation
- Train the tissue U-Net on PAIP 2019 or similar annotated liver data
- Pretrain the CellGAT encoder on a large nuclei dataset before MIL fine-tuning
- Minibatch training and validation-monitored early stopping for GraphMIL
- Tile-level supervision or auxiliary losses to give tile attention a real learning signal
- Slide-upload API endpoint with authentication; PDF report generation

## Citation

If you use this code, please cite the repository and the TCGA Research Network for the
underlying liver cancer cohort.

```bibtex
@software{liver_histo_ai,
  title  = {Liver-Histo-AI: histopathology feature extraction and multi-omics fusion for liver cancer},
  author = {Liver-Histo-AI Contributors},
  year   = {2026},
  url    = {https://github.com/<you>/liver-histo-ai}
}
```

## License

MIT, see [`LICENSE`](LICENSE). Bundled sample images carry their own licenses; see
`data/real_samples/SOURCES.md`.

## Acknowledgements

TCGA Research Network and the GDC; [OpenSlide](https://openslide.org/);
[torchstain](https://github.com/EIDOSLAB/torchstain) (sample images); scikit-image; HoVer-Net
(Graham et al., 2019); Macenko et al. (2009) and Reinhard et al. (2001) for stain normalization;
Veličković et al. (2018) for GAT; Ilse, Tomczak & Welling (2018) for attention MIL.
