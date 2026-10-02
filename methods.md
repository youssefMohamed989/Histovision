# Methods

## 1. Preprocessing

**Tiling.** `preprocessing/wsi_tiling.py` reads a slide with OpenSlide, builds a
tissue mask on a thumbnail (Otsu threshold on HSV saturation, morphological
open and close), and keeps tiles whose tissue fraction exceeds a threshold.
A manifest CSV records every tile position so results can be mapped back to
slide coordinates.

**Stain normalization.** `preprocessing/stain_norm.py` provides

- *Macenko*: estimates the two stain vectors from the optical density (OD)
  distribution via an eigen decomposition, orients the basis into the
  positive OD half space, finds robust extreme angles, then remaps the
  stain concentrations to a reference stain matrix.
- *Reinhard*: matches per channel mean and standard deviation in CIELAB.

## 2. Tissue segmentation

`segmentation/tissue_unet.py` is a five level U-Net (batch norm, transposed
convolution upsampling, skip connections) predicting four classes: tumor,
stroma, necrosis, normal parenchyma. Training uses Dice plus cross entropy
(`segmentation/losses.py`), Adam with cosine annealing, and flip augmentation.

## 3. Nuclei segmentation

Two interchangeable paths:

1. `segmentation/nuclei_seg.py`: a HoVer-Net style network with nuclear pixel,
   horizontal/vertical distance and optional nuclear type branches, decoded
   with a marker controlled watershed. Requires trained weights.
2. `segmentation/classical.py`: weight free baseline. Hematoxylin colour
   deconvolution, permissive blob detection at 0.4 x Otsu, then each blob's
   edge is redrawn at half of its own peak intensity so dark and pale nuclei
   are each delineated at their own contrast, followed by a distance transform
   watershed to split touching nuclei.

## 4. Features

| Family | Module | Content |
| --- | --- | --- |
| Morphology | `features/morphology.py` | area, perimeter, eccentricity, solidity, extent, axis ratio, circularity, per nucleus, aggregated to mean / std / median / p90 |
| Texture | `features/texture.py` | GLCM contrast, dissimilarity, homogeneity, energy, correlation, ASM at several distances, angle averaged |
| Spatial | `features/spatial.py` | nuclear density, nearest neighbour distance statistics |
| Tissue composition | `pipeline.py` and analysis | fraction of tile per predicted tissue class |

## 5. Omics and fusion

`omics/integration.py` loads RNA-seq, mutation and clinical tables, normalizes
expression (log1p, z score or quantile) and keeps the most variable genes.
`omics/fusion.py` implements late, early and gated attention fusion as PyTorch
modules. `models/classifier.py` wraps XGBoost and an MLP behind one API, and
`models/survival.py` provides penalized Cox regression and DeepSurv.

## 6. Demonstration analysis

`analysis/run_analysis.py` exercises every module above on synthetic data.

**Why synthetic.** Annotated liver WSIs are large and access controlled. The
generator (`synthetic.py`) produces H&E style tiles with exact tissue and
per nucleus ground truth, so segmentation accuracy is measured against truth
rather than estimated. Region specific nuclear biology is built in: tumor
nuclei grow larger, more pleomorphic and more hyperchromatic with grade;
stromal nuclei are elongated and sparse; normal hepatocyte nuclei are small
and round; necrosis is pale with faint debris. Staining variability is
simulated in optical density space so it is Beer-Lambert consistent.

**Simulated cohort.** A latent aggressiveness score z (class plus noise)
drives both the tile appearance and a simulated omics profile (upregulated,
downregulated, weakly informative and pure noise genes, plus five driver
mutation indicators) and a Weibull survival time with censoring. Histology
and omics each see z through independent noise, which is what makes fusion
informative.

**Evaluation.** Grade classification uses 5 fold stratified cross validation
repeated 3 times. Survival risk scores are out of fold: PCA (8 components) is
fit inside each training fold, then penalized Cox or DeepSurv, and the
concordance index is computed on pooled held out risks.

## 7. Advanced histology feature extraction

`features/advanced.py` (`AdvancedHistologyExtractor`) produces roughly 200
features per tile across six families, callable standalone or through
`pipeline.extract_tile_features` when `features.advanced: true` in the config
(the default). Run `python -m liver_histo_ai.features.advanced` to regenerate
`docs/FEATURE_CATALOG.md` with every feature name and a one line description.

| Family | Module | Content |
| --- | --- | --- |
| Nuclear shape | `features/nuclear_shape.py` | resampled contour Fourier descriptors (ellipticity, harmonics 2 to 6, high frequency energy), radial distance CV, discrete curvature and bending energy, roughness vs convex hull, concavity count and depth, log Hu moments |
| Chromatin | `features/chromatin.py` | intranuclear hematoxylin OD mean/std/skew/kurtosis/entropy, contrast against the cytoplasm ring, hyperchromatic fraction, chromatin clump count and size, nuclear envelope margination, masked GLCM computed only from intranuclear pixel pairs, uniform LBP, eosin and blue colour ratios |
| Cell graph | `features/cell_graph.py` | Delaunay edge length and triangle regularity (long boundary edges pruned), Voronoi cell area disorder and compactness, minimum spanning tree branching, Clark-Evans index, Ripley L(r) - r at several radii, DBSCAN clustering, nematic orientation order, Moran's I, large/small nucleus neighbour mixing, Morisita index and Gini of gridded density |
| Multiscale texture | `features/texture_advanced.py` | Gabor filter bank energy and anisotropy at 3 frequencies, multi radius uniform LBP entropy, box counting fractal dimension and gliding box lacunarity of the nuclear foreground, H&E optical density statistics, colourfulness, edge density |
| Tissue architecture | `features/architecture.py` | tissue class fractions, tumor nest count/size/solidity/fragmentation, what borders the tumor rim, signed distance density profile across the invasive front (core, edge, front, far field) |
| Slide/habitats | `features/habitats.py` | per feature distribution statistics across tiles (mean, std, quantiles, CV), unsupervised habitat clustering of tiles (KMeans on PCA of scaled features, labels ordered by a chosen axis) and the resulting habitat diversity, patchiness and boundary fraction on the tile grid |

Two utilities support using ~200 features on modest cohorts
(`features/selection.py`): `prune_correlated` (greedy redundancy removal) and
`stability_selection` (ANOVA-F selection frequency across subsamples), both
meant to be fit inside cross validation folds only.

`analysis/run_advanced.py` extracts this full feature set on the synthetic
cohort and produces figures 13 to 18: cell graph topology, Ripley/Clark-Evans
point pattern statistics by grade, nuclear shape and chromatin galleries with
Fourier spectra, the tumor invasive front density profile, a feature family
ablation plus stability selection and correlation structure, and unsupervised
spatial habitats compared against the true tissue map.

## 8. Real data validation and a bug it caught

`analysis/run_real_data_validation.py` runs the production code on genuine
microscopy images rather than the synthetic generator (see
`data/real_samples/SOURCES.md`). This is the only analysis in the repository
that does so, and it is what caught a real generalization bug: the classical
nuclei segmenter's original per-blob half-maximum threshold assumed each
low-threshold connected component was roughly one nucleus, which holds for
the cleanly separated synthetic tiles but not for real tissue, where many
touching nuclei of similar intensity merge into one blob at the permissive
first threshold. Thresholding that whole merged blob against its own single
global peak collapsed a cluster of dozens of real nuclei down to a tiny
fragment near the brightest point (19 detections on a real slide with
hundreds of visible nuclei). The fix computes the local peak in a small
sliding window instead, auto-sized per image from the tissue's own scale via
the distance transform of the permissive mask, which is robust to merged
blobs since the distance transform still ridges near each nucleus's own
centre. This raised real-image recall roughly 40-fold and also improved
segmentation F1/AJI on the synthetic benchmark (Section 6), so it was not a
synthetic-vs-real tradeoff -- the blob-level heuristic was simply wrong.

## 9. Limitations

- Every number produced by `analysis/` comes from synthetic or simulated
  data. It verifies correctness of the code paths, not clinical performance.
- Real tissue is messier than the generator: folds, pen marks, out of focus
  regions, overlapping nuclei and far richer texture. Expect lower accuracy.
- The classical nuclei segmenter recovers roughly 60 percent of the faintest
  (necrotic debris) nuclei; use a trained HoVer-Net for real data.
- The HoVer-Net and the tissue U-Net shipped in code have no pretrained
  weights for real liver tissue. Train them on annotated data such as the
  PAIP 2019 liver cancer challenge set.
- For real cohorts, patient level splits are required so that multiple tiles
  or slides from one patient never straddle train and test folds.
- The demo checkpoints in `checkpoints/` (`demo_tissue_unet.pt`, `demo_classifier.joblib`) were trained
  on the synthetic cohort only (the U-Net needs `tissue_seg.base_channels: 12`, not the default 32). The classifier expects the 123 column layout of the synthetic analysis
  tables, not the output of `run_pipeline` on a real slide, and has no `.features.json` manifest, so
  the pipeline will not use it for prediction. The `.joblib` file is a pickle: load only files you trust.
- `run_pipeline` uses the classical nuclei segmenter unless a trained HoVer-Net checkpoint exists
  (`nuclei_seg.backend`). A randomly initialised network is never used.
- Stain normalisation is fitted tile by tile, so stain appearance can vary between neighbouring tiles of one
  slide. Fitting one reference per slide (`fit_macenko_reference`) is the safer choice for real cohorts.
- Tiling only keeps full tiles: a partial row/column of tiles at the right and bottom slide edge is dropped.
- `build_cell_graph` z-scores node features within each tile. This removes absolute nuclear size and shape
  (only values relative to the other nuclei in the tile remain), which probably limits the grade signal
  GraphMIL can use. The GAT layers also have no self loops and encode edge direction by sin(angle) only.
  These are untested design limitations; the reported GraphMIL results were produced with them.
- The survival and classifier code has no hyperparameter tuning; reported metrics are cross validated on
  synthetic data only.

## 10. Intended use

Research and education only. Not a medical device. Nothing here has been validated on real patient data
and no output should inform diagnosis, prognosis or treatment.
