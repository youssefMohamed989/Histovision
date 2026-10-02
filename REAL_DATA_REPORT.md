# Real data validation

> This is the ONLY analysis in this repository that runs on genuine microscopy
> images rather than the synthetic generator. See `data/real_samples/SOURCES.md`
> for exactly where each image came from and its licence.

## Test 1: nuclei segmentation on real slides (no training required)
- torchstain target: 780 nuclei detected, mean area 66 px^2, mean circularity 0.95
- torchstain source: 1264 nuclei detected, mean area 49 px^2, mean circularity 0.95
- skimage IHC (colon): 536 nuclei detected, mean area 174 px^2, mean circularity 0.75

## Test 2: cross slide stain normalization (fit on one real slide, apply to another)
- Mean colour gap before normalization: 22.23
- After Macenko: 8.24
- After Reinhard: 0.44

## Test 3: advanced feature extraction + honest domain-gap check
- Features extracted per real image: {'torchstain target': 177, 'torchstain source': 177, 'skimage IHC (colon)': 177}
- Tissue U-Net (trained ONLY on synthetic data) applied to real images:
  - torchstain target: mean softmax confidence 0.88 (confidently WRONG, not uncertain), predicted composition {'tumor': 0.04, 'stroma': 0.0, 'necrosis': 0.05, 'normal': 0.91}
  - torchstain source: mean softmax confidence 0.83 (confidently WRONG, not uncertain), predicted composition {'tumor': 0.25, 'stroma': 0.02, 'necrosis': 0.6, 'normal': 0.12}
  - skimage IHC (colon): mean softmax confidence 0.92 (confidently WRONG, not uncertain), predicted composition {'tumor': 0.55, 'stroma': 0.0, 'necrosis': 0.21, 'normal': 0.24}

The model is not merely uncertain on real tissue -- it is confident and wrong, predicting
large near-uniform blocks that do not correspond to any real structure. This is the expected,
well known failure mode of a classifier evaluated far outside its training distribution, and is
reported here rather than hidden, and is exactly why `scripts/train_tissue_unet.py` and
`docs/data_preparation.md` describe training on real annotated data (e.g. PAIP 2019) before
using the tissue segmentation stage on real slides.

## Figures

![fig19_real_nuclei_segmentation](figures/fig19_real_nuclei_segmentation.png)
![fig20_real_stain_normalization](figures/fig20_real_stain_normalization.png)
![fig21_real_features_domain_gap](figures/fig21_real_features_domain_gap.png)
