# Demonstration analysis results

> All numbers below come from **synthetic** liver histology and a **simulated** multi-omics cohort (n = 150) with exact ground truth. They validate that the pipeline code works end to end. They are not clinical findings.

## Tissue segmentation (U-Net, held out 512 px slide)
- Pixel accuracy 0.992, mean Dice 0.990
- Dice per class: tumor 1.00, stroma 0.99, necrosis 0.99, normal 0.98

## Nuclei instance segmentation (color deconvolution + watershed)
- Precision 0.984, recall 0.953, F1 0.968, AJI 0.742 (700 predicted vs 723 true nuclei)
- Recall by region: tumor 0.99, stroma 0.99, necrosis 0.59, normal 0.93

## Stain normalization
- raw: between tile colour std 5.0, nuclei F1 0.940
- Macenko: between tile colour std 1.6, nuclei F1 0.899
- Reinhard: between tile colour std 0.1, nuclei F1 0.832

## Grade classification (5 fold x 3 repeat CV, XGBoost)
- Histology only: accuracy 0.784 +/- 0.059, macro AUC 0.871, macro F1 0.782
- Omics only: accuracy 0.624 +/- 0.079, macro AUC 0.821, macro F1 0.613
- Histology + omics: accuracy 0.820 +/- 0.044, macro AUC 0.901, macro F1 0.816

## Survival (out of fold C-index)
- Histology only (Cox): 0.746
- Omics only (Cox): 0.778
- Histology + omics (Cox): 0.767
- Histology + omics (DeepSurv): 0.741
- Grade label only: 0.747
- Log rank p across predicted risk tertiles: 1.50e-26

## Figures

![fig01_tissue_segmentation](figures/fig01_tissue_segmentation.png)
![fig02_unet_training](figures/fig02_unet_training.png)
![fig03_nuclei_segmentation](figures/fig03_nuclei_segmentation.png)
![fig04_nuclear_morphology](figures/fig04_nuclear_morphology.png)
![fig05_texture_and_spatial](figures/fig05_texture_and_spatial.png)
![fig06_stain_normalization](figures/fig06_stain_normalization.png)
![fig07_embedding](figures/fig07_embedding.png)
![fig08_classification](figures/fig08_classification.png)
![fig09_feature_importance](figures/fig09_feature_importance.png)
![fig10_survival](figures/fig10_survival.png)
![fig11_cohort_overview](figures/fig11_cohort_overview.png)
![fig12_grade_gallery](figures/fig12_grade_gallery.png)
