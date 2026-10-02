# Advanced feature analysis

> Synthetic data, simulated cohort of 150 patients. Validates code paths, not clinical performance.

- Advanced features per tile: 205 (151 after |r| < 0.95 pruning)
- Family sizes: nuclear morphology 23, nuclear shape 40, chromatin 48, cell graph 36, multiscale texture 30, tissue architecture 28

## Single family predictive power (macro AUC, grade)
- nuclear morphology: 0.881
- nuclear shape: 0.864
- chromatin: 0.898
- cell graph: 0.884
- multiscale texture: 0.859
- tissue architecture: 0.873

## Basic vs advanced (accuracy, macro AUC)
- Basic histology: acc 0.784 +/- 0.059, AUC 0.871
- Advanced histology: acc 0.822 +/- 0.045, AUC 0.889
- Advanced pruned (|r| < 0.95): acc 0.827 +/- 0.046, AUC 0.889
- Basic histology + omics: acc 0.820 +/- 0.044, AUC 0.901
- Advanced histology + omics: acc 0.827 +/- 0.039, AUC 0.897

## Spatial habitats vs true tissue (512 px slide, fully unsupervised)
- ARI 0.779, NMI 0.742, Shannon diversity 0.926, boundary fraction 0.163

## Point pattern statistics by grade (Kruskal-Wallis p)
- graph_clark_evans_R: p = 1.24e-01
- graph_voronoi_area_cv: p = 1.75e-01
- graph_delaunay_regularity_mean: p = 1.11e-05

## Figures

![fig13_cell_graph](figures/fig13_cell_graph.png)
![fig14_point_patterns](figures/fig14_point_patterns.png)
![fig15_nuclear_shape_chromatin](figures/fig15_nuclear_shape_chromatin.png)
![fig16_tumor_interface](figures/fig16_tumor_interface.png)
![fig17_feature_landscape](figures/fig17_feature_landscape.png)
![fig18_habitats](figures/fig18_habitats.png)
