# Feature catalog

205 features produced per tile by `AdvancedHistologyExtractor` (with default aggregations).

## Nuclear morphology (23)

| feature | meaning |
| --- | --- |
| `nuc_count` | Number of detected nuclei in the tile |
| `nuc_area_mean` | Mean nuclear area (px^2) across nuclei |
| `nuc_area_std` | Std. dev. of nuclear area (px^2) across nuclei |
| `nuc_perimeter_mean` | Mean nuclear perimeter (px) across nuclei |
| `nuc_perimeter_std` | Std. dev. of nuclear perimeter (px) across nuclei |
| `nuc_eccentricity_mean` | Mean nuclear eccentricity (0 = circle, near 1 = elongated) across nuclei |
| `nuc_eccentricity_std` | Std. dev. of nuclear eccentricity (0 = circle, near 1 = elongated) across nuclei |
| `nuc_solidity_mean` | Mean area over convex hull area across nuclei |
| `nuc_solidity_std` | Std. dev. of area over convex hull area across nuclei |
| `nuc_extent_mean` | Mean area over bounding box area across nuclei |
| `nuc_extent_std` | Std. dev. of area over bounding box area across nuclei |
| `nuc_major_axis_length_mean` | Mean best fit ellipse major axis length across nuclei |
| `nuc_major_axis_length_std` | Std. dev. of best fit ellipse major axis length across nuclei |
| `nuc_minor_axis_length_mean` | Mean best fit ellipse minor axis length across nuclei |
| `nuc_minor_axis_length_std` | Std. dev. of best fit ellipse minor axis length across nuclei |
| `nuc_axis_ratio_mean` | Mean major over minor axis length across nuclei |
| `nuc_axis_ratio_std` | Std. dev. of major over minor axis length across nuclei |
| `nuc_circularity_mean` | Mean 4 pi area / perimeter^2 (1 = perfect circle) across nuclei |
| `nuc_circularity_std` | Std. dev. of 4 pi area / perimeter^2 (1 = perfect circle) across nuclei |
| `nuc_convex_area_mean` | Mean convex hull area across nuclei |
| `nuc_convex_area_std` | Std. dev. of convex hull area across nuclei |
| `nuc_equivalent_diameter_mean` | Mean diameter of a circle with the same area across nuclei |
| `nuc_equivalent_diameter_std` | Std. dev. of diameter of a circle with the same area across nuclei |

## Nuclear shape (40)

| feature | meaning |
| --- | --- |
| `nuc_shp_fd_ellipticity_mean` | Fourier contour ellipticity (minor/major first harmonic); 1 = perfect ellipse |
| `nuc_shp_fd_ellipticity_std` | Fourier contour ellipticity (minor/major first harmonic); 1 = perfect ellipse |
| `nuc_shp_fd_harmonic2_mean` | Normalized Fourier amplitude of contour harmonic k (lobulation at scale k) |
| `nuc_shp_fd_harmonic2_std` | Normalized Fourier amplitude of contour harmonic k (lobulation at scale k) |
| `nuc_shp_fd_harmonic3_mean` | Normalized Fourier amplitude of contour harmonic k (lobulation at scale k) |
| `nuc_shp_fd_harmonic3_std` | Normalized Fourier amplitude of contour harmonic k (lobulation at scale k) |
| `nuc_shp_fd_harmonic4_mean` | Normalized Fourier amplitude of contour harmonic k (lobulation at scale k) |
| `nuc_shp_fd_harmonic4_std` | Normalized Fourier amplitude of contour harmonic k (lobulation at scale k) |
| `nuc_shp_fd_harmonic5_mean` | Normalized Fourier amplitude of contour harmonic k (lobulation at scale k) |
| `nuc_shp_fd_harmonic5_std` | Normalized Fourier amplitude of contour harmonic k (lobulation at scale k) |
| `nuc_shp_fd_harmonic6_mean` | Normalized Fourier amplitude of contour harmonic k (lobulation at scale k) |
| `nuc_shp_fd_harmonic6_std` | Normalized Fourier amplitude of contour harmonic k (lobulation at scale k) |
| `nuc_shp_fd_high_freq_energy_mean` | Energy in high frequency contour harmonics (jagged outline) |
| `nuc_shp_fd_high_freq_energy_std` | Energy in high frequency contour harmonics (jagged outline) |
| `nuc_shp_radial_cv_mean` | Coefficient of variation of centroid to boundary distance |
| `nuc_shp_radial_cv_std` | Coefficient of variation of centroid to boundary distance |
| `nuc_shp_radial_range_mean` | Range of centroid to boundary distance relative to its mean |
| `nuc_shp_radial_range_std` | Range of centroid to boundary distance relative to its mean |
| `nuc_shp_curvature_mean_abs_mean` | Contour curvature statistic scaled by equivalent radius |
| `nuc_shp_curvature_mean_abs_std` | Contour curvature statistic scaled by equivalent radius |
| `nuc_shp_curvature_std_mean` | Contour curvature statistic scaled by equivalent radius |
| `nuc_shp_curvature_std_std` | Contour curvature statistic scaled by equivalent radius |
| `nuc_shp_bending_energy_mean` | Normalized bending energy of the contour (circle = 1) |
| `nuc_shp_bending_energy_std` | Normalized bending energy of the contour (circle = 1) |
| `nuc_shp_n_inflections_mean` | Number of curvature sign changes (notches) |
| `nuc_shp_n_inflections_std` | Number of curvature sign changes (notches) |
| `nuc_shp_roughness_mean` | Perimeter divided by convex hull perimeter |
| `nuc_shp_roughness_std` | Perimeter divided by convex hull perimeter |
| `nuc_shp_n_concavities_mean` | Number of concavities in the convex deficiency |
| `nuc_shp_n_concavities_std` | Number of concavities in the convex deficiency |
| `nuc_shp_max_concavity_depth_mean` | Deepest concavity relative to equivalent radius |
| `nuc_shp_max_concavity_depth_std` | Deepest concavity relative to equivalent radius |
| `nuc_shp_concavity_area_frac_mean` | Convex deficiency area over hull area |
| `nuc_shp_concavity_area_frac_std` | Convex deficiency area over hull area |
| `nuc_shp_hu1_log_mean` | Log10 magnitude of Hu invariant moment |
| `nuc_shp_hu1_log_std` | Log10 magnitude of Hu invariant moment |
| `nuc_shp_hu2_log_mean` | Log10 magnitude of Hu invariant moment |
| `nuc_shp_hu2_log_std` | Log10 magnitude of Hu invariant moment |
| `nuc_shp_hu3_log_mean` | Log10 magnitude of Hu invariant moment |
| `nuc_shp_hu3_log_std` | Log10 magnitude of Hu invariant moment |

## Chromatin (48)

| feature | meaning |
| --- | --- |
| `nuc_chr_h_mean_mean` | Mean hematoxylin OD inside the nucleus (chromatin content) |
| `nuc_chr_h_mean_std` | Mean hematoxylin OD inside the nucleus (chromatin content) |
| `nuc_chr_h_std_mean` | Std of hematoxylin OD inside the nucleus |
| `nuc_chr_h_std_std` | Std of hematoxylin OD inside the nucleus |
| `nuc_chr_h_cv_mean` | Chromatin intensity coefficient of variation |
| `nuc_chr_h_cv_std` | Chromatin intensity coefficient of variation |
| `nuc_chr_h_skew_mean` | Skewness of intranuclear hematoxylin OD |
| `nuc_chr_h_skew_std` | Skewness of intranuclear hematoxylin OD |
| `nuc_chr_h_kurtosis_mean` | Kurtosis of intranuclear hematoxylin OD |
| `nuc_chr_h_kurtosis_std` | Kurtosis of intranuclear hematoxylin OD |
| `nuc_chr_h_p10_mean` | 10th percentile of intranuclear hematoxylin OD |
| `nuc_chr_h_p10_std` | 10th percentile of intranuclear hematoxylin OD |
| `nuc_chr_h_p90_mean` | 90th percentile of intranuclear hematoxylin OD |
| `nuc_chr_h_p90_std` | 90th percentile of intranuclear hematoxylin OD |
| `nuc_chr_h_entropy_mean` | Shannon entropy of intranuclear hematoxylin OD |
| `nuc_chr_h_entropy_std` | Shannon entropy of intranuclear hematoxylin OD |
| `nuc_chr_contrast_ring_mean` | Nuclear minus surrounding cytoplasm hematoxylin OD |
| `nuc_chr_contrast_ring_std` | Nuclear minus surrounding cytoplasm hematoxylin OD |
| `nuc_chr_hyperchromatic_frac_mean` | Fraction of nuclear pixels above mean + 1 SD |
| `nuc_chr_hyperchromatic_frac_std` | Fraction of nuclear pixels above mean + 1 SD |
| `nuc_chr_clump_count_mean` | Number of chromatin clumps |
| `nuc_chr_clump_count_std` | Number of chromatin clumps |
| `nuc_chr_clump_mean_area_mean` | Mean chromatin clump area |
| `nuc_chr_clump_mean_area_std` | Mean chromatin clump area |
| `nuc_chr_margination_mean` | Rim to core hematoxylin ratio (chromatin margination) |
| `nuc_chr_margination_std` | Rim to core hematoxylin ratio (chromatin margination) |
| `nuc_chr_internal_gradient_mean` | Mean Sobel gradient inside the nucleus |
| `nuc_chr_internal_gradient_std` | Mean Sobel gradient inside the nucleus |
| `nuc_chr_glcm_contrast_mean` | Masked co-occurrence texture using only intranuclear pixel pairs |
| `nuc_chr_glcm_contrast_std` | Masked co-occurrence texture using only intranuclear pixel pairs |
| `nuc_chr_glcm_homogeneity_mean` | Masked co-occurrence texture using only intranuclear pixel pairs |
| `nuc_chr_glcm_homogeneity_std` | Masked co-occurrence texture using only intranuclear pixel pairs |
| `nuc_chr_glcm_energy_mean` | Masked co-occurrence texture using only intranuclear pixel pairs |
| `nuc_chr_glcm_energy_std` | Masked co-occurrence texture using only intranuclear pixel pairs |
| `nuc_chr_glcm_entropy_mean` | Masked co-occurrence texture using only intranuclear pixel pairs |
| `nuc_chr_glcm_entropy_std` | Masked co-occurrence texture using only intranuclear pixel pairs |
| `nuc_chr_glcm_correlation_mean` | Masked co-occurrence texture using only intranuclear pixel pairs |
| `nuc_chr_glcm_correlation_std` | Masked co-occurrence texture using only intranuclear pixel pairs |
| `nuc_chr_lbp_entropy_mean` | Uniform local binary pattern statistic |
| `nuc_chr_lbp_entropy_std` | Uniform local binary pattern statistic |
| `nuc_chr_lbp_nonuniform_frac_mean` | Tissue class area fraction |
| `nuc_chr_lbp_nonuniform_frac_std` | Tissue class area fraction |
| `nuc_chr_blue_ratio_mean` | Blue share of nuclear colour |
| `nuc_chr_blue_ratio_std` | Blue share of nuclear colour |
| `nuc_chr_eosin_cytoplasm_mean` | Eosin OD (cytoplasm ring or nucleus) |
| `nuc_chr_eosin_cytoplasm_std` | Eosin OD (cytoplasm ring or nucleus) |
| `nuc_chr_eosin_nucleus_mean` | Eosin OD (cytoplasm ring or nucleus) |
| `nuc_chr_eosin_nucleus_std` | Eosin OD (cytoplasm ring or nucleus) |

## Cell graph (36)

| feature | meaning |
| --- | --- |
| `graph_delaunay_edge_mean` | Delaunay graph statistic (edge lengths, triangle regularity, degree) |
| `graph_delaunay_edge_std` | Delaunay graph statistic (edge lengths, triangle regularity, degree) |
| `graph_delaunay_edge_cv` | Delaunay graph statistic (edge lengths, triangle regularity, degree) |
| `graph_delaunay_edge_p90` | Delaunay graph statistic (edge lengths, triangle regularity, degree) |
| `graph_delaunay_long_edge_frac` | Delaunay graph statistic (edge lengths, triangle regularity, degree) |
| `graph_delaunay_regularity_mean` | Delaunay graph statistic (edge lengths, triangle regularity, degree) |
| `graph_delaunay_regularity_std` | Delaunay graph statistic (edge lengths, triangle regularity, degree) |
| `graph_delaunay_degree_mean` | Delaunay graph statistic (edge lengths, triangle regularity, degree) |
| `graph_delaunay_degree_std` | Delaunay graph statistic (edge lengths, triangle regularity, degree) |
| `graph_voronoi_area_mean` | Voronoi cell area disorder and compactness |
| `graph_voronoi_area_cv` | Voronoi cell area disorder and compactness |
| `graph_voronoi_area_skew` | Voronoi cell area disorder and compactness |
| `graph_voronoi_log_area_std` | Voronoi cell area disorder and compactness |
| `graph_voronoi_compactness_mean` | Voronoi cell area disorder and compactness |
| `graph_mst_edge_mean` | Minimum spanning tree statistic |
| `graph_mst_edge_cv` | Minimum spanning tree statistic |
| `graph_mst_branch_frac` | Minimum spanning tree statistic |
| `graph_mst_leaf_frac` | Minimum spanning tree statistic |
| `graph_clark_evans_R` | Clark Evans nearest neighbour index (<1 clustered, >1 regular) or z score |
| `graph_clark_evans_z` | Clark Evans nearest neighbour index (<1 clustered, >1 regular) or z score |
| `graph_ripley_L_minus_r_8` | Ripley L(r) - r; >0 clustering at radius r, <0 regularity |
| `graph_ripley_L_minus_r_16` | Ripley L(r) - r; >0 clustering at radius r, <0 regularity |
| `graph_ripley_L_minus_r_32` | Ripley L(r) - r; >0 clustering at radius r, <0 regularity |
| `graph_ripley_L_minus_r_48` | Ripley L(r) - r; >0 clustering at radius r, <0 regularity |
| `graph_dbscan_n_clusters` | DBSCAN cluster count, noise fraction or dominant cluster share |
| `graph_dbscan_noise_frac` | DBSCAN cluster count, noise fraction or dominant cluster share |
| `graph_dbscan_largest_frac` | DBSCAN cluster count, noise fraction or dominant cluster share |
| `graph_morisita_index` | Morisita aggregation index of gridded nuclear counts |
| `graph_density_gini` | Gini or CV of gridded nuclear density |
| `graph_density_cv` | Gini or CV of gridded nuclear density |
| `graph_orientation_order` | Nematic order or local alignment of nuclear long axes |
| `graph_orientation_local_alignment` | Nematic order or local alignment of nuclear long axes |
| `graph_moran_area` | Moran's I spatial autocorrelation of an attribute |
| `graph_mixing_same_frac` | Neighbour mixing between large and small nuclei |
| `graph_mixing_excess` | Neighbour mixing between large and small nuclei |
| `graph_moran_hematoxylin` | Moran's I spatial autocorrelation of an attribute |

## Multiscale texture (30)

| feature | meaning |
| --- | --- |
| `tex_gabor_f0.08_energy` | Gabor filter bank energy, anisotropy or orientation entropy |
| `tex_gabor_f0.08_anisotropy` | Gabor filter bank energy, anisotropy or orientation entropy |
| `tex_gabor_f0.16_energy` | Gabor filter bank energy, anisotropy or orientation entropy |
| `tex_gabor_f0.16_anisotropy` | Gabor filter bank energy, anisotropy or orientation entropy |
| `tex_gabor_f0.32_energy` | Gabor filter bank energy, anisotropy or orientation entropy |
| `tex_gabor_f0.32_anisotropy` | Gabor filter bank energy, anisotropy or orientation entropy |
| `tex_gabor_orientation_entropy` | Nematic order or local alignment of nuclear long axes |
| `tex_lbp_P8R1_entropy` | Uniform local binary pattern statistic |
| `tex_lbp_P8R1_nonuniform` | Uniform local binary pattern statistic |
| `tex_lbp_P16R2_entropy` | Uniform local binary pattern statistic |
| `tex_lbp_P16R2_nonuniform` | Uniform local binary pattern statistic |
| `tex_lbp_P24R3_entropy` | Uniform local binary pattern statistic |
| `tex_lbp_P24R3_nonuniform` | Uniform local binary pattern statistic |
| `tex_h_od_mean` | Optical density statistic from colour deconvolution |
| `tex_h_od_std` | Optical density statistic from colour deconvolution |
| `tex_e_od_mean` | Optical density statistic from colour deconvolution |
| `tex_e_od_std` | Optical density statistic from colour deconvolution |
| `tex_he_ratio` | Hematoxylin to eosin OD ratio |
| `tex_h_positive_area` | Fraction of tile that is hematoxylin positive |
| `tex_colourfulness` | Hasler and Susstrunk colourfulness |
| `tex_saturation_mean` | Mean HSV saturation of the tile |
| `tex_value_mean` | Mean HSV value (brightness) of the tile |
| `tex_intensity_entropy` | Shannon entropy of the gray image |
| `tex_gradient_mean` | Mean Sobel gradient magnitude of the tile |
| `tex_edge_density` | Canny edge pixel fraction |
| `tex_fractal_dimension` | Box counting fractal dimension of the nuclear foreground |
| `tex_fractal_r2` | Box counting fractal dimension of the nuclear foreground |
| `tex_lacunarity_r8` | Gliding box lacunarity of the nuclear foreground |
| `tex_lacunarity_r16` | Gliding box lacunarity of the nuclear foreground |
| `tex_lacunarity_r32` | Gliding box lacunarity of the nuclear foreground |

## Tissue architecture (28)

| feature | meaning |
| --- | --- |
| `arch_frac_tumor` | Tissue class area fraction |
| `arch_frac_stroma` | Tissue class area fraction |
| `arch_frac_necrosis` | Tissue class area fraction |
| `arch_frac_normal` | Tissue class area fraction |
| `arch_tumor_stroma_ratio` | Tumor to stroma area ratio |
| `arch_n_tumor_nests` | Tumor nest geometry |
| `arch_nest_area_mean` | Tumor nest geometry |
| `arch_nest_largest_frac` | Tumor nest geometry |
| `arch_nest_solidity_mean` | Tumor nest geometry |
| `arch_nest_fragmentation` | Tumor nest geometry |
| `arch_tumor_perimeter_density` | Tumor rim pixels per tile pixel |
| `arch_interface_frac_stroma` | Share of tumor rim adjacent to the named tissue class |
| `arch_interface_frac_necrosis` | Share of tumor rim adjacent to the named tissue class |
| `arch_interface_frac_normal` | Share of tumor rim adjacent to the named tissue class |
| `arch_nuc_density_tumor` | Nuclear density in the named tissue region |
| `arch_nuc_area_tumor` | Mean nuclear area in the named tissue region |
| `arch_nuc_density_stroma` | Nuclear density in the named tissue region |
| `arch_nuc_area_stroma` | Mean nuclear area in the named tissue region |
| `arch_dens_tumor_core` | Nuclear density in a signed distance band from the tumor boundary |
| `arch_dens_tumor_edge` | Nuclear density in a signed distance band from the tumor boundary |
| `arch_dens_front` | Nuclear density in a signed distance band from the tumor boundary |
| `arch_dens_far` | Nuclear density in a signed distance band from the tumor boundary |
| `arch_front_to_core_ratio` | Invasive front to core or far field density ratio |
| `arch_front_to_far_ratio` | Invasive front to core or far field density ratio |
| `arch_nuc_density_necrosis` | Nuclear density in the named tissue region |
| `arch_nuc_area_necrosis` | Mean nuclear area in the named tissue region |
| `arch_nuc_density_normal` | Nuclear density in the named tissue region |
| `arch_nuc_area_normal` | Mean nuclear area in the named tissue region |

