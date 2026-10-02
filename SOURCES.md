# Sources of the real sample images in this folder

These are genuine microscopy images used only to demonstrate and validate
the pipeline code (`analysis/run_real_data_validation.py`) on real tissue,
not synthetic data. None are hepatocellular/liver tissue and none carry
nuclei or tissue ground truth annotations -- the network available when
this repository was built could reach GitHub and PyPI but not the GDC/TCIA
servers that host real TCGA-LIHC liver slides, so those still need to be
pulled separately with `scripts/download_tcga_lihc.py` (which does work
against the public GDC API once you have normal internet access).

## `target.png`, `source.png`

Two real H&E stained tumor tissue slides used as the standard test pair for
stain normalization in the **torchstain** library.

- Source: https://github.com/EIDOSLAB/torchstain (`data/target.png`,
  `data/source.png`)
- License: MIT (torchstain repository, Copyright (c) EIDOSLAB). Keep that MIT notice with
  these two images if you redistribute them. The images' own provenance is described by
  the torchstain project.
- Used here for: real nuclei segmentation, and a genuine cross slide stain
  normalization test (fit a Macenko/Reinhard reference on one real slide,
  normalize the other real slide toward it).

## `skimage_ihc.png`

A real immunohistochemistry (DAB + hematoxylin counterstain) image of human
colon tissue.

- Source: bundled with `scikit-image` as `skimage.data.immunohistochemistry()`
  (no network fetch required -- it ships with the package)
- License / copyright: the scikit-image docstring for this dataset states "No known
  copyright restrictions" (image acquired at the Center for Microscopy And Molecular
  Imaging, CMMI). The scikit-image package itself is BSD-3-Clause.
- Used here as a stress test: it is not standard H&E (brown DAB chromogen
  instead of pink eosin), which is a fair, deliberately harder case for the
  hematoxylin colour deconvolution step.

## Regenerating this folder

```bash
mkdir -p data/real_samples
curl -sL -o data/real_samples/target.png \
    https://raw.githubusercontent.com/EIDOSLAB/torchstain/main/data/target.png
curl -sL -o data/real_samples/source.png \
    https://raw.githubusercontent.com/EIDOSLAB/torchstain/main/data/source.png
python3 -c "
from PIL import Image
from skimage.data import immunohistochemistry
Image.fromarray(immunohistochemistry()).save('data/real_samples/skimage_ihc.png')
"
```
