import numpy as np
from scipy import ndimage as ndi


def per_region_recall(pred, tile, iou=0.5):
    gt = tile["instances"]
    region_of = tile["nucleus_region"]
    hits = {r: [] for r in range(4)}
    for gid, sl in enumerate(ndi.find_objects(gt), start=1):
        g = gt[sl] == gid
        ov = pred[sl][g]
        ov = ov[ov > 0]
        ok = False
        if ov.size:
            pid = np.bincount(ov).argmax()
            inter = (ov == pid).sum()
            ok = inter / (g.sum() + (pred == pid).sum() - inter) >= iou
        hits[int(region_of[gid])].append(ok)
    return {r: float(np.mean(v)) if v else float("nan") for r, v in hits.items()}
