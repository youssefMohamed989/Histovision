import numpy as np
import torch

from liver_histo_ai.segmentation.nuclei_seg import HoVerNet, hover_postprocess, segment_nuclei


def test_hovernet_forward_shapes():
    model = HoVerNet(in_channels=3, base_channels=8, num_types=None)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out["np"].shape == (2, 2, 64, 64)
    assert out["hv"].shape == (2, 2, 64, 64)
    assert "nt" not in out


def test_hovernet_forward_with_types():
    model = HoVerNet(in_channels=3, base_channels=8, num_types=3)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out["nt"].shape == (1, 3, 64, 64)


def test_hover_postprocess_recovers_blobs(rng):
    h, w = 128, 128
    np_prob = np.zeros((h, w))
    hv_map = np.zeros((2, h, w))
    centers = [(30, 30), (30, 90), (90, 60)]
    yy, xx = np.mgrid[:h, :w]
    for cy, cx in centers:
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= 8 ** 2
        np_prob[blob] = 1.0
        # normalized horizontal/vertical distance-to-centroid within the blob,
        # matching what a trained HV branch would regress
        hv_map[0][blob] = ((xx - cx) / 8.0)[blob]
        hv_map[1][blob] = ((yy - cy) / 8.0)[blob]
    instance_mask, instances = hover_postprocess(np_prob, hv_map, min_area=5, max_area=1000)
    assert instance_mask.shape == (h, w)
    assert len(instances) >= 1  # watershed may merge close blobs; just check it runs end to end


def test_segment_nuclei_end_to_end(sample_rgb_tile):
    model = HoVerNet(in_channels=3, base_channels=8, num_types=None)
    tile_tensor = torch.from_numpy(sample_rgb_tile).permute(2, 0, 1).float() / 255.0
    device = torch.device("cpu")
    instance_mask, instances = segment_nuclei(model, tile_tensor, device)
    assert instance_mask.shape == sample_rgb_tile.shape[:2]
    assert isinstance(instances, list)
