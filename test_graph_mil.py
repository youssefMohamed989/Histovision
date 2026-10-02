import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from liver_histo_ai.features.morphology import extract_morphology_features
from liver_histo_ai.models.attention_mil import GatedAttentionMIL, train_mil
from liver_histo_ai.models.cell_gnn import (
    NODE_FEATURE_COLUMNS,
    CellGAT,
    build_cell_graph,
    collate_graphs,
    scatter_softmax,
)
from liver_histo_ai.models.graph_mil import GraphMIL, train_graph_mil
from liver_histo_ai.segmentation.classical import segment_nuclei_classical
from liver_histo_ai.synthetic import generate_tile


def _nuclei_table(seed=0, size=160, grade=1):
    d = generate_tile(size, np.random.default_rng(seed), grade=grade, weights=(1, 0, 0, 0))
    inst = segment_nuclei_classical(d["rgb"])
    return extract_morphology_features(inst)


# ---------------------------------------------------------------- scatter softmax
def test_scatter_softmax_matches_dense_softmax_per_group():
    scores = torch.tensor([1.0, 2.0, 0.5, -1.0, 3.0])
    index = torch.tensor([0, 0, 1, 1, 1])
    out = scatter_softmax(scores, index, num_segments=2)
    expected0 = F.softmax(scores[:2], dim=0)
    expected1 = F.softmax(scores[2:], dim=0)
    assert torch.allclose(out[:2], expected0, atol=1e-6)
    assert torch.allclose(out[2:], expected1, atol=1e-6)


def test_scatter_softmax_sums_to_one_per_group():
    torch.manual_seed(0)
    scores = torch.randn(37)
    index = torch.randint(0, 5, (37,))
    out = scatter_softmax(scores, index, num_segments=5)
    for g in range(5):
        mask = index == g
        if mask.any():
            assert torch.allclose(out[mask].sum(), torch.tensor(1.0), atol=1e-5)


def test_scatter_softmax_handles_empty_group():
    scores = torch.tensor([1.0, 2.0])
    index = torch.tensor([0, 0])
    out = scatter_softmax(scores, index, num_segments=3)  # group 1, 2 empty
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------- cell graph building
def test_build_cell_graph_shapes():
    table = _nuclei_table()
    g = build_cell_graph(table)
    assert g is not None
    assert g.x.shape == (len(table), len(NODE_FEATURE_COLUMNS))
    assert g.edge_index.shape[0] == 2
    assert g.edge_attr.shape == (g.edge_index.shape[1], 2)
    # edges are symmetric (both directions present)
    assert g.edge_index.shape[1] % 2 == 0


def test_build_cell_graph_too_few_nuclei_returns_none():
    table = pd.DataFrame({"centroid_row": [1.0, 2.0], "centroid_col": [1.0, 2.0],
                          "area": [10.0, 12.0], "eccentricity": [0.1, 0.2],
                          "solidity": [0.9, 0.9], "extent": [0.7, 0.7],
                          "circularity": [0.8, 0.8], "axis_ratio": [1.1, 1.2]})
    assert build_cell_graph(table) is None


def test_build_cell_graph_node_features_are_standardized():
    table = _nuclei_table()
    g = build_cell_graph(table)
    assert torch.allclose(g.x.mean(0), torch.zeros(g.x.shape[1]), atol=1e-4)


# ---------------------------------------------------------------- CellGAT
def test_cellgat_forward_shapes_single_graph():
    table = _nuclei_table()
    g = build_cell_graph(table)
    batch = collate_graphs([g])
    model = CellGAT(embed_dim=24, num_classes=3)
    out = model(batch)
    assert out["embedding"].shape == (1, 24)
    assert out["logits"].shape == (1, 3)
    assert out["node_attention"].shape == (g.num_nodes,)
    assert torch.allclose(out["node_attention"].sum(), torch.tensor(1.0), atol=1e-4)


def test_cellgat_batching_matches_looped_single_graphs():
    """A batch of 3 graphs must give the same per graph output as running each alone
    (up to floating point tolerance) -- this is the key correctness property of the
    block diagonal batching trick."""
    torch.manual_seed(0)
    graphs = [build_cell_graph(_nuclei_table(seed=s)) for s in (1, 2, 3)]
    model = CellGAT(embed_dim=16, num_classes=3)
    model.eval()
    batched = model(collate_graphs(graphs))
    for i, g in enumerate(graphs):
        single = model(collate_graphs([g]))
        assert torch.allclose(batched["embedding"][i], single["embedding"][0], atol=1e-4)
        assert torch.allclose(batched["logits"][i], single["logits"][0], atol=1e-4)


def test_cellgat_gradients_flow_to_all_parameters():
    table = _nuclei_table()
    g = build_cell_graph(table)
    model = CellGAT(embed_dim=16, num_classes=3)
    out = model(collate_graphs([g]))
    loss = out["logits"].sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all()


def test_cellgat_embed_single_helper():
    table = _nuclei_table()
    g = build_cell_graph(table)
    model = CellGAT(embed_dim=8)
    emb, attn = model.embed_single(g)
    assert emb.shape == (8,)
    assert attn.shape == (g.num_nodes,)
    assert np.isclose(attn.sum(), 1.0, atol=1e-4)


def test_cellgat_node_attention_is_permutation_equivariant():
    """Shuffling node order should shuffle the attention weights the same way,
    not change which underlying nucleus gets which weight."""
    table = _nuclei_table()
    g = build_cell_graph(table)
    model = CellGAT(embed_dim=12)
    model.eval()
    _, attn = model.embed_single(g)

    perm = np.random.default_rng(0).permutation(g.num_nodes)
    inv = np.argsort(perm)
    remap = {old: new for new, old in enumerate(perm)}
    ei = g.edge_index.numpy()
    shuffled_edges = np.vectorize(remap.get)(ei)
    from liver_histo_ai.models.cell_gnn import CellGraphData
    g2 = CellGraphData(x=g.x[perm], edge_index=torch.as_tensor(shuffled_edges),
                       edge_attr=g.edge_attr.clone(), pos=g.pos[perm])
    # reorder edge_attr to match the (already-shuffled-index) edges by re-deriving
    # which original edge each shuffled edge corresponds to is unnecessary here:
    # edge_attr values themselves don't depend on labels, only edge_index does,
    # and we kept the same edge_attr order alongside the same edge list order.
    _, attn2 = model.embed_single(g2)
    assert np.allclose(attn2[inv], attn, atol=1e-4)


# ---------------------------------------------------------------- Attention MIL
def test_gated_attention_mil_forward_shapes():
    torch.manual_seed(0)
    bags = [torch.randn(5, 10), torch.randn(3, 10), torch.randn(8, 10)]
    model = GatedAttentionMIL(in_dim=10, num_classes=3)
    out = model(bags)
    assert out["logits"].shape == (3, 3)
    assert len(out["attention"]) == 3
    for w, b in zip(out["attention"], bags, strict=True):
        assert w.shape == (b.shape[0],)
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)


def test_gated_attention_mil_predict_bag():
    model = GatedAttentionMIL(in_dim=6, num_classes=3)
    probs, attn = model.predict_bag(np.random.default_rng(0).normal(size=(4, 6)).astype(np.float32))
    assert probs.shape == (3,) and np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert attn.shape == (4,)


def test_train_mil_reduces_training_loss():
    rng = np.random.default_rng(0)
    bags, y = [], []
    for _ in range(40):
        label = rng.integers(0, 2)
        n = rng.integers(3, 7)
        signal = np.full((n, 4), label * 3.0)
        bags.append((signal + rng.normal(0, 0.5, (n, 4))).astype(np.float32))
        y.append(label)
    model = GatedAttentionMIL(in_dim=4, hidden=16, num_classes=2)
    hist = train_mil(model, bags, np.array(y), epochs=80, lr=5e-3)
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    assert hist["train_acc"][-1] > 0.85


def test_mil_attention_upweights_the_informative_instance():
    """One instance in the bag carries the class signal, the rest are noise;
    after training, attention should concentrate on the informative one more
    than chance (1/n)."""
    rng = np.random.default_rng(1)
    n_per_bag = 6
    bags, y, informative_idx = [], [], []
    for _ in range(60):
        label = rng.integers(0, 2)
        instances = rng.normal(0, 0.3, (n_per_bag, 5)).astype(np.float32)
        idx = rng.integers(0, n_per_bag)
        instances[idx, 0] += 4.0 * (1 if label else -1)
        bags.append(instances)
        y.append(label)
        informative_idx.append(idx)
    model = GatedAttentionMIL(in_dim=5, hidden=16, num_classes=2)
    train_mil(model, bags, np.array(y), epochs=150, lr=5e-3)
    hits = 0
    for b, idx in zip(bags, informative_idx, strict=True):
        _, attn = model.predict_bag(b)
        hits += int(attn.argmax() == idx)
    assert hits / len(bags) > 1.5 / n_per_bag  # comfortably above chance (1/6)


# ---------------------------------------------------------------- joint GraphMIL
def _patient_bags(n_patients=6, tiles_per_patient=3, seed=0):
    rng = np.random.default_rng(seed)
    bags, y = [], []
    for p in range(n_patients):
        label = p % 3
        tiles = []
        for _ in range(tiles_per_patient):
            table = _nuclei_table(seed=int(rng.integers(0, 10_000)), size=128, grade=label)
            g = build_cell_graph(table)
            if g is not None:
                tiles.append(g)
        if len(tiles) >= 2:
            bags.append(tiles)
            y.append(label)
    return bags, np.array(y)


def test_graph_mil_forward_shapes_and_attention_sizes():
    bags, _y = _patient_bags(n_patients=6, tiles_per_patient=3)
    model = GraphMIL(node_in_dim=len(NODE_FEATURE_COLUMNS), num_classes=3)
    out = model(bags)
    assert out["logits"].shape == (len(bags), 3)
    for i, bag in enumerate(bags):
        assert out["tile_attention"][i].shape == (len(bag),)
        assert torch.allclose(out["tile_attention"][i].sum(), torch.tensor(1.0), atol=1e-4)
        n_nodes = sum(g.num_nodes for g in bag)
        assert out["node_attention"][i].shape == (n_nodes,)


def test_graph_mil_predict_patient():
    bags, _y = _patient_bags(n_patients=4, tiles_per_patient=2)
    model = GraphMIL(node_in_dim=len(NODE_FEATURE_COLUMNS), num_classes=3)
    result = model.predict_patient(bags[0])
    assert result["probs"].shape == (3,)
    assert np.isclose(result["probs"].sum(), 1.0, atol=1e-4)
    assert result["tile_attention"].shape == (len(bags[0]),)


def test_graph_mil_end_to_end_gradients_and_training_runs():
    bags, y = _patient_bags(n_patients=9, tiles_per_patient=3)
    model = GraphMIL(node_in_dim=len(NODE_FEATURE_COLUMNS), embed_dim=16, num_classes=3)
    hist = train_graph_mil(model, bags, y, epochs=5)
    assert len(hist["train_loss"]) == 5
    assert all(np.isfinite(v) for v in hist["train_loss"])
    for p in model.parameters():
        assert p.grad is not None
