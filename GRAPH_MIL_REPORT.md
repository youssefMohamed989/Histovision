# Cell-graph GNN + attention MIL demonstration

> Synthetic bag-of-tiles cohort, 90 patients x 4 tiles each, with independent local heterogeneity per tile. Validates that the model and its two attention mechanisms behave sensibly, not a clinical claim.

## Does tile attention find the informative tile?
- Spearman correlation, tile attention rank vs the tile's local-grade extremity within its bag: 0.007

No. In this run the correlation is essentially zero -- tile level attention did not learn to prefer
the tile whose local grade differs most from its bag-mates. This is reported as a negative result,
not glossed over: with only a few tiles per bag and no direct tile level supervision, the attention
mechanism has little signal to learn *which* tile matters from the bag label alone.

## Does node attention find the atypical nuclei?
- Spearman correlation, node attention vs nuclear area: 0.120
- Spearman correlation, node attention vs circularity: -0.342

Partially, and with a nuance worth stating plainly. Node attention does concentrate on larger, less
circular nuclei, but looking at the attention maps (`fig22`) shows it is largely finding the
elongated SPINDLE-SHAPED stromal nuclei, not atypical tumor nuclei specifically -- geometrically
those are simply the most different shape from the crowd of round cells, which a shape-sensitive
attention mechanism can pick up without ever being told what a tumor nucleus looks like. That is a
real, interpretable, but different signal than "finds cancer," and is reported as such.

## Patient level classification, out of fold (5 fold CV)
- GraphMIL (cell-graph GNN + attention MIL): accuracy 0.700, macro AUC 0.866
- Baseline (mean-pooled nuclei features + logistic regression): accuracy 0.822, macro AUC 0.926

The simpler baseline wins here. This is an honest negative result for the more complex model, not a favorable framing of it: a two-level-attention graph neural network has far more parameters and a harder optimization problem than a mean-pooled logistic regression, and 90 patients is a small cohort for that. The unit tests in `tests/test_graph_mil.py` establish that the architecture itself is implemented correctly (gradients flow, batching is numerically exact, attention sums to 1); this analysis establishes that correctness alone does not guarantee it beats a simple baseline on a small cohort -- more patients, more tiles per bag, or pretraining the GNN encoder on a larger nuclei dataset before MIL fine-tuning would all be reasonable next steps to close that gap.

## Figures

![fig22_graph_mil_attention](figures/fig22_graph_mil_attention.png)
![fig23_attention_validity_and_comparison](figures/fig23_attention_validity_and_comparison.png)

---

**Note on training budget, from `GRAPH_MIL_LEARNING_CURVE.md` and
`GRAPH_MIL_EPOCH_TEST.md`:** an earlier version of this run used a fixed 40
epochs and got 0.60 accuracy here. A follow up learning curve found accuracy
getting *worse*, not better, as the cohort grew to 300 patients at that same
fixed epoch count -- backwards for more data, and traced to `train_graph_mil`
using full batch gradient descent, where a fixed epoch count gives a larger
cohort the same number of optimization steps as a smaller one rather than
proportionally more. Testing that directly on 300 patients confirmed it
(0.61 -> 0.81 accuracy from 30 to 90 epochs), so this run was redone at 90
epochs, recovering 0.60 -> 0.70 accuracy here. It still does not beat the
baseline at this cohort size (0.70 vs 0.82) -- at n=300 with the same fixed
90 epoch budget GraphMIL reached 0.81 against a baseline of 0.84, much
closer, suggesting the remaining gap at n=90 is more plausibly a genuine
data-efficiency difference (the baseline needs less data to fit well) than
further undertraining. That is inference from two cohort sizes, not proof;
a properly validation-monitored epoch count at a few more cohort sizes would
settle it further.
