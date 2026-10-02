# Epoch budget test: is the learning-curve degradation a training artifact?

> Identical 300 patient bags and cross validation folds at every epoch setting below;
> only the training budget changes. Tests the hypothesis from `GRAPH_MIL_LEARNING_CURVE.md`.

| epochs | GraphMIL accuracy | GraphMIL macro AUC |
| --- | --- | --- |
| 30 | 0.607 | 0.741 |
| 90 | 0.813 | 0.932 |

(Baseline accuracy at this cohort size, for reference: 0.840, unaffected by GraphMIL's epoch count.)

**Hypothesis confirmed.** More epochs (90 vs the original 30) recovered accuracy (0.607 -> 0.813), supporting that the n=300 result in the learning curve was undertrained, not a genuine architecture scaling failure. The fixed-epoch comparison there should be rerun with epochs tuned per cohort size (or minibatch SGD) for a fair scaling claim.
