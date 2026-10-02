# GraphMIL vs baseline: learning curve

> Does more training data close the gap found in `GRAPH_MIL_REPORT.md`, where a simple
> mean-pooled baseline beat the cell-graph GNN + attention MIL model on 90 synthetic patients?

| cohort size | GraphMIL acc | GraphMIL AUC | baseline acc | baseline AUC | gap (baseline - GraphMIL) |
| --- | --- | --- | --- | --- | --- |
| 90 | 0.633 | 0.802 | 0.811 | 0.927 | 0.178 |
| 180 | 0.622 | 0.797 | 0.844 | 0.956 | 0.222 |
| 300 | 0.557 | 0.749 | 0.840 | 0.963 | 0.283 |

**The gap widened, and it got worse in the wrong direction**: GraphMIL accuracy actually *decreased* as the cohort grew (0.633 -> 0.557 from 90 to 300 patients), while the baseline stayed flat or improved slightly. That is the opposite of what adding data should do to a model that is underperforming for lack of data, and it points to a different, more mundane explanation than "GraphMIL does not scale":

## A more likely explanation: fixed epochs, full batch gradient descent

`train_graph_mil` runs **one full batch gradient update per epoch** over every bag in the training
set, with the epoch count held fixed (30) across all three cohort sizes here. That means the number
of *optimization steps* was identical regardless of cohort size, while the loss surface the
optimizer has to fit grew more complex with more, more varied bags. In other words, this comparison
confounds cohort size with **effective training budget per bag**: the 300 patient run was very
likely undertrained relative to the 90 patient run, not fundamentally harder for the architecture to
fit given enough updates. A fair scaling test would hold total gradient steps (not epochs) constant,
or tune epochs/learning rate per cohort size with its own validation split, or move to minibatch SGD
so larger cohorts naturally get proportionally more updates. None of that was done here.

This is reported as what it is: a genuine, reproducible result from the exact code as configured,
and an important caveat about what that result does and does not show. It does **not** support
"GraphMIL fails to scale" as a general claim about the architecture; it supports "this specific
fixed-epoch training loop was not scaled correctly across cohort sizes," which is a bug in the
experiment design, not evidence about the model. Fixing the training loop to scale epochs (or use
minibatches) before re-running this comparison is the natural next step, left for a future revision
rather than quietly reworked here into a result that looks better.

## Figure

![learning curve](figures/fig24_graph_mil_learning_curve.png)
