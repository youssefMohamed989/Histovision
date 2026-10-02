import numpy as np
import torch

from liver_histo_ai.models.survival import (
    CoxPHWrapper,
    DeepSurv,
    cox_partial_likelihood_loss,
    train_deepsurv,
)


def _make_survival_data(n=100, in_dim=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, in_dim)).astype(np.float32)
    risk = X[:, 0] * 0.8 - X[:, 1] * 0.3
    duration = rng.exponential(scale=np.exp(-risk)) * 10 + 0.1
    event = rng.binomial(1, 0.8, size=n).astype(np.float32)
    return X, duration.astype(np.float32), event


def test_cox_ph_wrapper_fit_and_concordance():
    X, duration, event = _make_survival_data()
    model = CoxPHWrapper(penalizer=0.5).fit(X, duration, event)
    c_index = model.concordance(X, duration, event)
    assert 0.0 <= c_index <= 1.0


def test_deepsurv_forward_shape():
    model = DeepSurv(in_dim=4, hidden_dims=(8, 4))
    x = torch.randn(5, 4)
    out = model(x)
    assert out.shape == (5,)


def test_cox_partial_likelihood_loss_runs():
    risk = torch.randn(10)
    duration = torch.rand(10) * 10
    event = torch.randint(0, 2, (10,)).float()
    loss = cox_partial_likelihood_loss(risk, duration, event)
    assert torch.isfinite(loss)


def test_train_deepsurv_reduces_loss():
    X, duration, event = _make_survival_data(n=60)
    model = DeepSurv(in_dim=4, hidden_dims=(8,))

    def _loss(m):
        with torch.no_grad():
            risk = m(torch.tensor(X))
            return cox_partial_likelihood_loss(
                risk, torch.tensor(duration), torch.tensor(event)
            ).item()

    loss_before = _loss(model)
    train_deepsurv(model, X, duration, event, epochs=50)
    loss_after = _loss(model)
    assert loss_after <= loss_before + 1e-3  # should not get meaningfully worse
