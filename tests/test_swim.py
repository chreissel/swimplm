"""Checks of the SWIM NPLM test (no legacy counterpart exists)."""

import numpy as np
import pytest
from scipy.optimize import minimize

from swimplm import Swim


def _samples(shift=0.0, n_data=300, n_ref=1200, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=(n_data, 3))
    data[: n_data // 5] += shift  # a localised excess
    ref = rng.normal(0.0, 1.0, size=(n_ref, 3))
    return data, ref, np.ones(n_data), np.full(n_ref, n_data / n_ref)


def test_newton_reaches_the_convex_optimum():
    data, ref, w_d, w_r = _samples(shift=1.5)
    test = Swim(layer_widths=(40,), l2=1e-3)
    result = test.fit(data, ref, w_d, w_r)

    phi = Swim.features(result.info["layers"], np.concatenate((data, ref)))
    phi_d, phi_r = phi[: len(data)], phi[len(data):]
    pen = np.full(phi.shape[1], test.l2)
    pen[-1] = 0

    def objective(b):
        return np.sum(w_r * (np.exp(phi_r @ b) - 1)) - w_d @ (phi_d @ b) + 0.5 * np.sum(pen * b**2)

    reference = minimize(objective, np.zeros(phi.shape[1]), method="L-BFGS-B",
                         options={"maxiter": 10000, "ftol": 1e-14, "gtol": 1e-10})
    assert objective(result.info["beta"]) <= reference.fun + 1e-6


def test_t_is_non_negative_and_deterministic():
    data, ref, w_d, w_r = _samples()
    t1 = Swim(layer_widths=(64,), random_seed=3).fit(data, ref, w_d, w_r).t
    t2 = Swim(layer_widths=(64,), random_seed=3).fit(data, ref, w_d, w_r).t
    assert t1 >= 0
    assert t1 == t2


@pytest.mark.parametrize("uniform", [False, True])
def test_signal_increases_t(uniform):
    test = Swim(layer_widths=(64,), sample_uniformly=uniform, l2=1e-4)
    t_null = [test.fit(*_samples(0.0, seed=s)).t for s in range(5)]
    t_sig = [test.fit(*_samples(2.5, seed=s)).t for s in range(5)]
    assert np.median(t_sig) > np.max(t_null)


def test_two_hidden_layers_run():
    data, ref, w_d, w_r = _samples(shift=1.0)
    result = Swim(layer_widths=(64, 32), activation="relu").fit(data, ref, w_d, w_r)
    assert np.isfinite(result.t)
    assert len(result.info["layers"]) == 2
