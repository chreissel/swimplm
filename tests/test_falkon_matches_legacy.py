"""Falkon NPLM reproduces the original phlab-neurips25 code exactly."""

import numpy as np
import pytest

pytest.importorskip("falkon")

import utils.FLKutils as flk  # original implementation  # noqa: E402
import utils.GENutils as gen  # noqa: E402

from swimplm import Falkon  # noqa: E402
from swimplm.methods.falkon import candidate_widths  # noqa: E402


def _samples():
    rng = np.random.default_rng(0)
    data = rng.normal(0.2, 1.0, size=(400, 3))
    ref = rng.normal(0.0, 1.0, size=(1600, 3))
    return data, ref


def test_fit_matches_trainer_and_compute_t():
    data, ref = _samples()
    w = 400 / 1600
    sigma, M, lam, iters, seed = 1.7, 150, 1e-7, 1_000_000, 42

    x = np.concatenate((data, ref)).astype(float)
    y = np.concatenate((np.ones((len(data), 1)), np.zeros((len(ref), 1))))
    config = flk.get_logflk_config(M, sigma, [lam], weight=w, iter=[iters], seed=seed, cpu=True)
    preds = flk.trainer(x, y, config)
    t_legacy = flk.compute_t(preds, y, w)

    result = Falkon(sigma=sigma, M=M, lam=lam, iterations=iters, seed=seed, use_cpu=True).fit(
        data, ref, w_ref=np.full(len(ref), w))

    np.testing.assert_allclose(result.t, t_legacy, rtol=1e-10)
    np.testing.assert_allclose(result.f_ref, preds[len(data):, 0], rtol=1e-10)


def test_candidate_widths_match_candidate_sigma():
    _, ref = _samples()
    perc = [1, 25, 50, 75, 99]
    expected = gen.candidate_sigma(ref[:2000, :], perc=perc)
    expected = np.append(expected, expected[-1] * 2)
    np.testing.assert_allclose(candidate_widths(ref, perc), expected)


def test_rejects_event_dependent_reference_weights():
    data, ref = _samples()
    with pytest.raises(ValueError, match="single reference weight"):
        Falkon(sigma=1.0, M=10).fit(data, ref, w_ref=np.linspace(0.1, 1, len(ref)))
