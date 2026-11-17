"""Checks of the toy options added on top of the original run_realistic_toy."""

import numpy as np
import torch

from swimplm import NPLMResult, run_toys
from swimplm.toys import IsolatedRandomness


class Recorder:
    """Fake test that records its inputs; optionally uses numpy's global RNG like SparKer."""

    def __init__(self, consume_rng=False):
        self.calls, self.consume_rng = [], consume_rng

    def fit(self, data, ref, w_data=None, w_ref=None):
        if self.consume_rng:
            np.random.randint(10, size=30)
        self.calls.append((data.clone(), ref.clone(), w_ref.clone()))
        return NPLMResult(t=0.0)


def _inputs():
    g = torch.Generator().manual_seed(0)
    data = torch.randn(500, 3, generator=g, dtype=torch.float64)
    labels = (torch.arange(500) < 50).float()  # first 50 events are signal (label 1)
    ref = torch.randn(2000, 3, generator=g, dtype=torch.float64)
    return dict(data=data, data_labels=labels, data_weights=torch.ones(500, dtype=torch.float64),
                ref=ref, ref_labels=torch.zeros(2000), ref_weights=torch.ones(2000, dtype=torch.float64))


def _run(test, **kw):
    np.random.seed(3)
    torch.manual_seed(3)
    run_toys(test, **_inputs(), sig_label=1, n_ref=300, n_toys=3, drop_last_column=False,
             verbose=False, **kw)
    return test


def test_isolated_randomness_gives_identical_toys():
    plain = _run(Recorder())
    wrapped = _run(IsolatedRandomness(Recorder(consume_rng=True))).test
    unwrapped = _run(Recorder(consume_rng=True))
    for a, b in zip(plain.calls, wrapped.calls):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # without the wrapper, the RNG use inside fit changes later toys
    assert not all(torch.equal(a[0], b[0]) for a, b in zip(plain.calls, unwrapped.calls))


def test_yield_overrides_and_reference_normalisation():
    rec = _run(Recorder(), n_sig=0.0, n_bkg=100.0)
    for i in range(0, len(rec.calls), 2):  # alternative toys
        data, ref, w_ref = rec.calls[i]
        # sum of reference weights is n_bkg on average (Poisson-fluctuated n_ref)
        assert 60 < float(w_ref.sum()) < 140
    assert all(len(rec.calls[i][0]) < 160 for i in range(0, len(rec.calls), 2))


def test_signal_scale_zero_removes_signal():
    data = _inputs()["data"]
    signal = {tuple(r.tolist()) for r in data[:50]}
    rec = _run(Recorder(), signal_scale=0.0)
    for i in range(0, len(rec.calls), 2):
        assert not any(tuple(r.tolist()) in signal for r in rec.calls[i][0])
