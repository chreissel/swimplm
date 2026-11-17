"""SparKer and the toy runner reproduce the original phlab-neurips25 code exactly."""

import numpy as np
import torch

from swimplm import SparKer, run_toys

legacy = __import__("data.data_utils", fromlist=["sparkKer"])  # original implementation


def _seed(s):
    np.random.seed(s)
    torch.manual_seed(s + 1000)


def _sample(n, d, shift=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64) + shift
    mass = 100 + 10 * torch.randn(n, 1, generator=g, dtype=torch.float64)
    return torch.cat((x, mass), dim=1)  # last column plays the role of the mass


def test_single_fit_matches_sparkKer():
    data, ref = _sample(150, 3, shift=0.3, seed=1), _sample(400, 3, seed=2)
    w_data = torch.ones(len(data))
    w_ref = torch.full((len(ref),), 150 / 400, dtype=torch.float64)

    _seed(7)
    t_legacy = legacy.sparkKer(data, ref, None, -1, w_data, w_ref, plot=False,
                               splitmass=True, iWidth_init=[1.5])
    _seed(7)
    t_new = SparKer(widths=(1.5,)).fit(data[:, :-1], ref[:, :-1], w_data, w_ref).t

    np.testing.assert_allclose(t_new, t_legacy, rtol=1e-10)


def test_toys_match_run_realistic_toy():
    g = torch.Generator().manual_seed(3)
    data = torch.cat((_sample(300, 3, seed=4), _sample(60, 3, shift=1.0, seed=5)))
    data_labels = torch.cat((torch.zeros(300), torch.ones(60)))
    data_weights = torch.cat((torch.full((300,), 0.5), torch.full((60,), 0.25))).double()
    ref = _sample(1500, 3, seed=6)
    ref_labels = torch.zeros(1500)
    ref_weights = (0.5 + torch.rand(1500, generator=g)).double()

    _seed(11)
    t_alt_legacy, t_null_legacy = legacy.run_realistic_toy(
        400, data, data_labels, ref, ref_labels, 1,
        data_weights=data_weights, model_weights=ref_weights,
        ntoys=2, plot=False, width=2.0)
    _seed(11)
    t_alt, t_null = run_toys(SparKer(widths=(2.0,)), data, data_labels, data_weights,
                             ref, ref_labels, ref_weights, sig_label=1, n_ref=400,
                             n_toys=2, verbose=False)

    np.testing.assert_allclose(t_alt, t_alt_legacy, rtol=1e-10)
    np.testing.assert_allclose(t_null, t_null_legacy, rtol=1e-10)
