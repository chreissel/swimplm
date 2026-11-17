"""Falkon: NPLM with a Nystrom kernel logistic regression (LogisticFalkon).

This is a cleaned-up port of the Falkon implementation in phlab-neurips25:

* ``utils/FLKutils.py``: ``get_logflk_config``, ``trainer``, ``compute_t``
* ``utils/NPLMutils.py``: ``NPLM_dist`` (scan over kernel widths)
* ``utils/GENutils.py``: ``candidate_sigma`` (widths from pairwise distances)

The Falkon configuration and the test statistic are identical to the
original. ``tests/test_falkon_matches_legacy.py`` checks this.

Model
-----
M Gaussian kernels of fixed width sigma, centred on M events drawn uniformly
from the training sample (Nystrom centres), with an L2 penalty lambda. The fit
uses a weighted logistic loss in which reference events carry the weight
``w_ref``, the luminosity ratio N_expected(bkg) / N_ref. The output f is the
log density ratio, and

    t = 2 * ( w_ref * sum_{x in R} (1 - exp(f(x)))  +  sum_{x in D} f(x) ).

Differences from the original ``NPLM_dist`` (documented, not copied):
  * The original loops over the *percentiles* (1, 25, 50, 75, 99) and passes
    them as sigma, instead of the widths it computed from them.
    ``width_scan`` uses the computed widths.
  * The original sets ``use_cpu=True`` exactly when CUDA *is* available.
    Here ``use_cpu`` is an explicit argument.
  * The original seeds Falkon from the wall clock. Here the seed is an argument.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.distance import pdist

from ..nplm import NPLMResult, default_weights, to_numpy


def candidate_widths(
    reference,
    percentiles: Sequence[float] = (1, 25, 50, 75, 99),
    n_max: int = 2000,
    add_broad: bool = True,
) -> np.ndarray:
    """Kernel widths at percentiles of the pairwise distances in ``reference``.

    Same as ``candidate_sigma`` (``utils/GENutils.py``) applied to the first
    ``n_max`` reference events, with the 2x-largest "very broad kernel"
    appended as in ``NPLM_dist``.
    """
    ref = to_numpy(reference)[:n_max]
    widths = np.around(np.percentile(pdist(ref), percentiles), 2)
    if add_broad:
        widths = np.append(widths, widths[-1] * 2)
    return widths


def logfalkon_config(sigma, M, lam, w_ref, iterations, seed, use_cpu, cg_tolerance):
    """LogisticFalkon keyword arguments (``get_logflk_config`` in FLKutils)."""
    from falkon.gsc_losses import WeightedCrossEntropyLoss
    from falkon.kernels import GaussianKernel
    from falkon.options import FalkonOptions

    return {
        "kernel": GaussianKernel(sigma=sigma),
        "M": M,
        "penalty_list": [lam],
        "iter_list": [iterations],
        "options": FalkonOptions(
            cg_tolerance=cg_tolerance, keops_active="no", use_cpu=use_cpu, debug=False
        ),
        "seed": seed,
        "loss": WeightedCrossEntropyLoss(kernel=GaussianKernel(sigma=sigma), neg_weight=w_ref),
    }


@dataclass
class Falkon:
    """NPLM test with LogisticFalkon.

    Attributes:
        sigma: Gaussian kernel width (see ``candidate_widths``).
        M: Number of Nystrom centres.
        lam: L2 penalty. The ``NPLM_dist`` default is 1e-7.
        iterations: Maximum conjugate-gradient iterations. The ``NPLM_dist``
            default is 1e7.
        seed: Seed for choosing the Nystrom centres. Use a different seed
            for each toy, as the original does.
        use_cpu: Run Falkon on the CPU.
        cg_tolerance: CG tolerance; the original uses sqrt(1e-7).
    """

    sigma: float
    M: int
    lam: float = 1e-7
    iterations: int = 10_000_000
    seed: int | None = None
    use_cpu: bool = not torch.cuda.is_available()
    cg_tolerance: float = float(np.sqrt(1e-7))

    def fit(self, data, ref, w_data=None, w_ref=None) -> NPLMResult:
        from falkon import LogisticFalkon

        data, ref = to_numpy(data), to_numpy(ref)
        w_data = default_weights(w_data, len(data))
        w_ref = default_weights(w_ref, len(ref))
        if not np.allclose(w_data, 1.0):
            raise ValueError("Falkon NPLM does not support data weights; data events must have weight 1.")
        if not np.allclose(w_ref, w_ref[0]):
            raise ValueError(
                "Falkon NPLM needs a single reference weight (the luminosity ratio). "
                "Unweight the reference first, e.g. run_toys(..., unweight_reference=True)."
            )
        w = float(w_ref[0])

        x = np.concatenate((data, ref), axis=0).astype(float)
        y = np.concatenate((np.ones((len(data), 1)), np.zeros((len(ref), 1))), axis=0)

        config = logfalkon_config(
            self.sigma, self.M, self.lam, w, int(self.iterations), self.seed,
            self.use_cpu, self.cg_tolerance,
        )
        model = LogisticFalkon(**config)
        model.fit(torch.from_numpy(x), torch.from_numpy(y))
        f = model.predict(torch.from_numpy(x)).numpy()

        # compute_t in FLKutils
        t = 2 * (w * np.sum(1 - np.exp(f[y == 0])) + np.sum(f[y == 1]))
        return NPLMResult(t=float(t), f_data=f[: len(data), 0], f_ref=f[len(data) :, 0],
                          info={"sigma": self.sigma, "model": model})


def width_scan(data, ref, w_ref: float, widths: Sequence[float], M: int, lam: float = 1e-7,
               iterations: int = 10_000_000, seed: int | None = None,
               use_cpu: bool = not torch.cuda.is_available()) -> np.ndarray:
    """t for each kernel width, the analogue of ``NPLM_dist`` (see module docstring)."""
    return np.array([
        Falkon(sigma=s, M=M, lam=lam, iterations=iterations, seed=seed, use_cpu=use_cpu)
        .fit(data, ref, w_ref=np.full(len(ref), w_ref)).t
        for s in widths
    ])
