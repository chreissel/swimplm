"""Core NPLM quantities shared by all test methods.

NPLM ("New Physics Learning Machine") compares a *data* sample D with a
*reference* sample R that represents the background-only hypothesis. A model
f(x) for the log density ratio log[n(x|data) / n(x|ref)] is fitted by
minimising the extended-likelihood loss

    L[f] = sum_{x in R} w_x (exp(f(x)) - 1)  -  sum_{x in D} w_x f(x),

and the test statistic is t = -2 * min_f L[f]. Large t means the data are
unlikely under the reference.

Every method in ``swimplm.methods`` (SparKer, Falkon, SWIM) fits f in its own
way but returns the same ``NPLMResult``.

The significance helpers at the bottom are taken unchanged from
phlab-neurips25 (``data/data_utils.py``: ``zscore``, ``zemp``) and from
``experiments/higgs/Higgs_final_pvalues.ipynb`` (empirical p-value).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from scipy.stats import chi2, norm


@dataclass
class NPLMResult:
    """Outcome of one NPLM fit.

    Attributes:
        t: Test statistic t = -2 * L[f_hat].
        f_data: Fitted log-ratio f evaluated on the data events (if available).
        f_ref: Fitted log-ratio f evaluated on the reference events (if available).
        info: Method-specific diagnostics (iterations, sampled pairs, ...).
    """

    t: float
    f_data: np.ndarray | None = None
    f_ref: np.ndarray | None = None
    info: dict = field(default_factory=dict)


class NPLMTest(Protocol):
    """Interface implemented by every test method."""

    def fit(self, data, ref, w_data=None, w_ref=None) -> NPLMResult:
        """Fit f on (data, ref) and return the test statistic."""
        ...


# ---------------------------------------------------------------------------
# Loss and test statistic
# ---------------------------------------------------------------------------


def nplm_loss(f_data, f_ref, w_data, w_ref) -> float:
    """Extended-likelihood NPLM loss L[f] (numpy version)."""
    return float(np.sum(w_ref * (np.exp(f_ref) - 1.0)) - np.sum(w_data * f_data))


def nplm_loss_torch(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """NPLM loss in the (target, pred) layout used by SparKer.

    Identical to ``NPLMLoss`` in phlab-neurips25 ``models/losses.py``.

    Args:
        target: [N, 2] tensor; column 0 is the label (1 = data, 0 = reference),
            column 1 the per-event weight.
        pred: [N, 1] tensor with f(x).
    """
    f = pred[:, 0]
    y = target[:, 0]
    w = target[:, 1]
    return torch.sum((1 - y) * w * (torch.exp(f) - 1) - y * w * f)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def to_numpy(x) -> np.ndarray:
    """Convert a torch tensor or array-like to a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def default_weights(w, n: int) -> np.ndarray:
    """Return ``w`` as a float64 numpy array, or ones if ``w`` is None."""
    if w is None:
        return np.ones(n)
    return to_numpy(w).astype(np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# Significance from toy distributions
# ---------------------------------------------------------------------------


def z_asymptotic(t_alt, t_null) -> float:
    """Median significance assuming t_null ~ chi2(dof = median(t_null)).

    Same as ``zscore`` in phlab-neurips25 ``data/data_utils.py``.
    """
    t_alt, t_null = np.asarray(t_alt), np.asarray(t_null)
    dof = np.median(t_null)
    return float(-norm.ppf(chi2.sf(np.median(t_alt), dof)))


def z_empirical(t_alt, t_null) -> float:
    """Empirical significance of mean(t_alt) w.r.t. the null toys.

    Same as ``zemp`` in phlab-neurips25 ``data/data_utils.py``. If no null
    toy exceeds mean(t_alt), the p-value is set to 1 / len(t_alt), as in
    the original.
    """
    t_alt, t_null = np.asarray(t_alt), np.asarray(t_null)
    p = np.sum(t_null > np.mean(t_alt)) / t_null.shape[0]
    if p == 0:
        p = 1.0 / t_alt.shape[0]
    return float(norm.ppf(1 - p))


def p_value_empirical(t_obs: float, t_null) -> float:
    """Fraction of null toys with t > t_obs (Higgs_final_pvalues.ipynb)."""
    t_null = np.asarray(t_null)
    return float(np.sum(t_null > t_obs) / len(t_null))
