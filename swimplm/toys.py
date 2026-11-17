"""Pseudo-experiments ("toys") to calibrate an NPLM test.

``run_toys`` is a method-agnostic port of ``run_realistic_toy`` from
phlab-neurips25 (``data/data_utils.py``), the routine used for the Higgs 4l
results. It takes any test with a ``fit(data, ref, w_data, w_ref)`` method.
The sampling, its random-number order, and the event weights are unchanged,
so with ``SparKer`` and the same seeds it reproduces the original toys.

What one toy does
-----------------
Inputs are a weighted "data-like" sample (signal + background, labelled) and a
weighted reference sample (MC). Before the toys:

* The data-like signal and background are unweighted by accept-reject
  against their largest weight. Their expected yields are the sums of their
  weights.
* Reference events with the signal label are removed.

Then, for each toy:

* Alternative toy: Poisson(n_sig) signal + Poisson(n_bkg) background events,
  drawn with replacement, each with weight 1.
* Null toy: Poisson(n_bkg) events drawn from the *data-like background*,
  each with weight 1. The original notes that strictly these should come from
  the reference; we keep the original behaviour.
* Reference: Poisson(n_ref) reference events with weights
  w * N_ref_total / n_ref.
* t_alt = test(alternative toy, reference) and t_null = test(null toy, reference).
"""

from __future__ import annotations

import numpy as np
import torch

from .nplm import NPLMTest


def unweight(sample: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Accept-reject unweighting (``rebalance`` in the original)."""
    keep = torch.rand_like(weights) < weights / torch.max(weights)
    return sample[keep]


def run_toys(
    test: NPLMTest,
    data, data_labels, data_weights,
    ref, ref_labels, ref_weights,
    sig_label: int,
    n_ref: int,
    n_toys: int,
    drop_last_column: bool = True,
    unweight_reference: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``n_toys`` alternative and null toys and return (t_alt, t_null).

    Args:
        test: Any NPLM test (SparKer, Falkon, Swim).
        data, data_labels, data_weights: Data-like sample with process labels
            and event weights.
        ref, ref_labels, ref_weights: Reference sample, same format.
        sig_label: Label of the signal process.
        n_ref: Expected size of the reference sample in each toy.
        n_toys: Number of toys.
        drop_last_column: Remove the last feature (the invariant mass in the
            Higgs inputs) before fitting; ``splitmass=True`` in the original.
        unweight_reference: Unweight the reference once before the toys, so
            that every reference event has the same weight. Needed by Falkon,
            which accepts only a single reference weight. Off by default (the
            original behaviour).
        verbose: Print t for each toy.

    Randomness comes from the global numpy and torch generators, as in the
    original. Seed both for reproducibility.
    """
    data, data_labels, data_weights = map(torch.as_tensor, (data, data_labels, data_weights))
    ref, ref_labels, ref_weights = map(torch.as_tensor, (ref, ref_labels, ref_weights))

    # Reference: drop signal-labelled events.
    is_bkg_ref = ref_labels != sig_label
    refs, rweights = ref[is_bkg_ref], ref_weights[is_bkg_ref]

    # Data-like sample: split into signal and background and unweight.
    is_sig = data_labels == sig_label
    sigs, sweights = data[is_sig], data_weights[is_sig]
    bkgs, bweights = data[~is_sig], data_weights[~is_sig]
    n_sig, n_bkg = float(torch.sum(sweights)), float(torch.sum(bweights))
    if verbose:
        print(f"expected signal: {n_sig:.1f}, expected background: {n_bkg:.1f}")
    sigs = unweight(sigs, sweights)
    bkgs = unweight(bkgs, bweights)

    if unweight_reference:
        total = torch.sum(rweights)
        refs = unweight(refs, rweights)
        rweights = torch.full((len(refs),), float(total) / len(refs), dtype=torch.float64)

    ref_weight_scale = len(refs) / n_ref

    # Poisson-fluctuated sizes of all toys (drawn up front, as in the original).
    n_sigs = np.random.poisson(lam=n_sig, size=n_toys)
    n_bkgs = np.random.poisson(lam=n_bkg, size=n_toys)
    n_refs = np.random.poisson(lam=n_ref, size=n_toys)
    n_nulls = np.random.poisson(lam=n_bkg, size=n_toys)

    def prepare(x):
        x = x.detach()
        return x[:, :-1] if drop_last_column else x

    t_alt, t_null = [], []
    for i in range(n_toys):
        sig_idx = np.random.choice(len(sigs), size=n_sigs[i], replace=True)
        bkg_idx = np.random.choice(len(bkgs), size=n_bkgs[i], replace=True)
        ref_idx = np.random.choice(len(refs), size=n_refs[i], replace=True)
        null_idx = np.random.choice(len(bkgs), size=n_nulls[i], replace=True)

        alt_toy = torch.cat((sigs[sig_idx], bkgs[bkg_idx]))
        null_toy = bkgs[null_idx]
        ref_toy = refs[ref_idx]
        ref_toy_weights = rweights[ref_idx] * ref_weight_scale

        t_alt.append(test.fit(prepare(alt_toy), prepare(ref_toy),
                              torch.ones(len(alt_toy)), ref_toy_weights.detach()).t)
        t_null.append(test.fit(prepare(null_toy), prepare(ref_toy),
                               torch.ones(len(null_toy)), ref_toy_weights.detach()).t)
        if verbose:
            print(f"toy {i}: t_alt = {t_alt[-1]:.3f}, t_null = {t_null[-1]:.3f}")

    return np.array(t_alt), np.array(t_null)
