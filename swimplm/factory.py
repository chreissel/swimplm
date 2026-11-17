"""Build NPLM tests from short method names (used by the scripts).

Method names:
    sparker        SparKer (phlab-neurips25 settings)
    falkon         Falkon, kernel width = median pairwise reference distance
    falkon@<p>     Falkon, kernel width = <p>-th percentile of pairwise distances
    swim           SWIM, pairs sampled using the data/reference labels
    swim-uniform   SWIM, pairs sampled uniformly (ablation: plain random features)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .methods import Falkon, SparKer, Swim
from .methods.falkon import candidate_widths


@dataclass
class MethodSettings:
    """Hyperparameters of all methods (defaults as in the README)."""

    sparker_width: float = 2.0
    sparker_centroids: int = 30
    falkon_M: int = 1000
    falkon_lambda: float = 1e-7
    swim_widths: tuple[int, ...] = (512,)
    swim_activation: str = "tanh"
    swim_l2: float = 1.0
    seed: int = 1


def build_test(name: str, settings: MethodSettings, ref_features: np.ndarray):
    """Return the NPLM test called ``name``; ``ref_features`` sets Falkon's width."""
    if name == "sparker":
        return SparKer(widths=(settings.sparker_width,), n_centroids=(settings.sparker_centroids,))
    if name == "falkon" or name.startswith("falkon@"):
        percentile = float(name.split("@")[1]) if "@" in name else 50.0
        sigma = float(candidate_widths(ref_features, [percentile], add_broad=False)[0])
        # Always Falkon's CPU backend: toy jobs run on CPU nodes (a CPU-only Falkon build suffices).
        return Falkon(sigma=sigma, M=settings.falkon_M, lam=settings.falkon_lambda, use_cpu=True)
    if name in ("swim", "swim-uniform"):
        return Swim(layer_widths=tuple(settings.swim_widths), activation=settings.swim_activation,
                    sample_uniformly=(name == "swim-uniform"), l2=settings.swim_l2,
                    random_seed=settings.seed)
    raise ValueError(f"unknown method {name!r}; see swimplm.factory for the options")
