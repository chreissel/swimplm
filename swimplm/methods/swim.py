"""SWIM: NPLM with sampled hidden layers and a convex readout.

The hidden layers are sampled with the SWIM algorithm from the
``swimnetworks`` package (https://gitlab.com/fd-research/swimnetworks,
Bolager et al., arXiv:2306.16830). Each neuron is built from a pair of
events (x1, x2):

    w = s * (x2 - x1) / |x2 - x1|^2,     b = -<w, x1> - s',

so its activation changes between x1 and x2. In supervised mode, pairs are
drawn with probability proportional to |y2 - y1| / |x2 - x1|. With the NPLM
labels (data = 1, reference = 0), this favours nearby data/reference pairs,
i.e. the regions where the two samples differ. With
``sample_uniformly=True`` the labels are not used at all.

Once the hidden layers are fixed, f(x) = Phi(x) . beta + b0 is linear in
(beta, b0). The NPLM loss

    L(beta) = sum_R w (exp(f) - 1) - sum_D w f + (l2 / 2) |beta|^2

is then convex and is minimised with a damped Newton method. The penalty
does not apply to the bias b0. The test statistic is t = -2 * L without the
penalty term, the same convention as SparKer and Falkon.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from swimnetworks import Dense

from ..nplm import NPLMResult, default_weights, nplm_loss, to_numpy


@dataclass
class Swim:
    """NPLM test with SWIM-sampled features.

    Attributes:
        layer_widths: Number of neurons in each sampled hidden layer.
        activation: "tanh" or "relu"; the matching SWIM sampler is used.
        sample_uniformly: If True, pairs are drawn uniformly, ignoring the
            data/reference labels (plain random features from data pairs).
        l2: L2 penalty on the readout weights. This controls overfitting and
            must be tuned: with weak penalties (e.g. 1e-6) t is huge even
            without any signal.
        random_seed: Seed of the SWIM sampler (layer i uses random_seed + i).
        max_iter: Maximum number of Newton iterations.
        tol: Stop when the Newton decrement^2 / 2 drops below tol.
    """

    layer_widths: Sequence[int] = (512,)
    activation: str = "tanh"
    sample_uniformly: bool = False
    l2: float = 1.0
    random_seed: int = 1
    max_iter: int = 100
    tol: float = 1e-10

    def fit(self, data, ref, w_data=None, w_ref=None) -> NPLMResult:
        data, ref = to_numpy(data).astype(np.float64), to_numpy(ref).astype(np.float64)
        w_data = default_weights(w_data, len(data))
        w_ref = default_weights(w_ref, len(ref))

        x = np.concatenate((data, ref), axis=0)
        labels = np.concatenate((np.ones(len(data)), np.zeros(len(ref))))

        layers = self.sample_hidden_layers(x, labels)
        phi = self.features(layers, x)
        beta, n_iter = self.fit_readout(phi[: len(data)], phi[len(data) :], w_data, w_ref)

        f = phi @ beta
        f_data, f_ref = f[: len(data)], f[len(data) :]
        t = -2.0 * nplm_loss(f_data, f_ref, w_data, w_ref)
        return NPLMResult(
            t=t, f_data=f_data, f_ref=f_ref,
            info={"layers": layers, "beta": beta, "newton_iterations": n_iter},
        )

    # -- hidden layers -------------------------------------------------------

    def sample_hidden_layers(self, x: np.ndarray, labels: np.ndarray) -> list[Dense]:
        """Sample the hidden layers one after the other with swimnetworks.Dense."""
        y = None if self.sample_uniformly else labels.reshape(-1, 1)
        layers, h = [], x
        for i, width in enumerate(self.layer_widths):
            layer = Dense(
                layer_width=width,
                activation=self.activation,
                parameter_sampler=self.activation,
                sample_uniformly=self.sample_uniformly,
                random_seed=self.random_seed + i,
            )
            h = layer.fit_transform(h, y)
            layers.append(layer)
        return layers

    @staticmethod
    def features(layers: Sequence[Dense], x: np.ndarray) -> np.ndarray:
        """Hidden-layer output with a constant column appended for the bias."""
        h = x
        for layer in layers:
            h = layer.transform(h)
        return np.column_stack([h, np.ones(len(h))])

    # -- convex readout ------------------------------------------------------

    def fit_readout(self, phi_data, phi_ref, w_data, w_ref) -> tuple[np.ndarray, int]:
        """Minimise the penalised NPLM loss over beta with damped Newton steps."""
        n_params = phi_data.shape[1]
        penalty = np.full(n_params, self.l2)
        penalty[-1] = 0.0  # do not penalise the bias

        data_term = phi_data.T @ w_data  # constant part of the gradient

        def objective(beta):
            # Trial steps of the line search may overflow; they are rejected.
            with np.errstate(over="ignore"):
                f_ref = phi_ref @ beta
                return (np.sum(w_ref * (np.exp(f_ref) - 1)) - w_data @ (phi_data @ beta)
                        + 0.5 * np.sum(penalty * beta**2))

        beta = np.zeros(n_params)
        loss = objective(beta)
        for it in range(1, self.max_iter + 1):
            r = w_ref * np.exp(phi_ref @ beta)  # w * exp(f) on reference events
            grad = phi_ref.T @ r - data_term + penalty * beta
            hess = (phi_ref * r[:, None]).T @ phi_ref + np.diag(penalty)
            hess[np.diag_indices_from(hess)] += 1e-12  # numerical safety
            step = np.linalg.solve(hess, -grad)

            decrement = -grad @ step  # Newton decrement squared
            if decrement / 2 < self.tol:
                return beta, it - 1

            # Backtracking line search (Armijo condition).
            step_size = 1.0
            while True:
                candidate = beta + step_size * step
                new_loss = objective(candidate)
                if np.isfinite(new_loss) and new_loss <= loss - 0.25 * step_size * decrement:
                    break
                step_size *= 0.5
                if step_size < 1e-12:
                    return beta, it
            beta, loss = candidate, new_loss
        return beta, self.max_iter
