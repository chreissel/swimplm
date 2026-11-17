"""SparKer: NPLM with a sparse set of trainable Gaussian kernels.

This is a cleaned-up port of the SparKer implementation in phlab-neurips25:

* model:    ``utils/SPARKutils.py`` (``Hierarchical`` with
            ``model='Soft-SparKer2'``, ``KernelMethod_SoftMax_2``, ``KernelLayer``)
* fitting:  ``data/data_utils.py`` (``initSparkKer``, ``train_sparkKer``,
            ``sparkKer``)

The numerics (initialisation and its random-number order, dtypes, optimiser,
coefficient clipping, number of epochs, use of the minimum loss) are kept
identical, so that for the same numpy/torch seeds this class returns the same
t as the original ``sparkKer``. ``tests/test_sparker_matches_legacy.py``
checks this. Dropped from the original: debugging plots and the unused
'SparKer' / 'Soft-SparKer' model variants. The kernel width is fixed during
training, because the annealing in the original is commented out.

Model
-----
One "layer" holds M Gaussian kernels K_i(x) = exp(-|x - mu_i|^2 / (2 sigma^2))
with trainable centroids mu_i and coefficients a_i, combined with a soft-max
weighting:

    f(x) = sum_i a_i K_i(x)^2 / (sum_j K_j(x) + eps).

Several layers with different widths can be stacked; f is then their sum and
the layers are trained one after the other (layer n trains the parameters of
layers 0..n).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..nplm import NPLMResult, nplm_loss_torch

_EPS = 1e-10


class SoftSparKerLayer(nn.Module):
    """M Gaussian kernels with a soft-max readout (Soft-SparKer2)."""

    def __init__(self, centroids: torch.Tensor, width: float, coeffs_clip: float):
        super().__init__()
        n_centroids = centroids.shape[0]
        # Original: coefficients are float64 and start at zero; centroids are float32.
        self.coeffs = nn.Parameter(torch.zeros((n_centroids, 1), dtype=torch.float64))
        self.centroids = nn.Parameter(centroids.type(torch.float32))
        # Original keeps the width as a float64 tensor of shape [1].
        self.register_buffer("width", torch.tensor([width], dtype=torch.float64))
        self.coeffs_clip = coeffs_clip

    def kernels(self, x: torch.Tensor) -> torch.Tensor:
        """K_i(x) for all events and centroids, shape [N, M]."""
        dist_sq = (x[:, None, :] - self.centroids[None, :, :]) ** 2  # [N, M, d]
        return torch.exp(-0.5 * torch.sum(dist_sq / self.width**2, dim=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kernels(x)  # [N, M]
        z = torch.sum(k, dim=1, keepdim=True) + _EPS  # [N, 1]
        return torch.tensordot(k * k, self.coeffs, dims=([1], [0])) / z  # [N, 1]

    def clip_coeffs(self) -> None:
        self.coeffs.data = self.coeffs.data.clamp(-self.coeffs_clip, self.coeffs_clip)


class SparKerModel(nn.Module):
    """Sum of SoftSparKerLayers (``Hierarchical`` in the original)."""

    def __init__(self, layers: Sequence[SoftSparKerLayer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, up_to_layer: int | None = None) -> torch.Tensor:
        """f(x) summed over layers 0..up_to_layer (all layers by default)."""
        n = len(self.layers) - 1 if up_to_layer is None else up_to_layer
        return torch.stack([self.layers[i](x) for i in range(n + 1)], dim=0).sum(dim=0)

    def trainable_parameters(self, up_to_layer: int) -> list[nn.Parameter]:
        """Coefficients, then centroids, of layers 0..up_to_layer (original order)."""
        layers = self.layers[: up_to_layer + 1]
        return [layer.coeffs for layer in layers] + [layer.centroids for layer in layers]

    def clip_coeffs(self) -> None:
        for layer in self.layers:
            layer.clip_coeffs()


@dataclass
class SparKer:
    """NPLM test with a SparKer model.

    Defaults are those of ``sparkKer`` in phlab-neurips25. The Higgs runs
    used ``widths=(2.0,)`` (``run_script_ctr.py``) or ``(3.0,)``.

    Randomness comes from the global numpy and torch generators, as in the
    original. Seed them (``np.random.seed``, ``torch.manual_seed``) for
    reproducibility.

    Attributes:
        widths: Kernel width sigma of each layer.
        n_centroids: Number of kernels M in each layer.
        n_epochs: Full-batch Adam epochs per layer.
        learning_rate: Adam learning rate.
        coeffs_clip: Coefficients are clipped to [-coeffs_clip, coeffs_clip].
    """

    widths: Sequence[float] = (0.5,)
    n_centroids: Sequence[int] = (30,)
    n_epochs: Sequence[int] = (1001,)
    learning_rate: float = 0.01
    coeffs_clip: float = 100.0

    def fit(self, data, ref, w_data=None, w_ref=None) -> NPLMResult:
        data, ref = torch.as_tensor(data), torch.as_tensor(ref)
        w_data = torch.ones(len(data)) if w_data is None else torch.as_tensor(w_data)
        w_ref = torch.ones(len(ref)) if w_ref is None else torch.as_tensor(w_ref)

        features = torch.cat((data, ref), dim=0)
        labels = torch.cat((torch.ones(len(data)), torch.zeros(len(ref))))
        weights = torch.cat((w_data, w_ref))
        target = torch.stack((labels.double(), weights.double()), dim=1)  # [N, 2]

        model = self._init_model(features)
        min_loss = self._train(model, features, target)

        with torch.no_grad():
            f = model(features)[:, 0].double().numpy()
        return NPLMResult(
            t=-2.0 * min_loss,
            f_data=f[: len(data)],
            f_ref=f[len(data) :],
            info={"model": model, "min_loss": min_loss},
        )

    # -- internals ---------------------------------------------------------

    def _init_model(self, features: torch.Tensor) -> SparKerModel:
        """Random initialisation; same random-number order as ``initSparkKer``.

        The coefficients start at zero, but the original still draws
        binomial numbers for them first, so we draw them too to keep the
        random stream identical.
        """
        n_layers = len(self.widths)
        for i in range(n_layers):
            np.random.binomial(p=0.5, n=1, size=(self.n_centroids[i], 1))
        centroid_idx = [
            np.random.randint(features.shape[0], size=self.n_centroids[i])
            for i in range(n_layers)
        ]
        layers = [
            SoftSparKerLayer(features[centroid_idx[i], :], self.widths[i], self.coeffs_clip)
            for i in range(n_layers)
        ]
        return SparKerModel(layers)

    def _train(self, model: SparKerModel, features: torch.Tensor, target: torch.Tensor) -> float:
        """Train layer by layer with full-batch Adam (``train_sparkKer``).

        Returns the smallest loss seen during training. The loss is recorded
        before each optimiser step and the first one is exactly 0, because
        all coefficients start at zero, so the result is always <= 0.
        """
        min_loss = np.inf
        for n in range(len(model.layers)):
            optimizer = torch.optim.Adam(model.trainable_parameters(n), lr=self.learning_rate)
            for _ in range(int(self.n_epochs[n])):
                optimizer.zero_grad()
                loss = nplm_loss_torch(target, model(features, up_to_layer=n))
                loss.backward()
                optimizer.step()
                model.clip_coeffs()
                min_loss = min(min_loss, loss.item())
        return min_loss
