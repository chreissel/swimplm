"""Reading and writing embedding files.

The format is the one used by the phlab-neurips25 Higgs experiments:

    toy_data_out      [N_data, d]  embedding (+ optionally the mass as last column)
    toy_data_label    [N_data]     process label
    toy_data_weights  [N_data]     event weight
    toy_mc_out, toy_mc_label, toy_mc_weights   the same for the reference sample
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EmbeddingSample:
    """A data-like sample and a reference sample in embedding space."""

    data: torch.Tensor
    data_labels: torch.Tensor
    data_weights: torch.Tensor
    ref: torch.Tensor
    ref_labels: torch.Tensor
    ref_weights: torch.Tensor

    def as_toy_inputs(self) -> dict:
        """Keyword arguments for ``run_toys``."""
        return {
            "data": self.data, "data_labels": self.data_labels,
            "data_weights": self.data_weights.double(),
            "ref": self.ref, "ref_labels": self.ref_labels,
            "ref_weights": self.ref_weights.double(),
        }


def load_embedding_npz(path: str) -> EmbeddingSample:
    """Load an embedding file. Extra keys (e.g. a pickled model) are ignored."""
    f = np.load(path)
    t = {k: torch.tensor(f[k]) for k in (
        "toy_data_out", "toy_data_label", "toy_data_weights",
        "toy_mc_out", "toy_mc_label", "toy_mc_weights")}
    return EmbeddingSample(
        t["toy_data_out"], t["toy_data_label"], t["toy_data_weights"],
        t["toy_mc_out"], t["toy_mc_label"], t["toy_mc_weights"],
    )


def save_embedding_npz(path: str, sample: EmbeddingSample, **extra) -> None:
    """Write an embedding file. ``extra`` arrays are stored alongside."""
    def arr(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    np.savez(
        path,
        toy_data_out=arr(sample.data), toy_data_label=arr(sample.data_labels),
        toy_data_weights=arr(sample.data_weights),
        toy_mc_out=arr(sample.ref), toy_mc_label=arr(sample.ref_labels),
        toy_mc_weights=arr(sample.ref_weights),
        **{k: arr(v) for k, v in extra.items()},
    )
