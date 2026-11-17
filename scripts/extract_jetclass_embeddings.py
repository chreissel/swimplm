"""Embed JetClass jets with a trained contrastive model and write an embedding file.

The model and the dataset are set up from the same training config used with
``cli.py`` (e.g. ``configs/jetclass/jetClass_parT_simCLR_T0.5.yaml``), i.e. the
phlab-neurips25 JetClass setup:

* model:   the ``model`` section (JetClassSimCLRModel with a ParT encoder);
           the weights are loaded from the Lightning checkpoint.
* data:    the ``data`` section (JetClassDataset: classes, input_config,
           batch_size, num_workers). The test split is used, because only
           there does the loader return observers such as the jet mass.

Output (format: ``swimplm/io.py``): the jets are split at random into a
data-like sample and a reference sample (``--data-fraction``), each with
weight 1. The embedding is followed by the jet soft-drop mass as the last
column, so the default ``drop_last_column=True`` in the toys removes it again.
Labels are the JetClass label indices:

    0 = QCD, 1 = Tbqq (top), 2 = Wqq, 3 = Zqq, 4 = Hbb

Because the samples are unweighted, pass the expected yields to the toys,
e.g. with top as signal:

    python scripts/run_toys.py jetclass_embedding.npz --method swim \
        --sig-label 1 --n-bkg 20000 --n-sig 200 --n-ref 40000

Example:
    python scripts/extract_jetclass_embeddings.py \
        --config configs/jetclass/jetClass_parT_simCLR_T0.5.yaml \
        --ckpt work/train \
        --data-dir /path/to/JetClass --max-events 500000 \
        --output jetclass_embedding.npz
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import sys

import numpy as np
import torch
import yaml

# The training code (models/, data/) lives at the repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from swimplm.io import EmbeddingSample, save_embedding_npz  # noqa: E402

JETCLASS_LABELS = ["QCD", "Tbqq", "Wqq", "Zqq", "Hbb"]


def instantiate(spec):
    """Build an object from a Lightning-CLI style {class_path, init_args} dict (recursively)."""
    if isinstance(spec, dict) and "class_path" in spec:
        module, name = spec["class_path"].rsplit(".", 1)
        kwargs = {k: instantiate(v) for k, v in spec.get("init_args", {}).items()}
        return getattr(importlib.import_module(module), name)(**kwargs)
    return spec


def find_checkpoint(path: str) -> str:
    """Return ``path`` if it is a file, else the newest checkpoint below it.

    With the W&B logger, Lightning stores checkpoints under
    ``<default_root_dir>/<project>/<run id>/checkpoints/``. A ``last.ckpt``
    is preferred over other checkpoints.
    """
    if os.path.isfile(path):
        return path
    candidates = glob.glob(os.path.join(path, "**", "last.ckpt"), recursive=True)
    candidates = candidates or glob.glob(os.path.join(path, "**", "*.ckpt"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"no .ckpt file found below {path}")
    return max(candidates, key=os.path.getmtime)


def load_model(config: dict, ckpt_path: str, device: str):
    """Build the model from the config and load the checkpoint weights."""
    model = instantiate(config["model"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("state_dict", state))
    return model.to(device).eval()


def make_test_loader(config: dict, data_dir: str | None, batch_size: int | None, num_workers: int | None):
    """JetClassDataset test loader from the config's data section."""
    from data.datasets import JetClassDataset

    init_args = dict(config["data"]["init_args"])
    if data_dir is not None:
        init_args["data_dir"] = data_dir
    if batch_size is not None:
        init_args["batch_size"] = batch_size
    if num_workers is not None:
        init_args["num_workers"] = num_workers
    dataset = JetClassDataset(**init_args)
    missing = [c for c, files in dataset.test_file_dict.items() if not files]
    if missing:
        raise FileNotFoundError(
            f"no ROOT files for classes {missing} in {dataset.test_dir}; set --data-dir to the "
            "JetClass root (the directory containing test_20M/)")
    return dataset.test_dataloader()


@torch.no_grad()
def embed(model, loader, device: str, max_events: int | None, representation: str, mass_observer: str | None):
    """Run the encoder over the loader; return embeddings (+ mass column) and labels."""
    features, labels = [], []
    n = 0
    for x, y, observers in loader:
        x = {k: torch.as_tensor(v).to(device) for k, v in x.items()}
        h = model.encoder(x)
        if representation == "projection":
            h = model.projector(h)
        h = h.double().cpu()
        if mass_observer is not None:
            h = torch.cat((h, torch.as_tensor(observers[mass_observer]).double().reshape(-1, 1)), dim=1)
        features.append(h)
        labels.append(torch.as_tensor(y["_label_"]).reshape(-1))
        n += len(h)
        if max_events is not None and n >= max_events:
            break
    features, labels = torch.cat(features), torch.cat(labels)
    if max_events is not None:
        features, labels = features[:max_events], labels[:max_events]
    return features, labels


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="training config (cli.py format)")
    p.add_argument("--ckpt", required=True,
                   help="Lightning checkpoint, or a directory (e.g. the training root) to search for the newest one")
    p.add_argument("--output", required=True, help="output .npz")
    p.add_argument("--data-dir", default=None,
                   help="JetClass root containing test_20M/ (default: the path in data/datasets.py)")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--data-fraction", type=float, default=0.5,
                   help="fraction of jets in the data-like sample; the rest is the reference")
    p.add_argument("--representation", choices=["embed", "projection"], default="embed",
                   help="encoder output (default, as in the Higgs study) or projector output")
    p.add_argument("--mass-observer", default="jet_sdmass", help="observer appended as last column ('none' to skip)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    ckpt = find_checkpoint(args.ckpt)
    print(f"checkpoint: {ckpt}")
    model = load_model(config, ckpt, args.device)
    loader = make_test_loader(config, args.data_dir, args.batch_size, args.num_workers)
    mass = None if args.mass_observer.lower() == "none" else args.mass_observer
    features, labels = embed(model, loader, args.device, args.max_events, args.representation, mass)

    # Random split into data-like and reference samples.
    perm = torch.randperm(len(features), generator=torch.Generator().manual_seed(args.seed))
    n_data = int(args.data_fraction * len(features))
    d, r = perm[:n_data], perm[n_data:]
    sample = EmbeddingSample(features[d], labels[d], torch.ones(len(d)),
                             features[r], labels[r], torch.ones(len(r)))
    save_embedding_npz(args.output, sample, class_names=np.array(JETCLASS_LABELS))

    counts = {JETCLASS_LABELS[int(k)]: int(v) for k, v in zip(*torch.unique(labels, return_counts=True))}
    print(f"embedded {len(features)} jets ({counts}); embedding dim {features.shape[1] - (mass is not None)}")
    print(f"wrote {args.output}: {len(d)} data-like jets, {len(r)} reference jets")


if __name__ == "__main__":
    main()
