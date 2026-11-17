"""Calibrate an NPLM test with toys on an embedding file and report its significance.

The input is an .npz file in the format of the phlab-neurips25 Higgs
experiments (e.g. ``experiments/higgs/higgs_cntr_space.npz``):

    toy_data_out, toy_data_label, toy_data_weights   # data-like sample
    toy_mc_out,   toy_mc_label,   toy_mc_weights     # reference (MC) sample

The last column of ``*_out`` is the invariant mass. It is dropped before
fitting, as in the original.

Examples:
    python scripts/run_toys.py higgs_cntr_space.npz --method sparker --sparker-width 2.0
    python scripts/run_toys.py higgs_cntr_space.npz --method falkon --falkon-M 1000
    python scripts/run_toys.py higgs_cntr_space.npz --method swim --swim-widths 512
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from swimplm import Falkon, SparKer, Swim, run_toys, z_asymptotic, z_empirical
from swimplm.methods.falkon import candidate_widths


def build_test(args, ref_features):
    if args.method == "sparker":
        return SparKer(widths=(args.sparker_width,), n_centroids=(args.sparker_centroids,))
    if args.method == "falkon":
        sigma = args.falkon_sigma
        if sigma is None:  # median pairwise distance of the reference
            sigma = float(candidate_widths(ref_features, percentiles=[50], add_broad=False)[0])
            print(f"Falkon kernel width (median pairwise distance): {sigma}")
        return Falkon(sigma=sigma, M=args.falkon_M, lam=args.falkon_lambda)
    if args.method == "swim":
        return Swim(layer_widths=tuple(args.swim_widths), activation=args.swim_activation,
                    sample_uniformly=args.swim_uniform, l2=args.swim_l2,
                    random_seed=args.seed)
    raise ValueError(args.method)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="embedding .npz file")
    p.add_argument("--method", choices=["sparker", "falkon", "swim"], required=True)
    p.add_argument("--n-toys", type=int, default=10)
    p.add_argument("--n-ref", type=int, default=40000, help="expected reference size per toy")
    p.add_argument("--sig-label", type=int, default=1, help="label of the signal process")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", default=None, help="output .npz (default: <method>_toys_seed<seed>.npz)")
    # SparKer
    p.add_argument("--sparker-width", type=float, default=2.0)
    p.add_argument("--sparker-centroids", type=int, default=30)
    # Falkon
    p.add_argument("--falkon-sigma", type=float, default=None)
    p.add_argument("--falkon-M", type=int, default=1000)
    p.add_argument("--falkon-lambda", type=float, default=1e-7)
    # SWIM
    p.add_argument("--swim-widths", type=int, nargs="+", default=[512])
    p.add_argument("--swim-activation", choices=["tanh", "relu"], default="tanh")
    p.add_argument("--swim-uniform", action="store_true", help="ignore labels when sampling pairs")
    p.add_argument("--swim-l2", type=float, default=1e-6)
    args = p.parse_args()

    # Same seeding convention as the original run scripts.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed + 1000)

    f = np.load(args.input)
    data, data_labels, data_weights = (torch.tensor(f[k]) for k in
                                       ("toy_data_out", "toy_data_label", "toy_data_weights"))
    ref, ref_labels, ref_weights = (torch.tensor(f[k]) for k in
                                    ("toy_mc_out", "toy_mc_label", "toy_mc_weights"))

    test = build_test(args, ref[:, :-1].numpy())
    start = time.time()
    t_alt, t_null = run_toys(
        test, data, data_labels, data_weights.double(), ref, ref_labels, ref_weights.double(),
        sig_label=args.sig_label, n_ref=args.n_ref, n_toys=args.n_toys,
        unweight_reference=(args.method == "falkon"),
    )
    elapsed = time.time() - start

    print(f"\n{args.method}: {args.n_toys} toys in {elapsed:.1f} s "
          f"({elapsed / (2 * args.n_toys):.2f} s per fit)")
    print(f"median t_alt = {np.median(t_alt):.2f}, median t_null = {np.median(t_null):.2f}")
    print(f"Z (asymptotic) = {z_asymptotic(t_alt, t_null):.2f}, "
          f"Z (empirical) = {z_empirical(t_alt, t_null):.2f}")

    output = args.output or f"{args.method}_toys_seed{args.seed}.npz"
    np.savez(output, t_alt=t_alt, t_null=t_null, seconds=elapsed, args=str(vars(args)))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
