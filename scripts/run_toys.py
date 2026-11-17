"""Calibrate one NPLM test with toys on an embedding file and report its significance.

The input is an .npz embedding file (format: ``swimplm/io.py``), e.g. the
phlab-neurips25 Higgs file ``experiments/higgs/higgs_cntr_space.npz`` or the
output of ``scripts/extract_jetclass_embeddings.py``. By default the last
column (the mass) is dropped before fitting, as in the original.

Methods: sparker, falkon, falkon@<percentile>, swim, swim-uniform (see swimplm/factory.py).

Examples:
    python scripts/run_toys.py higgs_cntr_space.npz --method sparker
    python scripts/run_toys.py jetclass_embedding.npz --method swim --n-bkg 20000 --n-sig 200 --n-ref 40000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from swimplm import run_toys, z_asymptotic, z_empirical
from swimplm.factory import MethodSettings, build_test
from swimplm.io import load_embedding_npz


def add_toy_arguments(p: argparse.ArgumentParser) -> None:
    """Arguments shared with compare_methods.py."""
    p.add_argument("input", help="embedding .npz file")
    p.add_argument("--n-toys", type=int, default=10)
    p.add_argument("--n-ref", type=int, default=40000, help="expected reference size per toy")
    p.add_argument("--n-sig", type=float, default=None,
                   help="expected signal events per toy (default: sum of signal weights)")
    p.add_argument("--n-bkg", type=float, default=None,
                   help="expected background events per toy (default: sum of background weights)")
    p.add_argument("--sig-label", type=int, default=1, help="label of the signal process")
    p.add_argument("--keep-last-column", action="store_true",
                   help="do not drop the last column (the mass) before fitting")
    p.add_argument("--seed", type=int, default=1)
    # method hyperparameters
    p.add_argument("--sparker-width", type=float, default=2.0)
    p.add_argument("--sparker-centroids", type=int, default=30)
    p.add_argument("--falkon-M", type=int, default=1000)
    p.add_argument("--falkon-lambda", type=float, default=1e-7)
    p.add_argument("--swim-widths", type=int, nargs="+", default=[512])
    p.add_argument("--swim-activation", choices=["tanh", "relu"], default="tanh")
    p.add_argument("--swim-l2", type=float, default=1.0)


def method_settings(args) -> MethodSettings:
    return MethodSettings(
        sparker_width=args.sparker_width, sparker_centroids=args.sparker_centroids,
        falkon_M=args.falkon_M, falkon_lambda=args.falkon_lambda,
        swim_widths=tuple(args.swim_widths), swim_activation=args.swim_activation,
        swim_l2=args.swim_l2, seed=args.seed,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_toy_arguments(p)
    p.add_argument("--method", required=True)
    p.add_argument("--signal-scale", type=float, default=1.0, help="multiplies the expected signal yield")
    p.add_argument("--output", default=None, help="output .npz (default: <method>_toys_seed<seed>.npz)")
    args = p.parse_args()

    # Same seeding convention as the original run scripts.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed + 1000)

    sample = load_embedding_npz(args.input)
    ref_features = (sample.ref if args.keep_last_column else sample.ref[:, :-1]).numpy()
    test = build_test(args.method, method_settings(args), ref_features)

    start = time.time()
    t_alt, t_null = run_toys(
        test, **sample.as_toy_inputs(), sig_label=args.sig_label, n_ref=args.n_ref,
        n_toys=args.n_toys, n_sig=args.n_sig, n_bkg=args.n_bkg, signal_scale=args.signal_scale,
        drop_last_column=not args.keep_last_column,
        unweight_reference=args.method.startswith("falkon"),
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
