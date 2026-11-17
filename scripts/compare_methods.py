"""Compare NPLM methods on identical toys, as a function of the injected signal strength.

For every signal scale, all methods see exactly the same toys: the toy seed is
the same, and each fit runs with its own random state (``IsolatedRandomness``),
so a method's internal randomness cannot shift the toys.

The null toys follow the same distribution at every signal scale, so for each
method the null toys of all scales are pooled to compute Z.

Outputs (in --output-dir):
    toys_<method>_scale<s>.npz   t_alt, t_null and timing of each run
    summary.csv                  one row per (method, signal scale)

Example (Higgs file from phlab-neurips25):
    python scripts/compare_methods.py higgs_cntr_space.npz \
        --methods sparker falkon swim swim-uniform --signal-scales 0 0.5 1 2 --n-toys 50
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch
from run_toys import add_toy_arguments, method_settings

from swimplm import run_toys, z_asymptotic, z_empirical
from swimplm.factory import build_test
from swimplm.io import load_embedding_npz
from swimplm.toys import IsolatedRandomness


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_toy_arguments(p)
    p.add_argument("--methods", nargs="+", default=["sparker", "falkon", "swim", "swim-uniform"])
    p.add_argument("--signal-scales", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument("--output-dir", default="comparison")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sample = load_embedding_npz(args.input)
    ref_features = (sample.ref if args.keep_last_column else sample.ref[:, :-1]).numpy()
    settings = method_settings(args)

    rows = []
    for method in args.methods:
        base_test = build_test(method, settings, ref_features)
        runs = {}  # signal scale -> (t_alt, t_null, seconds)
        for scale in args.signal_scales:
            # Same toy seed (and same fit seeds) for every method at a given scale.
            np.random.seed(args.seed)
            torch.manual_seed(args.seed + 1000)
            test = IsolatedRandomness(base_test, seed=args.seed)
            start = time.time()
            t_alt, t_null = run_toys(
                test, **sample.as_toy_inputs(), sig_label=args.sig_label, n_ref=args.n_ref,
                n_toys=args.n_toys, n_sig=args.n_sig, n_bkg=args.n_bkg, signal_scale=scale,
                drop_last_column=not args.keep_last_column,
                unweight_reference=method.startswith("falkon"), verbose=False,
            )
            runs[scale] = (t_alt, t_null, time.time() - start)
            np.savez(os.path.join(args.output_dir, f"toys_{method}_scale{scale:g}.npz"),
                     t_alt=t_alt, t_null=t_null, seconds=runs[scale][2])
            print(f"{method}, signal scale {scale:g}: done in {runs[scale][2]:.1f} s")

        # Null toys follow the same distribution at every scale: pool them.
        t_null_pooled = np.concatenate([r[1] for r in runs.values()])
        for scale, (t_alt, _, seconds) in runs.items():
            row = {
                "method": method,
                "signal_scale": scale,
                "median_t_alt": float(np.median(t_alt)),
                "median_t_null": float(np.median(t_null_pooled)),
                "Z_asymptotic": z_asymptotic(t_alt, t_null_pooled),
                "Z_empirical": z_empirical(t_alt, t_null_pooled),
                "seconds_per_fit": seconds / (2 * args.n_toys),
            }
            rows.append(row)
            print("  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {os.path.join(args.output_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
