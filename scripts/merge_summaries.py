"""Merge the summary.csv files of several comparison jobs into one table.

Each comparison job writes <run>/summary.csv. This script collects all of them
below a directory, adds a ``run`` column (the job's directory name), writes
``merged_summary.csv`` and prints the table.

Example:
    python scripts/merge_summaries.py work/comparison
"""

from __future__ import annotations

import argparse
import csv
import glob
import os


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory")
    p.add_argument("--output", default=None, help="default: <directory>/merged_summary.csv")
    args = p.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.directory, "**", "summary.csv"), recursive=True)):
        run = os.path.relpath(os.path.dirname(path), args.directory)
        with open(path) as f:
            rows += [{"run": run, **row} for row in csv.DictReader(f)]
    if not rows:
        raise SystemExit(f"no summary.csv found below {args.directory}")

    output = args.output or os.path.join(args.directory, "merged_summary.csv")
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    columns = ["run", "method", "signal_scale", "median_t_null", "median_t_alt",
               "Z_asymptotic", "Z_empirical", "seconds_per_fit"]

    def fmt(value):
        try:
            return f"{float(value):.3g}"
        except ValueError:
            return value

    table = [columns] + [[fmt(row[c]) for c in columns] for row in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(columns))]
    for r in table:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
