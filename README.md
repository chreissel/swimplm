# swimplm

NPLM anomaly tests in learned (contrastive) embeddings, comparing three ways to fit the
log density ratio between data and reference:

| Method | Model | Fit |
|---|---|---|
| **SparKer** | ~30 Gaussian kernels with trainable centroids | non-convex, full-batch Adam |
| **Falkon** | Nyström Gaussian kernels (uniform centres) | convex, LogisticFalkon |
| **SWIM** | hidden layers sampled from data pairs ([swimnetworks](https://gitlab.com/fd-research/swimnetworks)) | convex, Newton on the NPLM loss |

SparKer and Falkon are ports of the code in
[sambt/phlab-neurips25](https://github.com/sambt/phlab-neurips25) @ `f5956e5`, and give
**identical** results to it (checked in `tests/`).

## Layout

```
swimplm/                    NPLM package
  nplm.py                   NPLM loss, NPLMResult, significance (Z, p-values)
  toys.py                   toy calibration (port of run_realistic_toy), any method;
                            yield/signal-strength options, IsolatedRandomness
  io.py                     embedding .npz format (read/write)
  factory.py                method names -> configured tests (used by the scripts)
  methods/sparker.py        SparKer   (port of SPARKutils + data_utils.sparkKer)
  methods/falkon.py         Falkon    (port of FLKutils / NPLMutils)
  methods/swim.py           SWIM features + convex Newton readout
scripts/
  extract_jetclass_embeddings.py   trained checkpoint + JetClass -> embedding .npz
  run_toys.py                      one method: toys on an embedding .npz, print Z
  compare_methods.py               all methods on identical toys vs signal strength
tests/                      SparKer/Falkon vs original code, toy options, SWIM checks

cli.py, submit.sh           embedding training (Lightning CLI), copied unchanged
configs/                    JetClass ParT SimCLR configs
models/ data/ utils/ opt/   phlab-neurips25 code, copied unchanged (training + reference
                            implementation used by the tests)
```

All methods share one interface:

```python
from swimplm import SparKer, Falkon, Swim, run_toys, z_empirical

result = Swim(layer_widths=(512,)).fit(data, ref, w_data, w_ref)   # result.t
t_alt, t_null = run_toys(SparKer(widths=(2.0,)), data, data_labels, data_weights,
                         ref, ref_labels, ref_weights, sig_label=1, n_ref=40000, n_toys=100)
print(z_empirical(t_alt, t_null))
```

## Installation

```bash
pip install -e .
```

Falkon has no PyPI wheel; build it from source (CPU build shown):

```bash
pip install --no-build-isolation git+https://github.com/FalkonML/falkon.git
```

Run the tests from the repository root (the comparison tests also need the dependencies
of the original code: `lightning`, `torchvision`, `awkward`, `uproot`, `h5py`, `jax`,
`pylorentz`, `lmfit`, `corner`, `torchmetrics`):

```bash
python -m pytest tests
```

## Running toys

On an embedding file in the phlab-neurips25 Higgs format (`toy_data_out`, `toy_data_label`,
`toy_data_weights`, `toy_mc_out`, `toy_mc_label`, `toy_mc_weights`; last column = mass):

```bash
python scripts/run_toys.py higgs_cntr_space.npz --method sparker --sparker-width 2.0
```

```bash
python scripts/run_toys.py higgs_cntr_space.npz --method falkon --falkon-M 1000
```

```bash
python scripts/run_toys.py higgs_cntr_space.npz --method swim --swim-widths 512
```

## JetClass workflow

1. Train the embedding (see *Embedding training* below).
2. Embed the JetClass test set. The model and data loader are set up from the same
   training config. Jets are split into a data-like and a reference half, and the
   soft-drop mass is appended as the last column.

   ```bash
   python scripts/extract_jetclass_embeddings.py --config configs/jetclass/jetClass_parT_simCLR_T0.5.yaml --ckpt path/to/last.ckpt --data-dir /path/to/JetClass/test_20M --max-events 500000 --output jetclass_embedding.npz
   ```

3. Compare the methods on identical toys. Labels: 0 QCD, 1 top, 2 W, 3 Z, 4 H. The samples
   are unweighted, so set the expected yields:

   ```bash
   python scripts/compare_methods.py jetclass_embedding.npz --sig-label 1 --n-bkg 20000 --n-sig 200 --n-ref 40000 --methods sparker falkon swim swim-uniform --signal-scales 0 0.5 1 2 --n-toys 50 --output-dir comparison
   ```

   This writes `comparison/summary.csv` (Z and time per fit for each method and signal
   scale) plus the t distributions. Method names: `sparker`, `falkon`, `falkon@<percentile>`
   (Falkon kernel width), `swim`, `swim-uniform` (ablation: label-free pair sampling).

**Regularisation matters.** Without signal, a weakly regularised fit can still give a
very large t. For SWIM with 128 neurons on background-only toys, t drops from about
6·10⁴ (`--swim-l2 1e-6`) to about 50 (`--swim-l2 10`). Tune `--swim-l2`,
`--falkon-lambda`/`--falkon-M` and the SparKer width on null toys before comparing power.

## Notes on faithfulness

- **SparKer**: initialisation, random-number order, dtypes, Adam settings, coefficient
  clipping and the "minimum loss" convention are unchanged. Debug plots were removed.
- **Falkon**: configuration and `t` are unchanged. Three issues in the original
  `NPLM_dist` were *not* copied:
  - it passed the percentiles (1, 25, ...) as kernel widths, instead of the widths
    computed from them;
  - it set `use_cpu=True` exactly when CUDA was available;
  - it seeded Falkon from the wall clock.

  Falkon accepts only one reference weight, so for weighted MC use
  `run_toys(..., unweight_reference=True)`. The script does this automatically.
- **Toys**: the null toys are drawn from the data-like background, as in the original
  (`run_realistic_toy`).

## Embedding training

```bash
python cli.py fit --config configs/jetclass/jetClass_parT_simCLR_T0.5.yaml
```

The JetClass data paths are hard-coded for the Cannon cluster in `data/datasets.py`
(`JetClassDataset`), and logging uses `WandbLogger`.
