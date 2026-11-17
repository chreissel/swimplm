# swimplm

Sampled (SWIM) networks for fast, convex NPLM-style anomaly tests and background
estimation with uncertainties in learned contrastive embeddings.

## Embedding training (copied from phlab-neurips25)

The contrastive-embedding training infrastructure in this repo is a 1-to-1 copy of
[sambt/phlab-neurips25](https://github.com/sambt/phlab-neurips25) at commit
`f5956e51ddc48c44bb209a15c8de370d9a70fe3a`, restricted to what the JetClass example needs:

- `cli.py`, `submit.sh` – Lightning CLI entry point and SLURM submission script
- `models/`, `data/`, `utils/`, `opt/` – copied unchanged (the full packages are kept so the
  import chain of `data/datasets.py` and `models/litmodels.py` resolves as in the original)
- `configs/jetclass/`, `configs/jetclass_data_configs/` – JetClass ParT SimCLR configs

Train with:

```bash
python cli.py fit --config configs/jetclass/jetClass_parT_simCLR_T0.5.yaml
```

Notes:

- Data paths are hard-coded for the Cannon cluster in `data/datasets.py`
  (`JetClassDataset.train_dir`, `val_dir`, `test_dir`) and must be changed to point at your
  JetClass ROOT files.
- Logging uses `WandbLogger` (set in the configs).
- Python dependencies needed to import and run the JetClass pipeline: `torch`, `torchvision`,
  `lightning`, `jsonargparse[signatures]`, `awkward`, `uproot`, `h5py`, `jax`, `pylorentz`,
  `numpy`, `scipy`, `lmfit`, `matplotlib`, `corner`, `torchmetrics`, `pyyaml`, `tqdm`, `wandb`.
