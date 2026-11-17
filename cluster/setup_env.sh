#!/bin/bash
# Create the conda environment and install swimplm, swimnetworks and Falkon.
#
#   bash cluster/setup_env.sh          # Falkon with GPU support (needs nvcc: set CUDA_MODULE)
#   FORCE_ONLY_CPU=1 bash cluster/setup_env.sh   # CPU-only Falkon
#
# Run from the repository root, ideally on a node that has the CUDA toolkit.
set -euo pipefail
source cluster/config.sh

if [ -n "$CUDA_MODULE" ]; then module load "$CUDA_MODULE"; fi
eval "$(conda shell.bash hook)"
if conda env list | grep -qE "^$ENV_NAME\s"; then
    conda env update -n "$ENV_NAME" -f environment.yml
else
    conda env create -n "$ENV_NAME" -f environment.yml
fi
conda activate "$ENV_NAME"

pip install --no-deps -e .
# Falkon has no PyPI wheel; build it from the tested commit.
pip install --no-build-isolation \
    "git+https://github.com/FalkonML/falkon.git@04f168e9d465829163d0f3512487c55f875fc45f"

python -c "import torch, falkon, swimnetworks, swimplm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('environment OK')"
