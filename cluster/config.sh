# Cluster settings for swimplm. Edit the first block; the scripts in cluster/
# source this file (always run them from the repository root).

# ---------------------------------------------------------------------------
# 1. Site settings (edit these)
# ---------------------------------------------------------------------------
# JetClass root directory, containing train_100M/, val_5M/ and test_20M/.
export JETCLASS_DIR="/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass"
# Where training runs, embeddings, comparison results and logs go.
export WORK_DIR="$PWD/work"
# SLURM partitions for GPU jobs (training, embedding) and CPU jobs (toys).
export GPU_PARTITION="gpu"
export CPU_PARTITION="shared"
# Extra sbatch options for every job, e.g. "--account=my_lab".
export SBATCH_EXTRA=""
# Conda environment created by cluster/setup_env.sh.
export ENV_NAME="swimplm"
# Module providing nvcc for the Falkon GPU build and CUDA at run time
# (e.g. "cuda/12.4.1-fasrc01"); leave empty if not needed.
export CUDA_MODULE=""

# How to activate the environment inside a job.
activate_env() {
    if [ -n "$CUDA_MODULE" ]; then module load "$CUDA_MODULE"; fi
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
}

# ---------------------------------------------------------------------------
# 2. Experiment settings
# ---------------------------------------------------------------------------
export TRAIN_CONFIG="configs/jetclass/jetClass_parT_simCLR_T0.5.yaml"
export EMBEDDING="$WORK_DIR/jetclass_embedding.npz"
export MAX_EVENTS=500000          # jets embedded from the JetClass test set

# Toys: JetClass labels are 0 QCD, 1 top, 2 W, 3 Z, 4 H.
export SIG_LABEL=1
export N_BKG=20000                # expected background jets per toy
export N_SIG=200                  # expected signal jets per toy (scaled by SIGNAL_SCALES)
export N_REF=40000                # expected reference jets per toy
export N_TOYS=50
export SIGNAL_SCALES="0 0.5 1 2"
export METHODS="sparker falkon swim swim-uniform"

# Hyperparameter scan (cluster/submit_hparam_scan.sh): one comparison job per value.
export SWIM_L2_SCAN="0.01 0.1 1 10 100"
export FALKON_LAMBDA_SCAN="1e-7 1e-5 1e-3"
export SPARKER_WIDTH_SCAN="0.5 1 2"
export SCAN_SIGNAL_SCALES="0 1"

# Compute nodes usually have no internet: log W&B offline (sync later with `wandb sync`).
export WANDB_MODE=offline
