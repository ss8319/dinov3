#!/bin/bash
# Tip: set a meaningful job name at submission time, e.g.
#   sbatch --job-name=fusion_mlp_hidden_layer_2 train_fusion_slurm.sh
# The %x token below will be replaced by the job name
#SBATCH --job-name=test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=39G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -e  # Exit on error

# Set SLURM environment variables for DINOv3 distributed setup (single GPU)
export SLURM_PROCID=0
export SLURM_LOCALID=0  
export SLURM_NODEID=0
export SLURM_NTASKS=1
export SLURM_JOB_NUM_NODES=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Set paths
DINOV3_ROOT="/home/ssim0068/multimodal-AD/src/mri/dinov3"
DATA_ROOT="/home/ssim0068/data/ADNI_v2"
WEIGHTS_PATH="${DINOV3_ROOT}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
# WEIGHTS_PATH="${DINOV3_ROOT}/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# Dataset size control: "tiny", "small", or "full"
DATASET_SIZE="${DATASET_SIZE:-full}"

# Set CSV filenames based on dataset size
if [ "$DATASET_SIZE" == "tiny" ]; then
    TRAIN_CSV="ad_train_tiny.csv"
    VAL_CSV="ad_val_tiny.csv"
    TEST_CSV="ad_test_tiny.csv"
    BATCH_SIZE=4
    MAX_ITERS=50
elif [ "$DATASET_SIZE" == "small" ]; then
    TRAIN_CSV="ad_train_small.csv"
    VAL_CSV="ad_val_small.csv"
    TEST_CSV="ad_test_small.csv"
    BATCH_SIZE=8
    MAX_ITERS=100
else
    # full dataset
    TRAIN_CSV="ad_train.csv"
    VAL_CSV="ad_val.csv"
    TEST_CSV="ad_test.csv"
    BATCH_SIZE=32
    MAX_ITERS=500
fi 

OUTPUT_DIR="${DINOV3_ROOT}/runs/adni_${DATASET_SIZE}_test_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "==================================="
echo "DINOv3 ADNI Smoke Test"
echo "==================================="
echo "DINOv3 Root: ${DINOV3_ROOT}"
echo "Data Root: ${DATA_ROOT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "==================================="

# Navigate to multimodal root for uv environment
cd /home/ssim0068/multimodal-AD

# Run DINOv3 log regression with minimal data (smoke test)
echo "Starting DINOv3 log regression test..."
echo ""

PYTHONPATH=${DINOV3_ROOT} uv run python -m dinov3.eval.log_regression \
  model.hub_repo_dir=${DINOV3_ROOT} \
  model.hub_model=dinov3_vitb16 \
  model.pretrained_weights="${WEIGHTS_PATH}" \
  train.dataset="ADNI:split=TRAIN:root=${DATA_ROOT}/images_mni305:extra=${DATA_ROOT}/csvs:csv_filename=${TRAIN_CSV}" \
  train.val_dataset="ADNI:split=VAL:root=${DATA_ROOT}/images_mni305:extra=${DATA_ROOT}/csvs:csv_filename=${VAL_CSV}" \
  eval.test_dataset="ADNI:split=TEST:root=${DATA_ROOT}/images_mni305:extra=${DATA_ROOT}/csvs:csv_filename=${TEST_CSV}" \
  train.batch_size=${BATCH_SIZE} \
  train.train_features_device=cpu \
  train.train_dtype=float32 \
  train.max_train_iters=${MAX_ITERS} \
  output_dir="${OUTPUT_DIR}" \
  train.num_workers=2 \
  eval.batch_size=${BATCH_SIZE} \
  eval.num_workers=2

echo ""
echo "==================================="
echo "Test completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "End time: $(date)"
echo "==================================="
