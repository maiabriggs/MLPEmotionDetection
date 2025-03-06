#!/bin/bash
# set -x
# echo "Job running on ${SLURM_JOB_NODELIST}"
# TIMESTAMP=$(date +'%Y-%m-%d_%H-%M-%S')
# echo "Job started: $TIMESTAMP"
# source ~/.bashrc
# set -e
# echo "Activating conda environment: mlp"
# source /home/s2186747/miniconda3/etc/profile.d/conda.sh
# conda activate mlp

# Arguments
DATASET_NAME="TIF_DB"
SOURCE_DATA_DIR="/home/maia/Documents/MLP/mlp-cw3/MLPEmotionDetection/datasets/TIF_DB"
EPOCHS=20
BATCH_SIZE=64

# Create the output directory
BASE_OUTPUT_DIR="/home/maia/Documents/MLP/mlp-cw3/MLPEmotionDetection/output/class-test"
OUTPUT_NAME="TIF_CLASS_TEST"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${OUTPUT_NAME}"
mkdir -p "$OUTPUT_DIR"
echo "Outputs will be saved to: $OUTPUT_DIR"

# Run the Python script
python3 /home/maia/Documents/MLP/mlp-cw3/MLPEmotionDetection/src/main_with_class_balancing.py \
    --dataset_name "$DATASET_NAME" \
    --dataset_path "$SOURCE_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"

echo "Job completed successfully."
