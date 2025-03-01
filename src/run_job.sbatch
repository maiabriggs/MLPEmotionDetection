#!/bin/bash
#SBATCH --job-name=emotion_training
#SBATCH --output=/home/%u/logs/%x_%j.out  # Log file in user-specific directory
#SBATCH --error=/home/%u/logs/%x_%j.err   # Error file
#SBATCH --time=04:00:00                   # Time limit
#SBATCH --mem=8000                        # Memory allocation (8GB)
#SBATCH --cpus-per-task=2                 # Number of CPUs
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --nodes=1                         # Use 1 node

set -x
echo "Job running on ${SLURM_JOB_NODELIST}"
TIMESTAMP=$(date +'%Y-%m-%d_%H-%M-%S')
echo "Job started: $TIMESTAMP"
source ~/.bashrc
set -e
echo "Activating conda environment: mlp"
source /home/s2186747/miniconda3/etc/profile.d/conda.sh
conda activate mlp

# Arguments
DATASET_NAME="TIF"
SOURCE_DATA_DIR="/home/s2186747/TIF"
EPOCHS=20
BATCH_SIZE=64

# Create the output directory
BASE_OUTPUT_DIR="/home/s2186747/git/MLPEmotionDetection/output"
OUTPUT_NAME="$ResNet19_${DATASET_NAME}_${SLURM_JOB_ID}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${OUTPUT_NAME}"
mkdir -p "$OUTPUT_DIR"
echo "Outputs will be saved to: $OUTPUT_DIR"

# Define the scratch output directory
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
SCRATCH_OUTPUT_DIR="${SCRATCH_HOME}/output/${DATASET_NAME}_${TIMESTAMP}"

# Copy data to scratch disk
SCRATCH_DATA_DIR="/${SCRATCH_HOME}/project_data"
echo "Creating scratch data directory..."
mkdir -p $SCRATCH_DATA_DIR
echo "Copying data to scratch disk..."
rsync --archive --update --compress --progress --exclude='/afs' --exclude='/proc' "$SOURCE_DATA_DIR/" "$SCRATCH_DATA_DIR/"
echo "Data copied to scratch disk."

# Run the Python script
python /home/%u/projects/main.py \
    --dataset_name "$DATASET_NAME" \
    --dataset_path "$SCRATCH_DATA_DIR" \
    --output_dir "$SCRATCH_OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"

echo "Copying outputs from scratch to final output directory..."
rsync --archive --update --compress --progress "$SCRATCH_OUTPUT_DIR/" "$OUTPUT_DIR/"
echo "Outputs copied back to distributed filesystem."

# Archieve scratch directory
ARCHIVE_PATH="/home/${USER}/project/scratch_archives"
mkdir -p ${ARCHIVE_PATH}
ARCHIVE_FILE="${ARCHIVE_PATH}/project_data_${TIMESTAMP}.tar.gz"
echo "Archiving scratch directory to ${ARCHIVE_FILE}"
tar -czf ${ARCHIVE_FILE} -C ${SCRATCH_HOME} project_data

echo "Job completed successfully."
