#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=oct-segmenter       # Job name
#SBATCH --mail-type=END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=64gb                     # Memory to request
#SBATCH --gres=gpu:k80:1	       # Request a k20x
#SBATCH --time=1:30:00                 # Time limit hrs:min:sec
#SBATCH --output=oct-segmenter-benchmark-%j.out  # Standard output log
#SBATCH --error=oct-segmenter-benchmark-%j.err   # Standard error log

SIF_PATH="/data/$USER/oct-segmenter-0.3.0.sif"
SIF_CMD="singularity run --nv $SIF_PATH"
INPUT_DIR="/data/$USER/WayneOCTimages_Sorted"
PARTITION_DIR="/data/$USER/partition"
TRAINING_INPUT_DIR="$PARTITION_DIR/training"
VALIDATION_INPUT_DIR="$PARTITION_DIR/validation"
TRAINING_HDF5_DIR="."
TRAINING_OUTPUT_DIR="training_output"

module load singularity
. /usr/local/current/singularity/app_conf/sing_binds
mkdir -p $PARTITION_DIR
mkdir -p $TRAINING_OUTPUT_DIR

echo "PARTITION"
echo "-----------------------------------------"
$SIF_PATH partition -i $INPUT_DIR -o $PARTITION_DIR --training 0.3 --validation 0.5 --test 0.2

echo "GENERATE TRAINING"
echo "-----------------------------------------"
$SIF_PATH generate training --training-input-dir $TRAINING_INPUT_DIR --validation-input-dir $VALIDATION_INPUT_DIR -w -o $TRAINING_HDF5_DIR

echo "TRAIN"
echo "-----------------------------------------"
$SIF_PATH train -i $TRAINING_HDF5_DIR/training_dataset.hdf5 -o $TRAINING_OUTPUT_DIR -c training-config.json

