#!/bin/bash
#SBATCH --partition=norm
#SBATCH --job-name=oct-segmenter       # Job name
#SBATCH --mail-type=END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --cpus-per-task=2              # Number of CPUs/hyperthreads to request
#SBATCH --mem=4gb                      # Memory to request
#SBATCH --time=01:00:00                # Time limit hrs:min:sec
#SBATCH --output=oct-segmenter-%j.out  # Standard output log
#SBATCH --error=oct-segmenter-%j.err   # Standard error log

SIF_PATH="/data/$USER/oct-segmenter.sif"
TRAIN_DIR="/data/$USER/train_dir"
OUTPUT_DIR="/data/$USER/output_dir"

export SINGULARITY_BINDPATH="/data/$USER,/lscratch"
module load singularity
$SIF_PATH partition -i $INPUT_DIR -o $OUTPUT_DIR --training 0.3 --validation 0.5 --test 0.2
