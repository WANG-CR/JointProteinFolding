#! /bin/bash

# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.2
source activate $SCRATCH/envs/$ENV_NAME
echo env done

EXAMPLE_DIR=$SCRATCH/biofold/example_data

# 2. With pre-alignments
python run_pretrained_openfold.py \
    $EXAMPLE_DIR/7n4l_cat30x.fasta \
    --use_precomputed_alignments $EXAMPLE_DIR/alignments \
    --output_dir $EXAMPLE_DIR \
    --model_device cuda:0 \
    --param_path $SCRATCH/biofold/pretrained_weights/alphafold2_params/params_model_1.npz \
    --no_recycling_iters 3 \
    --relax false \
