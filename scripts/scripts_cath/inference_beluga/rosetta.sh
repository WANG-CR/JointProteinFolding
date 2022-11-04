#! /bin/bash

# 1. Activate environement
ENV_NAME=antibody_gen
module load cuda/11.4
source activate $ENV_NAME
echo env done
WORK_DIR=$SCRATCH/antibody/alphafold
DATA_DIR=$SCRATCH/database
# 2. batch inference

# YAML_CONFIG_PRESET=default
# VERSION=d_nostop_seqgrad_v1
# CKPT_NAME=epoch27-step6999-val_loss=2.447.ckpt

YAML_CONFIG_PRESET=default
VERSION=d_v1
CKPT_NAME=epoch34-step17499-val_loss=2.193.ckpt

python predict_rosetta.py \
    $DATA_DIR/database/pdb/rosetta-renum \
    $WORK_DIR/output/cath_gen/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --is_antibody true \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax true \
    --seed 42 \