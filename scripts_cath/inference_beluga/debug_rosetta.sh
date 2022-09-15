#! /bin/bash

# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done
WORK_DIR=$SCRATCH/antibody/alphafold
DATA_DIR=$SCRATCH/database
# 2. batch inference
###################
### 

YAML_CONFIG_PRESET=default
VERSION=sabdab0618_H3_CDR_2022
CKPT_NAME=epoch30-step15499-val_loss=2.450.ckpt

python predict_rosetta.py \
    /home/tjhec/scratch/database/antibody/sabdab/pdb/test_test \
    $WORK_DIR/output/cath_gen/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --version $VERSION \
    --is_antibody true \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/debug-${YAML_CONFIG_PRESET}-${VERSION} \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \
    --seed 42 \