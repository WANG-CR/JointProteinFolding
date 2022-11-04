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
    /home/tjhec/scratch/database/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh3_test \
    $WORK_DIR/output/cath_gen/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --version $VERSION \
    --is_antibody true \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/debug-${YAML_CONFIG_PRESET}-${VERSION} \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \
    --seed 42 \



# 2. batch inference
###################
### 
YAML_CONFIG_PRESET=default_H3
VERSION=sabdab0618_H3_H3_2022
CKPT_NAME=epoch37-step18999-val_loss=3.670.ckpt

python predict_rosetta.py \
    /home/tjhec/scratch/database/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh3_test \
    $WORK_DIR/output/cath_gen/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --version $VERSION \
    --is_antibody true \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/debug-${YAML_CONFIG_PRESET}-${VERSION} \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \
    --seed 42 \


# YAML_CONFIG_PRESET=default
# VERSION=d_nostop_seqgrad_v1
# CKPT_NAME=epoch27-step6999-val_loss=2.447.ckpt

# YAML_CONFIG_PRESET=default
# VERSION=d_v1
# CKPT_NAME=epoch34-step17499-val_loss=2.193.ckpt

# YAML_CONFIG_PRESET=default
# VERSION=seed2022
# CKPT_NAME=epoch36-step18499-val_loss=2.072.ckpt

# python predict_rosetta.py \
#     /home/tjhec/scratch/database/database/pdb/rosetta-renum \
#     $WORK_DIR/output/cath_gen/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
#     --is_antibody true \
#     --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
#     --output_dir $WORK_DIR/output/rosetta_benchmark/debug-${YAML_CONFIG_PRESET}-${VERSION} \
#     --model_device cuda:0 \
#     --no_recycling_iters 3 \
#     --relax false \
#     --seed 42 \

#     $DATA_DIR/sabdab/pdb/20220831_99_True_All__4_cdrs_0.9_0.05_test_reduced \
# /home/tjhec/scratch/database/database/pdb/rosetta-renum
        # --relax true \