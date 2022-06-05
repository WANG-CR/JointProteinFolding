#! /bin/bash

# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done
WORK_DIR=$SCRATCH/af2gen

# 3. Biofold
# 3.1 Single fasta inference
YAML_CONFIG_PRESET=gen
VERSION=v1
CKPT_NAME=epoch33-step16523-val_loss=0.400.ckpt
python run_pretrained_biofold.py \
    $WORK_DIR/example_data/inference/7n4l.fasta \
    $WORK_DIR/output/wandb_af2gen/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pred_pdb_dir $WORK_DIR/database/pdb/20220603_99_True_All__4_sabdab \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/example_data/inference \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \
    --seed 2024 \

python run_pretrained_biofold.py \
    $WORK_DIR/example_data/inference/3mxw.fasta \
    $WORK_DIR/output/wandb_af2gen/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pred_pdb_dir $WORK_DIR/database/pdb/rosetta \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/example_data/inference \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \
# 3.2 batch inference
# See scripts_narvel/inference/