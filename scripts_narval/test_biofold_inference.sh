#! /bin/bash

# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done
WORK_DIR=$SCRATCH/biofold

# 2. Prediction
YAML_CONFIG_PRESET=wm5000
VERSION=narval_v1
CKPT_NAME=epoch77-step37127-val_loss=0.91.ckpt
python run_pretrained_biofold.py \
    $WORK_DIR/example_data/inference/4nnp.fasta \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/20220319_99_True_All__4_sabdab \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/example_data/inference \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \

YAML_CONFIG_PRESET=wm10000
VERSION=narval_v1
CKPT_NAME=epoch76-step36651-val_loss=0.91.ckpt
python run_pretrained_biofold.py \
    $WORK_DIR/example_data/inference/3mxw.fasta \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/rosetta \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/example_data/inference \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \

# 3. batch inference
YAML_CONFIG_PRESET=wm10000
VERSION=narval_v1
CKPT_NAME=epoch76-step36651-val_loss=0.91.ckpt
python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/rosetta.fasta \
    0 \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/rosetta \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \


YAML_CONFIG_PRESET=wm5000
VERSION=narval_v1
CKPT_NAME=epoch76-step36651-val_loss=0.91.ckpt
python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/rosetta.fasta \
    0 \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/rosetta \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \