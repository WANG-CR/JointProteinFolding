#! /bin/bash

# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done
WORK_DIR=$SCRATCH/biofold
EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/rosetta_merged

# 2. batch inference
YAML_CONFIG_PRESET=wm5000_long_esm
VERSION=narval_v1
CKPT_NAME=epoch55-step26655-val_loss=0.91.ckpt
python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/rosetta.fasta \
    0 \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/rosetta \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    --residue_embedding_dir $EMBED_DIR/ \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \

YAML_CONFIG_PRESET=wm5000_long
VERSION=narval_v1
CKPT_NAME=epoch59-step28559-val_loss=0.92.ckpt
python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/rosetta.fasta \
    0 \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/rosetta \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    --residue_embedding_dir $EMBED_DIR/ \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \