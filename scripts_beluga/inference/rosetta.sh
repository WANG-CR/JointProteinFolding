#! /bin/bash

# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done
WORK_DIR=$SCRATCH/af2gen

# 2. batch inference

YAML_CONFIG_PRESET=gen
VERSION=v3
CKPT_NAME=epoch30-step15065-val_loss=2.125.ckpt
python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/rosetta.fasta \
    0 \
    $WORK_DIR/output/wandb_af2gen/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pred_pdb_dir $WORK_DIR/database/pdb/rosetta \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax true \


############# cat #############
YAML_CONFIG_PRESET=cat
VERSION=oas_v1
CKPT_NAME=epoch39-step19039-val_loss=0.938.ckpt
python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/rosetta.fasta \
    0 \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/rosetta \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    --residue_embedding_dir $EMBED_DIR2/ \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \

YAML_CONFIG_PRESET=cat
VERSION=narval_v1
CKPT_NAME=epoch49-step23799-val_loss=0.916.ckpt
python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/rosetta.fasta \
    0 \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/rosetta \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-seed5 \
    --residue_embedding_dir $EMBED_DIR/ \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \
    --seed 202205

YAML_CONFIG_PRESET=cat
VERSION=v2
CKPT_NAME=epoch46-step22371-val_loss=0.918.ckpt
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

YAML_CONFIG_PRESET=cat
VERSION=v4
CKPT_NAME=epoch45-step21895-val_loss=0.917.ckpt
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

####### finetune cat #############
YAML_CONFIG_PRESET=cat_refine
VERSION=v2_finetune
CKPT_NAME=epoch27-step13327-val_loss=0.903.ckpt
python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/rosetta.fasta \
    0 \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/rosetta \
    --pred_pdb_dir $WORK_DIR/output/rosetta_benchmark/cat-narval_v1 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    --residue_embedding_dir $EMBED_DIR/ \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \

YAML_CONFIG_PRESET=cat_refine
VERSION=finetune_v1
CKPT_NAME=epoch31-step15231-val_loss=0.903.ckpt
python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/rosetta.fasta \
    0 \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/rosetta \
    --pred_pdb_dir $WORK_DIR/output/rosetta_benchmark/cat-narval_v1 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    --residue_embedding_dir $EMBED_DIR/ \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \

YAML_CONFIG_PRESET=cat_refine
VERSION=finetune_v3
CKPT_NAME=epoch29-step14279-val_loss=0.905.ckpt
python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/rosetta.fasta \
    0 \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/rosetta \
    --pred_pdb_dir $WORK_DIR/output/rosetta_benchmark/cat-narval_v1 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    --residue_embedding_dir $EMBED_DIR/ \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \

YAML_CONFIG_PRESET=cat_refine
VERSION=finetune_v4
CKPT_NAME=epoch28-step13803-val_loss=0.903.ckpt
python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/rosetta.fasta \
    0 \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/rosetta \
    --pred_pdb_dir $WORK_DIR/output/rosetta_benchmark/cat-narval_v1 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    --residue_embedding_dir $EMBED_DIR/ \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \

YAML_CONFIG_PRESET=cat_attn
VERSION=attn_v1
CKPT_NAME=epoch35-step17135-val_loss=0.937.ckpt
python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/rosetta.fasta \
    0 \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/rosetta \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    --residue_embedding_dir $EMBED_DIR2/ \
    --residue_attn_dir  $ATTN_DIR/ \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \