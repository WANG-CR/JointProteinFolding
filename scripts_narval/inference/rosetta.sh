#! /bin/bash

# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done
WORK_DIR=$SCRATCH/biofold
EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/rosetta_merged

# 2. batch inference
YAML_CONFIG_PRESET=replace
VERSION=narval_v1
CKPT_NAME=epoch54-step26179-val_loss=0.919.ckpt
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

YAML_CONFIG_PRESET=msa
VERSION=narval_v1
CKPT_NAME=epoch48-step23323-val_loss=0.915.ckpt
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

YAML_CONFIG_PRESET=decay
VERSION=narval_v1
CKPT_NAME=epoch55-step26655-val_loss=0.924.ckpt
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

YAML_CONFIG_PRESET=cat_refine
VERSION=narval_v1
CKPT_NAME=epoch48-step23323-val_loss=0.907.ckpt
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


############# cat #############
YAML_CONFIG_PRESET=cat
VERSION=narval_v1
CKPT_NAME=epoch49-step23799-val_loss=0.916.ckpt
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