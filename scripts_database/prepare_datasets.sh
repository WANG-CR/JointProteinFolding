#! /bin/bash

# narval
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME

# 1. create database
DATABASE_DIR=$SCRATCH/af2gen/sabdab
python scripts_database/create_database.py \
    $DATABASE_DIR/download \
    $DATABASE_DIR/pdb \
    $DATABASE_DIR/fasta \
    $DATABASE_DIR/info \


# 2. split sabdab into train/valid set according to release date,
#    and generate merged fasta files for further batch inference, and msa generation
DATABASE_VERSION=20220603_99_True_All__4
python scripts_database/process_dataset.py \
    $DATABASE_DIR/pdb \
    $DATABASE_DIR/fasta \
    $DATABASE_DIR/info \
    $DATABASE_VERSION \
    --merge_rosetta true \
    --merge_therapeutics true

# 4. generate pretraining LM embeddings from esm1b
ESM_MODEL_PATH=$SCRATCH/biofold/pretrained_weights/esm_params/esm1b_t33_650M_UR50S.pt
python scripts_database/extract_esm.py \
    $ESM_MODEL_PATH \
    $DATABASE_DIR/fasta/merged/${DATABASE_VERSION}_train.fasta \
    $SCRATCH/biofold/pretrained_embeddings/esm1b/${DATABASE_VERSION}_train \
    --repr_layers 33 \
    --include per_tok \

python scripts_database/extract_esm.py \
    $ESM_MODEL_PATH \
    $DATABASE_DIR/fasta/merged/${DATABASE_VERSION}_valid.fasta \
    $SCRATCH/biofold/pretrained_embeddings/esm1b/${DATABASE_VERSION}_valid \
    --repr_layers 33 \
    --include per_tok \

python scripts_database/extract_esm.py \
    $ESM_MODEL_PATH \
    $DATABASE_DIR/fasta/merged/rosetta.fasta \
    $SCRATCH/biofold/pretrained_embeddings/esm1b/rosetta \
    --repr_layers 33 \
    --include per_tok \

python scripts_database/extract_esm.py \
    $ESM_MODEL_PATH \
    $DATABASE_DIR/fasta/merged/therapeutics.fasta \
    $SCRATCH/biofold/pretrained_embeddings/esm1b/therapeutics \
    --repr_layers 33 \
    --include per_tok \

# 5. merge the generated H/L chain embeddings
python scripts_database/merge_esm.py \
    $SCRATCH/biofold/pretrained_embeddings/esm1b/${DATABASE_VERSION}_train \
    $SCRATCH/biofold/pretrained_embeddings/esm1b/${DATABASE_VERSION}_train_merged \
    --model_name esm1b

python scripts_database/merge_esm.py \
    $SCRATCH/biofold/pretrained_embeddings/esm1b/${DATABASE_VERSION}_valid \
    $SCRATCH/biofold/pretrained_embeddings/esm1b/${DATABASE_VERSION}_valid_merged \
    --model_name esm1b

python scripts_database/merge_esm.py \
    $SCRATCH/biofold/pretrained_embeddings/esm1b/rosetta \
    $SCRATCH/biofold/pretrained_embeddings/esm1b/rosetta_merged \
    --model_name esm1b

python scripts_database/merge_esm.py \
    $SCRATCH/biofold/pretrained_embeddings/esm1b/therapeutics \
    $SCRATCH/biofold/pretrained_embeddings/esm1b/therapeutics_merged \
    --model_name esm1b

# 4. generate pretraining LM embeddings from antiberty
OAS_MODEL_PATH=$SCRATCH/biofold/pretrained_weights/oas_params/IgFold/igfold_1.ckpt
python scripts_database/extract_antiberty.py \
    $DATABASE_DIR/fasta/merged/${DATABASE_VERSION}_train.fasta \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/${DATABASE_VERSION}_train_feat \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/${DATABASE_VERSION}_train_attn \
    $OAS_MODEL_PATH \
python scripts_database/merge_esm.py \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/${DATABASE_VERSION}_train_feat \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/${DATABASE_VERSION}_train_feat_merged \
    --model_name oas_antiberty

python scripts_database/extract_antiberty.py \
    $DATABASE_DIR/fasta/merged/${DATABASE_VERSION}_valid.fasta \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/${DATABASE_VERSION}_valid_feat \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/${DATABASE_VERSION}_valid_attn \
    $OAS_MODEL_PATH \
python scripts_database/merge_esm.py \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/${DATABASE_VERSION}_valid_feat \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/${DATABASE_VERSION}_valid_feat_merged \
    --model_name oas_antiberty

python scripts_database/extract_antiberty.py \
    $DATABASE_DIR/fasta/merged/rosetta.fasta \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/rosetta_feat \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/rosetta_attn \
    $OAS_MODEL_PATH \
python scripts_database/merge_esm.py \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/rosetta_feat \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/rosetta_feat_merged \
    --model_name oas_antiberty

python scripts_database/extract_antiberty.py \
    $DATABASE_DIR/fasta/merged/therapeutics.fasta \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/therapeutics_feat \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/therapeutics_attn \
    $OAS_MODEL_PATH \
python scripts_database/merge_esm.py \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/therapeutics_feat \
    $SCRATCH/biofold/pretrained_embeddings/oas_antiberty/therapeutics_feat_merged \
    --model_name oas_antiberty