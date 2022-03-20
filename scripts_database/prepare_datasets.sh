#! /bin/bash

# narval
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME

# 1. create database
DATABASE_DIR=$SCRATCH/biofold/database
python scripts_database/create_database.py \
    $DATABASE_DIR/download \
    $DATABASE_DIR/pdb \
    $DATABASE_DIR/fasta \
    $DATABASE_DIR/info

# 2. split sabdab into train/valid set according to release date,
#    and generate merged fasta files for further batch inference, and msa generation
python scripts_database/process_dataset.py \
    $DATABASE_DIR/pdb \
    $DATABASE_DIR/fasta \
    $DATABASE_DIR/info \
    20220319_99_True_All__4 \
    --merge_rosetta true \
    --merge_therapeutics true

# 4. generate pretraining LM embeddings from esm1b

