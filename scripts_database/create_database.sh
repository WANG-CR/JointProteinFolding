#! /bin/bash

# narval start #
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
# narval end #

# local #
ENV_NAME=biofold
source activate $ENV_NAME
DATABASE_DIR=$HOME/database
python scripts_database/create_database.py \
    $DATABASE_DIR/download \
    $DATABASE_DIR/pdb \
    $DATABASE_DIR/fasta \
    $DATABASE_DIR/info
# local #

