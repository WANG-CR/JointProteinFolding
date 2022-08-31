#! /bin/bash

# narval
ENV_NAME=cath_gen
module load cuda/11.4
source activate $ENV_NAME

# =============================================================================
# download and generate datasets
# =============================================================================
DATABASE_DIR=$SCRATCH/structure_datasets/sabdab
python scripts_sabdab/antigen_agnostic/create_database.py \
    $DATABASE_DIR/download \
    $DATABASE_DIR/pdb \
    $DATABASE_DIR/fasta \
    $DATABASE_DIR/info \

# =============================================================================
# cluster: 769 clusters
# =============================================================================

mkdir $DATABASE_DIR/cluster
mmseqs easy-cluster \
    $DATABASE_DIR/fasta/20220831_99_True_All__4_cdr/cdrs.fasta \
    $DATABASE_DIR/cluster/cdrs \
    $DATABASE_DIR/cluster \
    --min-seq-id 0.6 -c 0.8 --cov-mode 0

# =============================================================================
# split sabdab into train/valid/test set according to mmseqs clustering results
# min_seq_id = 0.6
# --cdr_idx 0 means we process CDRs clustering data
# =============================================================================

DATABASE_VERSION=20220831_99_True_All__4
python scripts_sabdab/antigen_agnostic/process_dataset.py \
    $DATABASE_DIR/pdb \
    $DATABASE_DIR/fasta \
    $DATABASE_DIR/info \
    $DATABASE_VERSION \
    --cdr_idx 0 \
    --cluster_res $DATABASE_DIR/cluster/cdrs_cluster.tsv \
    --seed 2022 \
    --train_ratio 0.9 \
    --valid_ratio 0.05 \

# =============================================================================
# Get test cdrs, and then cluster again with min_seq_id = 0.4, and then copy the data again
python scripts_sabdab/antigen_agnostic/reduce_testset.py \
    --input_dir $DATABASE_DIR/pdb/20220831_99_True_All__4_cdrs_0.9_0.05_test \
    --output_dir $DATABASE_DIR/fasta/20220831_99_True_All__4_cdr_0.9_0.05_test \

# remaining: 32
mmseqs easy-cluster \
    $DATABASE_DIR/fasta/20220831_99_True_All__4_cdr_0.9_0.05_test/cdrs.fasta \
    $DATABASE_DIR/cluster/test_cdrs \
    $DATABASE_DIR/cluster \
    --min-seq-id 0.4 -c 0.8 --cov-mode 0

python scripts_sabdab/antigen_agnostic/reduce_testset.py \
    --input_dir $DATABASE_DIR/pdb/20220831_99_True_All__4_cdrs_0.9_0.05_test \
    --output_dir $DATABASE_DIR/pdb/20220831_99_True_All__4_cdrs_0.9_0.05_test_reduced \
    --cluster $DATABASE_DIR/cluster/test_cdrs_cluster.tsv \
# =============================================================================