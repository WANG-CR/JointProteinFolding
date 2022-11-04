#! /bin/bash

# narval
ENV_NAME=cath_gen
module load cuda/11.4
source activate $ENV_NAME

# =============================================================================
# download and generate datasets
# =============================================================================
DATABASE_DIR=$SCRATCH/structure_datasets/sabdab
python scripts_sabdab/create_database.py \
    $DATABASE_DIR/download \
    $DATABASE_DIR/pdb \
    $DATABASE_DIR/fasta \
    $DATABASE_DIR/info \

# =============================================================================
# cluster
# =============================================================================

mkdir $DATABASE_DIR/cluster
mmseqs easy-cluster \
    $DATABASE_DIR/fasta/20220831_Fv_4_All_cdr/cdrs.fasta \
    $DATABASE_DIR/cluster/cdrs \
    $DATABASE_DIR/cluster \
    --min-seq-id 0.6 -c 0.8 --cov-mode 0

# mmseqs easy-cluster \
#     $DATABASE_DIR/fasta/20220831_Fv_4_All_cdr/cdrh3.fasta \
#     $DATABASE_DIR/cluster/cdrh3 \
#     $DATABASE_DIR/cluster \
#     --min-seq-id 0.4 -c 0.8 --cov-mode 1

# mmseqs easy-cluster \
#     $DATABASE_DIR/fasta/20220618_Fv_4_All_cdr/cdrh2.fasta \
#     $DATABASE_DIR/cluster/cdrh2 \
#     $DATABASE_DIR/cluster \
#     --min-seq-id 0.4 -c 0.8 --cov-mode 1

# mmseqs easy-cluster \
#     $DATABASE_DIR/fasta/20220618_Fv_4_All_cdr/cdrh1.fasta \
#     $DATABASE_DIR/cluster/cdrh1 \
#     $DATABASE_DIR/cluster \
#     --min-seq-id 0.4 -c 0.8 --cov-mode 1

# mmseqs easy-cluster \
#     $DATABASE_DIR/fasta/20220618_Fv_4_All_cdr/cdrl3.fasta \
#     $DATABASE_DIR/cluster/cdrl3 \
#     $DATABASE_DIR/cluster \
#     --min-seq-id 0.4 -c 0.8 --cov-mode 1

# mmseqs easy-cluster \
#     $DATABASE_DIR/fasta/20220618_Fv_4_All_cdr/cdrl2.fasta \
#     $DATABASE_DIR/cluster/cdrl2 \
#     $DATABASE_DIR/cluster \
#     --min-seq-id 0.4 -c 0.8 --cov-mode 1

# mmseqs easy-cluster \
#     $DATABASE_DIR/fasta/20220618_Fv_4_All_cdr/cdrl1.fasta \
#     $DATABASE_DIR/cluster/cdrl1 \
#     $DATABASE_DIR/cluster \
#     --min-seq-id 0.4 -c 0.8 --cov-mode 1

# =============================================================================
# split sabdab into train/valid/test set according to mmseqs clustering results
# min_seq_id = 0.6
# --cdr_idx 0 means we process CDRs clustering data
# =============================================================================

DATABASE_VERSION=20220831_Fv_4_All
python scripts_sabdab/process_dataset.py \
    $DATABASE_DIR/pdb \
    $DATABASE_DIR/fasta \
    $DATABASE_DIR/info \
    $DATABASE_VERSION \
    --cdr_idx 0 \
    --cluster_res $DATABASE_DIR/cluster/cdrs_cluster.tsv \
    --seed 2022 \
    --train_ratio 0.9 \
    --valid_ratio 0.05 \
