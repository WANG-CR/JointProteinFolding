#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=186G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --output=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/benchmark/debug_folding_benchmarking.out
#SBATCH --error=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/benchmark/debug_folding_benchmarking.err
#SBATCH --qos=main

ENV_NAME=pf2
source activate $ENV_NAME

WORK_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding #current path
DATA_DIR=/home/chuanrui/scratch/database/structure_datasets/

BENCHMARK=CASP14
GT_PDB=$DATA_DIR/CASP14_esm/pdb
FASTA=$DATA_DIR/CASP14_esm/fasta
PREDICT_PDB=$DATA_DIR/CASP14_esm/predict_esm_v0
LOG=$WORK_DIR/benchmark/esmfold
MODEL=esm2_v0

# convert pdb to fasta
python convert_pdb_to_fasta.py --input_dir $GT_PDB --output_dir $FASTA

# make predictions
python folding_predict.py --fasta_dir $FASTA \
    --output_dir $PREDICT_PDB \
    --max_length 1024 \
    --chunk_size 512 \
    --model $MODEL \

# evaluate tm-score
python evalute_tm.py --gt_dir $GT_PDB \
    --predict_dir $PREDICT_PDB \
    --log_dir $LOG/${MODEL}_${BENCHMARK}.txt \