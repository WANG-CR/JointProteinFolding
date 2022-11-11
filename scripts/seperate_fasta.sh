#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/b/bengioy/chuanrui/backup/alphafold/output/slurm_log/seperate_fasta.out
#SBATCH --error=/scratch/b/bengioy/chuanrui/backup/alphafold/output/slurm_log/seperate_fasta.err

# 1. Activate environment
ENV_NAME=fold3
module load cuda/11.2
source activate /scratch/b/bengioy/chuanrui/envs/fold3
echo env done

# # 2. Seperate fasta
# DATA_DIR=/scratch/b/bengioy/chuanrui/backup/database
# FASTA_DIR=$DATA_DIR/uniref50.fasta
# OUTPUT_DIR=$DATA_DIR/fasta/uniref50/raw

# srun python seperate_fasta.py --fasta_path $FASTA_DIR --output_dir $OUTPUT_DIR


# 2. Seperate fasta
DATA_DIR=/scratch/b/bengioy/chuanrui/backup/database
FASTA_DIR=$DATA_DIR/miniprotein_20221110_uniref50.fasta
OUTPUT_DIR=$DATA_DIR/fasta/miniprotein_uniref50_20221110

srun python scripts/seperate_fasta.py --fasta_path $FASTA_DIR --output_dir $OUTPUT_DIR