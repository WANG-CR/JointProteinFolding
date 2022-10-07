#! /bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=10
#SBATCH --mem=186G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --output=/home/chuanrui/scratch/research/ProteinFolding/alphafold/output/slurm_log/inverse_debug.out
#SBATCH --error=/home/chuanrui/scratch/research/ProteinFolding/alphafold/output/slurm_log/inverse_debug.err
#SBATCH --qos=unkillable

ENV_NAME=pf2
module load CUDA/11.3.1
source activate $ENV_NAME
echo env done

# 2. debug with 8 samples.
WORK_DIR=$SCRATCH/research/ProteinFolding/alphafold
DATA_DIR=$SCRATCH/database
TRAIN_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
OUTPUT_DIR=$WORK_DIR/output

srun python train_invfold.py --train_data_dir $TRAIN_DIR \
    --output_dir $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/inverse_debug.yml \
    --precision 16 --gpus 1 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version v1 \
    --wandb_project pf_toy \
    --deepspeed_config_path deepspeed_config.json \
    --train_epoch_len 1000 \