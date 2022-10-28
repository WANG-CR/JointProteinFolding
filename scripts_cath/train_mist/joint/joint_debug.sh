#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/b/bengioy/chuanrui/backup/alphafold/output/slurm_log/joint_seperate_debug_amp16_finetuneLM.out
#SBATCH --error=/scratch/b/bengioy/chuanrui/backup/alphafold/output/slurm_log/joint_seperate_debug_amp16_finetuneLM.err

# 1. Activate environment
ENV_NAME=fold3
module load cuda/11.2
source activate /scratch/b/bengioy/chuanrui/envs/fold3
echo env done

# 2. Training
WORK_DIR=/scratch/b/bengioy/chuanrui/backup/alphafold
DATA_DIR=/scratch/b/bengioy/chuanrui/backup/database
TRAIN_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
OUTPUT_DIR=$WORK_DIR/output

TORCH_DISTRIBUTED_DEBUG=INFO srun python train_joint_debug.py $TRAIN_DIR $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/joint.yml \
    --precision 16 --gpus 4 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version joint_seperate_debug_amp16_finetuneLM \
    --wandb_project mist_pf \
    --train_epoch_len 1000 \