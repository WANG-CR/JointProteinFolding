#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/b/bengioy/chuanrui/backup/alphafold/output_miniprotein/slurm_log/inverse_8layer.out
#SBATCH --error=/scratch/b/bengioy/chuanrui/backup/alphafold/output_miniprotein/slurm_log/inverse_8layer.err

# 1. Activate environment
ENV_NAME=fold3
module load cuda/11.2
source activate /scratch/b/bengioy/chuanrui/envs/fold3
echo env done

# 2. Training
WORK_DIR=/scratch/b/bengioy/chuanrui/backup/alphafold
DATA_DIR=/scratch/b/bengioy/chuanrui/backup/database
TRAIN_DIR=$DATA_DIR/miniprotein/train
VALID_DIR=$DATA_DIR/miniprotein/valid
OUTPUT_DIR=$WORK_DIR/output_miniprotein

# high learning rate, low WD
TORCH_DISTRIBUTED_DEBUG=INFO srun python train_invfold.py $TRAIN_DIR $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/replace_esmFold_16+8.yml \
    --precision 32 --gpus 4 --log_every_n_steps 25 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version inverse_8layer \
    --wandb_project miniprotein \
    --train_epoch_len 500 \

