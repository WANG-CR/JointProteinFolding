#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/b/bengioy/chuanrui/backup/alphafold/output_miniprotein/slurm_log/esmfold_fp32_v2_lowLR.out
#SBATCH --error=/scratch/b/bengioy/chuanrui/backup/alphafold/output_miniprotein/slurm_log/esmfold_fp32_v2_lowLR.err

# v3: apply weight decay to deminish NaN problem
ENV_NAME=fold3
module load cuda/11.2
source activate /scratch/b/bengioy/chuanrui/envs/fold3
echo env done

# 2. debug with 8 samples.
WORK_DIR=/scratch/b/bengioy/chuanrui/backup/alphafold
DATA_DIR=/scratch/b/bengioy/chuanrui/backup/database
TRAIN_DIR=$DATA_DIR/miniprotein/train
VALID_DIR=$DATA_DIR/miniprotein/valid
OUTPUT_DIR=$WORK_DIR/output_miniprotein

# low learning rate, high weight decay = 1e-5, change seed
TORCH_DISTRIBUTED_DEBUG=INFO srun python train_protein.py $TRAIN_DIR $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/replace_esmFold_16layer.yml \
    --precision 32 --gpus 4 --log_every_n_steps 25 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version esmfold_fp32_v2_lowLR \
    --wandb_project miniprotein \
    --train_epoch_len 500 \
