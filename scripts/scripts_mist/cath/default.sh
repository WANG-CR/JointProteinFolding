#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/b/bengioy/chuanrui/backup/alphafold/output/slurm_log/default.out
#SBATCH --error=/scratch/b/bengioy/chuanrui/backup/alphafold/output/slurm_log/default.err

ENV_NAME=fold3
module load cuda/11.2
source activate $ENV_NAME
echo env done

# 2. debug with 8 samples.
TRAIN_DIR=$SCRATCH/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=$SCRATCH/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
TEST_DIR=$SCRATCH/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_test
OUTPUT_DIR=$SCRATCH/cath_gen/output

TORCH_DISTRIBUTED_DEBUG=INFO srun python train_protein.py $TRAIN_DIR $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/default.yml \
    --precision 16 --gpus 4 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version default \
    --wandb_project mist_pf \
    --train_epoch_len 1000 \