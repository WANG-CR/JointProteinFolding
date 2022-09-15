#! /bin/bash
#SBATCH --account=def-tjhec
#SBATCH --cpus-per-task=10
#SBATCH --mem=186G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/home/tjhec/scratch/antibody/alphafold/output/slurm_log/seed2024.out
#SBATCH --error=/home/tjhec/scratch/antibody/alphafold/output/slurm_log/seed2024.err
#SBATCH --qos=unkillable

ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done

# 2. debug with 8 samples.
WORK_DIR=$SCRATCH/antibody/alphafold
DATA_DIR=$SCRATCH/database
TRAIN_DIR=$DATA_DIR/sabdab/pdb/20220831_99_True_All__4_cdrs_0.9_0.05_train
VALID_DIR=$DATA_DIR/sabdab/pdb/20220831_99_True_All__4_cdrs_0.9_0.05_valid
OUTPUT_DIR=$WORK_DIR/output

srun python train_antibody.py $TRAIN_DIR $OUTPUT_DIR \
    --is_antibody true \
    --val_data_dir $VALID_DIR \
    --seed 2023 \
    --yaml_config_preset yaml_config/default_H3.yml \
    --precision 16 --gpus 4 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version seed2024 \
    --wandb_project cath_gen \
    --deepspeed_config_path deepspeed_config.json \
    --train_epoch_len 2000 \