#! /bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --mem=186G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=3:00:00
#SBATCH --exclusive
#SBATCH --output=/home/shichenc/scratch/antibody/alphafold/output/slurm_log/base_debug_v1.out
#SBATCH --error=/home/shichenc/scratch/antibody/alphafold/slurm_log/base_debug_v1.err
#SBATCH --qos=unkillable

ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done

# 2. debug with 8 samples.
WORK_DIR=$SCRATCH/antibody/alphafold
DATA_DIR=$SCRATCH/database
TRAIN_DIR=$DATA_DIR/database/pdb/20220319_99_True_All__4_train
VALID_DIR=$DATA_DIR/database/pdb/20220319_99_True_All__4_valid
OUTPUT_DIR=$WORK_DIR/output

srun python train_antibody.py $TRAIN_DIR $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2022 \
    --yaml_config_preset yaml_config/base.yml \
    --precision 16 --gpus 4 --log_every_n_steps 10 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version antibody2 \
    --wandb_project cath_gen \
    --deepspeed_config_path deepspeed_config.json \
    --train_epoch_len 1000 \