#! /bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --mem=186G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=3:00:00
#SBATCH --exclusive
#SBATCH --output=/home/shichenc/scratch/cath_gen/output/slurm_log/default_short_v1.out
#SBATCH --error=/home/shichenc/scratch/cath_gen/output/slurm_log/default_short_v1.err
#SBATCH --qos=unkillable

ENV_NAME=cath_gen
module load cuda/11.4
source activate $ENV_NAME
echo env done

# 2. debug with 8 samples.
TRAIN_DIR=$SCRATCH/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=$SCRATCH/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
TEST_DIR=$SCRATCH/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_test
OUTPUT_DIR=$SCRATCH/cath_gen/output

srun python train_cath.py $TRAIN_DIR $OUTPUT_DIR \
    --ss_file $SCRATCH/structure_datasets/cath/raw/ss_annotation_31885.pkl \
    --val_data_dir $VALID_DIR \
    --seed 2022 \
    --yaml_config_preset yaml_config/default.yml \
    --precision 16 --gpus 4 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version b_short_v1 \
    --wandb_project cath_gen \
    --deepspeed_config_path deepspeed_config.json \
    --train_epoch_len 2000 \