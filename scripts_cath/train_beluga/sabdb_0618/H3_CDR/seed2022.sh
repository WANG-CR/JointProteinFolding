#! /bin/bash
#SBATCH --account=def-tjhec
#SBATCH --cpus-per-task=10
#SBATCH --mem=186G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/home/tjhec/scratch/antibody/alphafold/output/slurm_log/sabdab0618_H3_CDR_seed2022.out
#SBATCH --error=/home/tjhec/scratch/antibody/alphafold/output/slurm_log/sabdab0618_H3_CDR_seed2022.err
#SBATCH --qos=unkillable

ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done

# 2. debug with 8 samples.
WORK_DIR=$SCRATCH/antibody/alphafold
DATA_DIR=$SCRATCH/database
TRAIN_DIR=$DATA_DIR/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh3_train
VALID_DIR=$DATA_DIR/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh3_valid
OUTPUT_DIR=$WORK_DIR/output

# srun 
srun python train_antibody.py $TRAIN_DIR $OUTPUT_DIR \
    --is_antibody true \
    --val_data_dir $VALID_DIR \
    --seed 2022 \
    --yaml_config_preset yaml_config/default.yml \
    --precision 16 --gpus 4 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version sabdab0618_H3_CDR_2022 \
    --wandb_project cath_gen \
    --deepspeed_config_path deepspeed_config.json \
    --train_epoch_len 2000 \

