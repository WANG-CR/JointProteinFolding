#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=48
#SBATCH --mem=498G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --exclusive
#SBATCH --output=/home/chuanrui/scratch/research/b/JointProteinFolding/log/debug/debug_gpu2.out
#SBATCH --error=/home/chuanrui/scratch/research/b/JointProteinFolding/log/debug/debug_gpu2.err
#SBATCH --qos=main

# 1. Activate environement
ENV_NAME=pf2
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME
module load cuda/11.4
echo env done

WORK_DIR=/home/chuanrui/scratch/research/b/JointProteinFolding #current path
DATA_DIR=/home/chuanrui/scratch/data #database path
# TRAIN_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
# VALID_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
TRAIN_DIR=/home/chuanrui/scratch/data/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=/home/chuanrui/scratch/data/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
OUTPUT_DIR=/home/chuanrui/scratch/research/b/JointProteinFolding/output/debug



TORCH_DISTRIBUTED_DEBUG=DETAIL srun python train_esmfold_from_pretrained.py $TRAIN_DIR \
    --output_dir $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/data_augment/baseline_bs1_12block.yml \
    --precision 32  --gpus 2 --log_every_n_steps 2 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version debug_deepspeed \
    --wandb_project data_augment_gpu \
    --deepspeed_config_path /home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/deepspeed_config.json \
    # --resume_from_ckpt /home/Xcwang/scratch/JointProteinFolding/output_cath/debug_endeavour/baseline_8layer_3B_largebs-cath_baseline_bs16_epochlen7680_step165_P5/checkpoints/epoch216-step2169-val_loss=2.236.ckpt \
