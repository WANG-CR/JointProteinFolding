#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=12
#SBATCH --mem=124G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --exclusive
#SBATCH --output=/home/chuanrui/scratch/research/b/JointProteinFolding/log/debug/debug_env.out
#SBATCH --error=//home/chuanrui/scratch/research/b/JointProteinFolding/log/debug/debug_env.err
#SBATCH --qos=main

# 1. Activate environement
ENV_NAME=pf3
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME
module load cuda/11.4
echo env done

WORK_DIR=/home/chuanrui/scratch/research/b/JointProteinFolding #current path
DATA_DIR=/home/chuanrui/scratch/data #database path
TRAIN_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
OUTPUT_DIR=/home/chuanrui/scratch/research/b/JointProteinFolding/output/debug
YAML_CONFIG_PRESET=initial_training_no_msa

# 3. execute inference script
TORCH_DISTRIBUTED_DEBUG=DETAIL srun python train_joint_freezeF_noEMA.py $TRAIN_DIR \
    --output_dir $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --gpus 1 \
    --yaml_config_preset yaml_config/joint_finetune/esm3B+8inv_samlldim.yml \
    --precision 32  --log_every_n_steps 5 \
    --train_epoch_len 2 \
    # --accelerator gpu \
    # --resume_from_ckpt_backward output/inverse/checkpoint/epoch203-step50999-val_loss=1.595.ckpt \
    # --resume_model_weights_only true \
