#! /bin/bash
#SBATCH -D /home/Xcwang/scratch/beluga/JointProteinFolding
#SBATCH -J folding
#SBATCH --get-user-env
#SBATCH --partition=extq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=24:00:00
#SBATCH --reservation=25643
#SBATCH --output=/home/Xcwang/scratch/beluga/JointProteinFolding/log/debug_validation.out
#SBATCH --error=/home/Xcwang/scratch/beluga/JointProteinFolding/log/debug_validation.err

# /nfs/work04/chuanrui/output/debug_trainF.err
# /panfs/users/Xcwang/JointProteinFolding
source ~/scratch/cpu1_for_endeavour/bin/activate

# 2. set up path
# you can also change the dataset to miniprotein dataset
WORK_DIR=/home/Xcwang/scratch/beluga/JointProteinFolding #current path
TRAIN_DIR=/nfs/work04/chuanrui/data/sampled_protein/sampling # train data path
VALID_DIR=/home/Xcwang/scratch/database/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid # test data path
OUTPUT_DIR=/nfs/work04/chuanrui/output/data_augment #path of the output files

export NUMEXPR_MAX_THREADS=100
export NUMEXPR_NUM_THREADS=100
export OMP_NUM_THREADS=100

# 3. execute training script
# parameter list: 
#   yaml_config_preset: configuration file of the model
#   precision: 16 or 32
#   gpus: 1 or 4 or more
#   train_epoch_len: number of samples in each epoch

TORCH_DISTRIBUTED_DEBUG=DETAIL srun python train_esmfold_from_pretrained.py $TRAIN_DIR \
    --output_dir $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/data_augment/baseline_bs2.yml \
    --precision 32  --log_every_n_steps 2 \
    --train_epoch_len 4 \
    --accelerator cpu \
    --devices 4 \
    --num_nodes 1 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version debug_validation \
    --wandb_project data_augment \
    # --resume_from_ckpt /home/Xcwang/scratch/JointProteinFolding/output_cath/debug_endeavour/baseline_8layer_3B_largebs-cath_baseline_bs16_epochlen7680_step165_P5/checkpoints/epoch216-step2169-val_loss=2.236.ckpt \
