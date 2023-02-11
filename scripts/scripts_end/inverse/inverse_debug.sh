#! /bin/bash
#SBATCH -D /panfs/users/Xcwang/JointProteinFolding
#SBATCH -J folding
#SBATCH --get-user-env
#SBATCH --partition=extq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=1:00:00
#SBATCH --reservation=25517
#SBATCH --output=log/debug_inverse_1node.out
#SBATCH --error=log/debug_inverse_1node.err

source ~/scratch/pf_cpu/bin/activate

# 2. set up path
# you can also change the dataset to miniprotein dataset
WORK_DIR=/home/Xcwang/scratch/JointProteinFolding #current path
DATA_DIR=/home/Xcwang/scratch/database/structure_datasets/cath/processed #database path
TRAIN_DIR=$DATA_DIR/top_split_512_2023_0.01_0.04_train # train data path
VALID_DIR=$DATA_DIR/top_split_512_2023_0.01_0.04_valid # test data path
OUTPUT_DIR=$WORK_DIR/output/inverse #path of the output files


export NUMEXPR_MAX_THREADS=100
export NUMEXPR_NUM_THREADS=100
export OMP_NUM_THREADS=100

# 3. execute training script
# parameter list: 
#   yaml_config_preset: configuration file of the model
#   precision: 16 or 32
#   gpus: 1 or 4 or more
#   train_epoch_len: number of samples in each epoch

TORCH_DISTRIBUTED_DEBUG=DETAIL srun python train_invfold.py $TRAIN_DIR $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/baseline_inverse/inverse.yml \
    --precision 32  --log_every_n_steps 50 \
    --train_epoch_len 100 \
    # --train_epoch_len 7680 \
    # --accelerator cpu \
    # --devices 48 \
    # --num_nodes 1 \
    # --wandb true \
    # --wandb_entity chuanrui \
    # --wandb_version cath_baseline_bs16_epochlen7680_step165_P5 \
    # --wandb_project debug_endeavour \
    # --resume_from_ckpt /home/Xcwang/scratch/JointProteinFolding/output_cath/debug_endeavour/baseline_8layer_3B_largebs-cath_baseline_bs16_epochlen7680_step165_P5/checkpoints/epoch216-step2169-val_loss=2.236.ckpt \
