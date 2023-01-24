#! /bin/bash
#SBATCH -D /panfs/users/Xcwang/JointProteinFolding
#SBATCH -J folding
#SBATCH --get-user-env
#SBATCH --partition=extq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=12:00:00
#SBATCH --reservation=25496
#SBATCH --output=log/debug_tmloss_1.out
#SBATCH --error=log/debug_tmloss_1.err

source ~/scratch/pf_cpu/bin/activate

# 2. set up path
# you can also change the dataset to miniprotein dataset
WORK_DIR=/home/Xcwang/scratch/JointProteinFolding #current path
DATA_DIR=/home/Xcwang/scratch/database/miniprotein #database path
TRAIN_DIR=$DATA_DIR/train # train data path
VALID_DIR=$DATA_DIR/valid # test data path
OUTPUT_DIR=$WORK_DIR/output #path of the output files

# export NUMEXPR_MAX_THREADS=8
# export NUMEXPR_NUM_THREADS=8
# export OMP_NUM_THREADS=8

export NUMEXPR_MAX_THREADS=48
export NUMEXPR_NUM_THREADS=48
export OMP_NUM_THREADS=48

# 3. execute training script
# parameter list: 
#   yaml_config_preset: configuration file of the model
#   precision: 16 or 32
#   gpus: 1 or 4 or more
#   train_epoch_len: number of samples in each epoch

TORCH_DISTRIBUTED_DEBUG=DETAIL srun python train_protein.py $TRAIN_DIR $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/baseline/baseline_8layer.yml \
    --precision 32  --log_every_n_steps 50 \
    --train_epoch_len 4000 \
    --accelerator cpu \
    --devices 1 \
    --num_nodes 1 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version tm_loss_debug \
    --wandb_project debug_endeavour \
