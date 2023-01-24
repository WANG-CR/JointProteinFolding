#! /bin/bash
#SBATCH -D /panfs/users/Xcwang/JointProteinFolding
#SBATCH -J folding
#SBATCH --get-user-env
#SBATCH --partition=workq
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=112
#SBATCH --time=1:00:00
#SBATCH --reservation=25496
#SBATCH --output=train_foldingModel_example_multinode.8.56.out
#SBATCH --error=train_foldingModel_example_multinode.8.56.err

source ~/scratch/pf_cpu/bin/activate

# 2. set up path
# you can also change the dataset to miniprotein dataset
WORK_DIR=/home/Xcwang/scratch/JointProteinFolding #current path
DATA_DIR=/home/Xcwang/scratch/database/miniprotein #database path
TRAIN_DIR=$DATA_DIR/train # train data path
VALID_DIR=$DATA_DIR/valid # test data path
OUTPUT_DIR=$WORK_DIR/output #path of the output files

export NUMEXPR_MAX_THREADS=112
export NUMEXPR_NUM_THREADS=56
export OMP_NUM_THREADS=56

# 3. execute training script
# parameter list: 
#   yaml_config_preset: configuration file of the model
#   precision: 16 or 32
#   gpus: 1 or 4 or more
#   train_epoch_len: number of samples in each epoch

TORCH_DISTRIBUTED_DEBUG=INFO srun python train_protein.py $TRAIN_DIR $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/replace_esmFold_midLR.yml \
    --precision 32  --log_every_n_steps 50 \
    --train_epoch_len 1000 \
    --accelerator cpu \
    --devices 8 \
