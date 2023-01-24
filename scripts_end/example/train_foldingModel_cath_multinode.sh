#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4 
#SBATCH --cpus-per-task=10 
#SBATCH --mem=186G 
#SBATCH --exclusive
#SBATCH --time=20:00
#SBATCH --output=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/output/slurm_log/multinode_debug.out
#SBATCH --error=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/output/slurm_log/multinode_debug.err

# 1. activate environment
ENV_NAME=cpu1 # your environment name
# module load cuda/11.2 # load cuda modules if needed
source activate $ENV_NAME
echo env done

# 2. set up path
# you can also change the dataset to miniprotein dataset
WORK_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding #current path
DATA_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/database #database path
TRAIN_DIR=$DATA_DIR/train # train data path
VALID_DIR=$DATA_DIR/valid # test data path
OUTPUT_DIR=$WORK_DIR/output #path of the output files

# 3. execute training script
# parameter list: 
#   yaml_config_preset: configuration file of the model
#   precision: 16 or 32
#   train_epoch_len: number of samples in each epoch
#   accelerator: neccesary to specify during distributed training. options are "cpu", "gpu", "auto"
#   num_nodes: number of node, defaut = 1
#   devices: number of devices, only specify during distributed training. If we use 2 node, each node contains 4 gpus, than devices = 8.

TORCH_DISTRIBUTED_DEBUG=INFO srun python train_protein.py $TRAIN_DIR $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/replace_esmFold_midLR.yml \
    --precision 32  --log_every_n_steps 50 \
    --train_epoch_len 1000 \
    --accelerator cpu \
    --devices 2 \
    --num_nodes 2 \
