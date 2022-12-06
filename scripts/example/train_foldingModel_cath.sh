#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/b/bengioy/chuanrui/backup/alphafold/output/slurm_log/train_foldingModel_example.out
#SBATCH --error=/scratch/b/bengioy/chuanrui/backup/alphafold/output/slurm_log/train_foldingModel_example.err

# 1. activate environment
ENV_NAME=fold3 # your environment name
module load cuda/11.2 # load cuda modules if needed
source activate $ENV_NAME
echo env done

# 2. set up path
# you can also change the dataset to miniprotein dataset
WORK_DIR=/scratch/b/bengioy/chuanrui/backup/alphafold #current path
DATA_DIR=/scratch/b/bengioy/chuanrui/backup/database #database path
TRAIN_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train # train data path
VALID_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid # test data path
OUTPUT_DIR=$WORK_DIR/output #path of the output files

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
    --precision 16 --gpus 4 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version train_foldingModel_example \
    --wandb_project example \
    --train_epoch_len 1000 \
