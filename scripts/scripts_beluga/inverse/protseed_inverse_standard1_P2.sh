#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=10
#SBATCH --mem=186G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/protseed_inverse_standard1_P2.out
#SBATCH --error=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/protseed_inverse_standard1_P2.err
#SBATCH --qos=main

ENV_NAME=pf2 # your environment name
# module load cuda/11.2 # load cuda modules if needed
source activate $ENV_NAME
echo env done

# 2. debug with 8 samples.
WORK_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding #current path
DATA_DIR=/home/chuanrui/scratch/database/structure_datasets/cath42/standard
TRAIN_DIR=$DATA_DIR/train
VALID_DIR=$DATA_DIR/validation
OUTPUT_DIR=$WORK_DIR/output/inverse


# batched training if possible
TORCH_DISTRIBUTED_DEBUG=DETAIL srun python train_invfold.py $TRAIN_DIR $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/baseline_inverse/protseed1.yml \
    --precision 32  --log_every_n_steps 5 \
    --gpus 4 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version protseed_inverse_standard1_P2 \
    --wandb_project debug_beluga \
    --resume_from_ckpt output/inverse/debug_beluga/protseed1-protseed_inverse_standard1/checkpoints/epoch06-step15770-val_loss=1.809.ckpt \
