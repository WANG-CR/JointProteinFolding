#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=186G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/baseline_finetune_beluga_1node.out
#SBATCH --error=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/baseline_finetune_beluga_1node.err
#SBATCH --qos=unkillable

ENV_NAME=cpu1 # your environment name
# module load cuda/11.2 # load cuda modules if neede
source activate $ENV_NAME
echo env done

# 2. set up path
WORK_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding #current path
DATA_DIR=/home/chuanrui/scratch/database #database path
TRAIN_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
OUTPUT_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/output/finetune


# 3. execute inference script
TORCH_DISTRIBUTED_DEBUG=DETAIL srun python train_joint_freezeF.py $TRAIN_DIR \
    --output_dir $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/joint_finetune/esm3B+8inv_samlldim_4ddp.yml \
    --resume_from_ckpt_backward output/inverse/checkpoint/epoch203-step50999-val_loss=1.595.ckpt \
    --resume_model_weights_only true \
    --precision 32  --log_every_n_steps 5 \
    --train_epoch_len 1000 \
    --accelerator cpu \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version baseline_finetune_beluga_1node \
    --wandb_project debug_beluga \
