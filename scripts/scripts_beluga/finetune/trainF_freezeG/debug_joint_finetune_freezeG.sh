#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=10
#SBATCH --mem=186G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/trainF_freezeG/debug_finetune.out
#SBATCH --error=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/trainF_freezeG/debug_finetune.err
#SBATCH --qos=main


ENV_NAME=cpu1 # your environment name
# module load cuda/11.2 # load cuda modules if neede
source activate $ENV_NAME
echo env done

# 2. set up path
WORK_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding #current path
DATA_DIR=/home/chuanrui/scratch/database #database path
TRAIN_DIR=$DATA_DIR/structure_datasets/cath42/pdbChains/train
VALID_DIR=$DATA_DIR/structure_datasets/cath42/pdbChains/valid
OUTPUT_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/output/trainF_freezeG


# 3. execute inference script
TORCH_DISTRIBUTED_DEBUG=DETAIL srun python train_joint_freezeG_noEMA.py $TRAIN_DIR \
    --output_dir $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/joint_finetune/debug_bs1.yml \
    --resume_from_ckpt_backward output/inverse/checkpoint_for_inference/inverse_block8_bs1_lr5e-4-baseline_standard_lr5e-4_block8_EMA/epoch161-step40499-val_loss=1.827.ckpt \
    --resume_model_weights_only true \
    --precision 32  --log_every_n_steps 5 \
    --train_epoch_len 50 \
    --accelerator cpu \
