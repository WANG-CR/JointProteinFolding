#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/b/bengioy/chuanrui/backup/alphafold/output_miniprotein/slurm_log/esmfold_fp32_v2_lowLR.out
#SBATCH --error=/scratch/b/bengioy/chuanrui/backup/alphafold/output_miniprotein/slurm_log/esmfold_fp32_v2_lowLR.err

# we are only using reconstruction loss
# in this script, we leverage the pretrained folding model and inverse folding model for miniprotein

ENV_NAME=fold3
module load cuda/11.2
source activate /scratch/b/bengioy/chuanrui/envs/fold3
echo env done


WORK_DIR=/scratch/b/bengioy/chuanrui/backup/alphafold
DATA_DIR=/scratch/b/bengioy/chuanrui/backup/database
TRAIN_DIR=$DATA_DIR/miniprotein/train
VALID_DIR=$DATA_DIR/miniprotein/valid
FASTA_DIR=$DATA_DIR/fasta/miniprotein_uniref50_20221110
OUTPUT_DIR=$WORK_DIR/output_miniprotein

TORCH_DISTRIBUTED_DEBUG=INFO srun python train_joint.py $TRAIN_DIR \
    --fasta_dir $FASTA_DIR \
    --output_dir $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/replace_esmFold_16+16.yml \
    --precision 32 --gpus 1 --log_every_n_steps 25 \
    --wandb false \
    --resume_model_weights_only true \
    --train_epoch_len 500 \
    --resume_from_ckpt_f $OUTPUT_DIR/miniprotein/replace_esmFold_16layer-esmfold_fp32_v2_lowLR/checkpoints/epoch78-step9874-val_loss=0.742.ckpt \
    --resume_from_ckpt_g $OUTPUT_DIR/miniprotein/replace_esmFold_16+16-inverse_16layer/checkpoints/epoch62-step7874-val_loss=0.511.ckpt
