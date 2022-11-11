#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/b/bengioy/chuanrui/backup/alphafold/output/slurm_log/mini_finetune_both_foldingBaselineDay1.out
#SBATCH --error=/scratch/b/bengioy/chuanrui/backup/alphafold/output/slurm_log/mini_finetune_both_foldingBaselineDay1.err

# 1. Activate environment
ENV_NAME=fold3
module load cuda/11.2
source activate /scratch/b/bengioy/chuanrui/envs/fold3
echo env done

# 2. Training
WORK_DIR=/scratch/b/bengioy/chuanrui/backup/alphafold
DATA_DIR=/scratch/b/bengioy/chuanrui/backup/database
TRAIN_DIR=$DATA_DIR/miniprotein/train
VALID_DIR=$DATA_DIR/miniprotein/valid
FASTA_DIR=$DATA_DIR/fasta/miniprotein_uniref50_20221110
OUTPUT_DIR=$WORK_DIR/output_miniprotein

# joint baseline 1 configuration: fp32, lm finetune, small set of sequence data, weight decay
TORCH_DISTRIBUTED_DEBUG=INFO srun python train_joint.py $TRAIN_DIR \
    --fasta_dir $FASTA_DIR \
    --output_dir $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/joint.yml \
    --precision 32 --gpus 4 --log_every_n_steps 25 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version mini_finetune_both_foldingBaselineDay1 \
    --wandb_project miniprotein \
    --train_epoch_len 1000 \
    --resume_model_weights_only true \
    --resume_from_ckpt_f $OUTPUT_DIR/miniprotein/replace-replace_fp32_v3/checkpoints/epoch108-step13624-val_loss=0.745.ckpt \
    --resume_from_ckpt_g $OUTPUT_DIR/miniprotein/inverse2-inverse_fp32_v2/checkpoints/epoch46-step5874-val_loss=0.752.ckpt \
