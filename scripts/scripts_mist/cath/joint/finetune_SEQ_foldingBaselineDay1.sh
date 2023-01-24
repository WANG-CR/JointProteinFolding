#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b/bengioy/chuanrui/backup/alphafold/output/slurm_log/finetune_SEQ_foldingBaselineDay1.out
#SBATCH --error=/scratch/b/bengioy/chuanrui/backup/alphafold/output/slurm_log/finetune_SEQ_foldingBaselineDay1.err

# 1. Activate environment
ENV_NAME=fold3
module load cuda/11.2
source activate /scratch/b/bengioy/chuanrui/envs/fold3
echo env done

# 2. Training
WORK_DIR=/scratch/b/bengioy/chuanrui/backup/alphafold
DATA_DIR=/scratch/b/bengioy/chuanrui/backup/database
TRAIN_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
FASTA_DIR=$DATA_DIR/uniref50/fasta
OUTPUT_DIR=$WORK_DIR/output

# joint baseline 1 configuration: fp32, lm finetune, small set of sequence data, weight decay
TORCH_DISTRIBUTED_DEBUG=INFO srun python finetune_joint.py $TRAIN_DIR \
    --fasta_dir $FASTA_DIR \
    --output_dir $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/joint.yml \
    --precision 32 --gpus 4 --log_every_n_steps 25 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version finetune_SEQ_foldingBaselineDay1 \
    --wandb_project mist_pf \
    --train_epoch_len 2000 \
    --resume_model_weights_only true \
    --resume_from_ckpt_f $OUTPUT_DIR/mist_pf/replace-replace_baseline_fp32_v4_WD/checkpoints/epoch51-step12999-val_loss=2.708.ckpt \
    --resume_from_ckpt_g $OUTPUT_DIR/mist_pf/inverse-inverse_baseline_fp32_v2_WD/checkpoints/epoch201-step50499-val_loss=1.620.ckpt \
