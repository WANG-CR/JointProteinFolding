#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/b/bengioy/chuanrui/backup/alphafold/output_miniprotein/train_jointModel_example.out
#SBATCH --error=/scratch/b/bengioy/chuanrui/backup/alphafold/output_miniprotein/train_jointModel_example.err

# 1. Activate environment
ENV_NAME=fold3
module load cuda/11.2
source activate $ENV_NAME
echo env done

# 2. set up path
# difference from folding and inverseFolding: additional FASTA_DIR which contains fasta file
WORK_DIR=/scratch/b/bengioy/chuanrui/backup/alphafold
DATA_DIR=/scratch/b/bengioy/chuanrui/backup/database
TRAIN_DIR=$DATA_DIR/miniprotein/train
VALID_DIR=$DATA_DIR/miniprotein/valid
FASTA_DIR=$DATA_DIR/fasta/miniprotein_uniref50_20221110
OUTPUT_DIR=$WORK_DIR/output_miniprotein


# 3. execute training script
# difference with folding and inverseFolding:
#   * you can resume foldingModel f from checkpoint
#   * you can resume inverseFoldingModel g from checkpoint

TORCH_DISTRIBUTED_DEBUG=INFO srun python train_joint.py $TRAIN_DIR \
    --fasta_dir $FASTA_DIR \
    --output_dir $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/replace_esmFold_16+16.yml \
    --precision 32 --gpus 1 --log_every_n_steps 25 \
    --wandb true \
    --wandb_entity chuanrui \
    --wandb_version train_jointModel_example \
    --wandb_project example \
    --train_epoch_len 500 \
    --resume_model_weights_only true \
    # --resume_from_ckpt_f $OUTPUT_DIR/miniprotein/replace_esmFold_16layer-esmfold_fp32_v2_lowLR/checkpoints/epoch78-step9874-val_loss=0.742.ckpt \
    # --resume_from_ckpt_g $OUTPUT_DIR/miniprotein/replace_esmFold_16+16-inverse_16layer/checkpoints/epoch62-step7874-val_loss=0.511.ckpt