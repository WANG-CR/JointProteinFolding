#! /bin/bash
#SBATCH --account=ctb-lcharlin
#SBATCH --cpus-per-task=12
#SBATCH --mem=498G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/shichenc/scratch/biofold/output/slurm_log/inference.out
#SBATCH --error=/home/shichenc/scratch/biofold/output/slurm_log/inference.err
#SBATCH --qos=unkillable

# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done
WORK_DIR=$SCRATCH/biofold
EMBED_DIR1=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_train_merged
EMBED_DIR2=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_valid_merged

# 2. batch inference
YAML_CONFIG_PRESET=cat
VERSION=narval_v1
CKPT_NAME=epoch49-step23799-val_loss=0.916.ckpt

python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/20220319_99_True_All__4_train.fasta \
    0 \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/database/pdb/${YAML_CONFIG_PRESET}-${VERSION}_train \
    --residue_embedding_dir $EMBED_DIR1/ \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \

python batchrun_pretrained_biofold.py \
    $WORK_DIR/database/fasta/merged/20220319_99_True_All__4_valid.fasta \
    0 \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/database/pdb/${YAML_CONFIG_PRESET}-${VERSION}_valid \
    --residue_embedding_dir $EMBED_DIR2/ \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \