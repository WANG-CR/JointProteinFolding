#! /bin/bash
#SBATCH --account=ctb-lcharlin
#SBATCH --cpus-per-task=12
#SBATCH --mem=498G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/home/shichenc/scratch/biofold/output/slurm_log/cat_v4.out
#SBATCH --error=/home/shichenc/scratch/biofold/output/slurm_log/cat_v4.err
#SBATCH --qos=unkillable

ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
WORK_DIR=$SCRATCH/biofold
TRAIN_DIR=$WORK_DIR/database/pdb/20220319_99_True_All__4_train
VALID_DIR=$WORK_DIR/database/pdb/20220319_99_True_All__4_valid
TRAIN_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_train_merged
VALID_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_valid_merged
OUTPUT_DIR=$WORK_DIR/output

YAML_CONFIG_PRESET=cat
srun python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/ \
    2021-12-31 \
    --seed 2025 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --train_embedding_dir $TRAIN_EMBED_DIR/ \
    --val_embedding_dir $VALID_EMBED_DIR/ \
    --precision 16 --gpus 4 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version v4 \
    --wandb_project wandb_biofold \
    --deepspeed_config_path deepspeed_config_scc.json \
    --sabdab_summary_file $WORK_DIR/database/info/20220319_99_True_All__4_sabdab_summary.tsv \
