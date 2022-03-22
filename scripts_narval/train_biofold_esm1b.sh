#! /bin/bash
#SBATCH --account=ctb-lcharlin
#SBATCH --cpus-per-task=12
#SBATCH --mem=498G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=/home/shichenc/scratch/biofold/output/slurm_log/esm1b_cat_v1.out
#SBATCH --error=/home/shichenc/scratch/biofold/output/slurm_log/esm1b_cat_v1.err
#SBATCH --qos=unkillable

#! /bin/bash
# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
WORK_DIR=$SCRATCH/biofold
TRAIN_DIR=$WORK_DIR/database/pdb/20220319_99_True_All__4_train
VALID_DIR=$WORK_DIR/database/pdb/20220319_99_True_All__4_valid
TRAIN_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_train_merged
VALID_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_valid_merged
OUTPUT_DIR=$WORK_DIR/output

CONFIG_PRESET=esm1b_cat
python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/$CONFIG_PRESET \
    2021-12-31 \
    --config_preset $CONFIG_PRESET \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --train_embedding_dir $TRAIN_EMBED_DIR/ \
    --val_embedding_dir $VALID_EMBED_DIR/ \
    --sabdab_summary_file $WORK_DIR/database/info/20220319_99_True_All__4_sabdab_summary.tsv \
    --log_lr \
    --checkpoint_every_epoch \
    --seed 2022 \
    --deepspeed_config_path deepspeed_config_scc.json \
    --precision 16 \
    --gpus 4 --replace_sampler_ddp=True \
    --script_modules false \
    --wandb \
    --wandb_id v1 \
    --wandb_project biofold \
    --experiment_name esm1b_cat_narval24h \

