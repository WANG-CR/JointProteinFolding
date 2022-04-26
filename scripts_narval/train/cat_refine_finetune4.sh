#! /bin/bash
#SBATCH --account=ctb-lcharlin
#SBATCH --cpus-per-task=12
#SBATCH --mem=498G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/home/shichenc/scratch/biofold/output/slurm_log/cat_refine_finetune_v4.out
#SBATCH --error=/home/shichenc/scratch/biofold/output/slurm_log/cat_refine_finetune_v4.err
#SBATCH --qos=unkillable

ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
WORK_DIR=$SCRATCH/biofold
TRAIN_DIR=$WORK_DIR/database/pdb/20220319_99_True_All__4_train
VALID_DIR=$WORK_DIR/database/pdb/20220319_99_True_All__4_valid
PRED_TRAIN_DIR=$WORK_DIR/database/pdb/cat-narval_v1_train
PRED_VALID_DIR=$WORK_DIR/database/pdb/cat-narval_v1_valid
TRAIN_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_train_merged
VALID_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_valid_merged
OUTPUT_DIR=$WORK_DIR/output

YAML_CONFIG_PRESET=cat_refine
srun python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/ \
    2021-12-31 \
    --seed 2022 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --train_embedding_dir $TRAIN_EMBED_DIR/ \
    --val_embedding_dir $VALID_EMBED_DIR/ \
    --pred_train_pdb_dir $PRED_TRAIN_DIR/ \
    --pred_val_pdb_dir $PRED_VALID_DIR/ \
    --precision 16 --gpus 4 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version v4_finetune \
    --wandb_project wandb_biofold \
    --deepspeed_config_path deepspeed_config_scc.json \
    --sabdab_summary_file $WORK_DIR/database/info/20220319_99_True_All__4_sabdab_summary.tsv \
    --resume_model_weights_only true \
    --resume_from_ckpt $OUTPUT_DIR/wandb_biofold/cat-narval_v1/checkpoints/epoch49-step23799-val_loss=0.916.ckpt \
