#! /bin/bash
# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done

# 2. debug with 8 samples.
WORK_DIR=$SCRATCH/biofold
TRAIN_DIR=$WORK_DIR/example_data/training/train
VALID_DIR=$WORK_DIR/example_data/training/valid
TRAIN_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_train_merged
VALID_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_valid_merged
PRED_TRAIN_DIR=$WORK_DIR/database/pdb/cat-narval_v1_train
PRED_VALID_DIR=$WORK_DIR/database/pdb/cat-narval_v1_valid
OUTPUT_DIR=$WORK_DIR/output

YAML_CONFIG_PRESET=cat_refine
python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
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
    --precision 16 --gpus 1 --log_every_n_steps 1 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version test_code \
    --wandb_project wandb_biofold \
    --deepspeed_config_path deepspeed_config_scc.json \
    --sabdab_summary_file $WORK_DIR/database/info/20220319_99_True_All__4_sabdab_summary.tsv \

# 3. debug with the whole train set.
WORK_DIR=$SCRATCH/biofold
TRAIN_DIR=$WORK_DIR/database/pdb/20220319_99_True_All__4_train
VALID_DIR=$WORK_DIR/database/pdb/20220319_99_True_All__4_valid
TRAIN_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_train_merged
VALID_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_valid_merged
PRED_TRAIN_DIR=$WORK_DIR/database/pdb/cat-narval_v1_train
PRED_VALID_DIR=$WORK_DIR/database/pdb/cat-narval_v1_valid
OUTPUT_DIR=$WORK_DIR/output

YAML_CONFIG_PRESET=cat_refine
python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
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
    --precision 16 --gpus 1 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version test_code \
    --wandb_project wandb_biofold \
    --deepspeed_config_path deepspeed_config_scc.json \
    --sabdab_summary_file $WORK_DIR/database/info/20220319_99_True_All__4_sabdab_summary.tsv \

YAML_CONFIG_PRESET=vanilla
srun python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/ \
    2021-12-31 \
    --seed 2022 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --precision 16 --gpus 1 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version test_code \
    --wandb_project wandb_biofold \
    --deepspeed_config_path deepspeed_config_scc.json \
    --sabdab_summary_file $WORK_DIR/database/info/20220319_99_True_All__4_sabdab_summary.tsv \

YAML_CONFIG_PRESET=esm1b_cat
srun python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/ \
    2021-12-31 \
    --seed 2022 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --train_embedding_dir $TRAIN_EMBED_DIR/ \
    --val_embedding_dir $VALID_EMBED_DIR/ \
    --precision 16 --gpus 1 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version test_code \
    --wandb_project wandb_biofold \
    --deepspeed_config_path deepspeed_config_scc.json \
    --sabdab_summary_file $WORK_DIR/database/info/20220319_99_True_All__4_sabdab_summary.tsv \


#### oas 
WORK_DIR=$SCRATCH/biofold
TRAIN_DIR=$WORK_DIR/example_data/training/train
VALID_DIR=$WORK_DIR/example_data/training/valid
TRAIN_EMBED_DIR=$WORK_DIR/pretrained_embeddings/oas_antiberty/20220319_99_True_All__4_train_feat_merged
VALID_EMBED_DIR=$WORK_DIR/pretrained_embeddings/oas_antiberty/20220319_99_True_All__4_valid_feat_merged
TRAIN_ATTN_DIR=$WORK_DIR/pretrained_embeddings/oas_antiberty/20220319_99_True_All__4_train_attn
VALID_ATTN_DIR=$WORK_DIR/pretrained_embeddings/oas_antiberty/20220319_99_True_All__4_valid_attn
OUTPUT_DIR=$WORK_DIR/output

YAML_CONFIG_PRESET=replace
srun python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/ \
    2021-12-31 \
    --seed 2022 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --train_embedding_dir $TRAIN_EMBED_DIR/ \
    --val_embedding_dir $VALID_EMBED_DIR/ \
    --precision 16 --gpus 1 --log_every_n_steps 1 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version test_code \
    --wandb_project wandb_biofold \
    --deepspeed_config_path deepspeed_config_scc.json \
    --sabdab_summary_file $WORK_DIR/database/info/20220319_99_True_All__4_sabdab_summary.tsv \

YAML_CONFIG_PRESET=cat_attn
srun python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/ \
    2021-12-31 \
    --seed 2022 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --train_embedding_dir $TRAIN_EMBED_DIR/ \
    --val_embedding_dir $VALID_EMBED_DIR/ \
    --train_attn_dir $TRAIN_ATTN_DIR/ \
    --val_attn_dir $VALID_ATTN_DIR/ \
    --precision 16 --gpus 1 --log_every_n_steps 1 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version test_code \
    --wandb_project wandb_biofold \
    --deepspeed_config_path deepspeed_config_scc.json \
    --sabdab_summary_file $WORK_DIR/database/info/20220319_99_True_All__4_sabdab_summary.tsv \