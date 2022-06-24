#! /bin/bash
# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done

# 2. debug with 8 samples.
WORK_DIR=$SCRATCH/af2gen
TRAIN_DIR=$WORK_DIR/example_data/training2/train
VALID_DIR=$WORK_DIR/example_data/training2/valid
OUTPUT_DIR=$WORK_DIR/output

YAML_CONFIG_PRESET=gen
python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/ \
    2021-12-31 \
    --seed 2022 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --precision 16 --gpus 1 --log_every_n_steps 1 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version debug \
    --wandb_project wandb_af2gen \
    --deepspeed_config_path deepspeed_config_scc.json \
    --sabdab_summary_file $WORK_DIR/sabdab/info/20220618_Fv_4_All_sabdab_summary.tsv \

YAML_CONFIG_PRESET=gen_abag
srun python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/ \
    2021-12-31 \
    --seed 2022 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --precision 16 --gpus 4 --log_every_n_steps 1 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version debug \
    --wandb_project wandb_af2gen \
    --deepspeed_config_path deepspeed_config_scc.json \
    --sabdab_summary_file $WORK_DIR/sabdab/info/20220618_Fv_4_All_sabdab_summary.tsv \
    --trunc_antigen true \

YAML_CONFIG_PRESET=gen_ab
srun python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/ \
    2021-12-31 \
    --seed 2022 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --precision 16 --gpus 4 --log_every_n_steps 1 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version debug \
    --wandb_project wandb_af2gen \
    --deepspeed_config_path deepspeed_config_scc.json \
    --sabdab_summary_file $WORK_DIR/sabdab/info/20220618_Fv_4_All_sabdab_summary.tsv \
    --trunc_antigen false \

YAML_CONFIG_PRESET=gen_complex
srun python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/ \
    2021-12-31 \
    --seed 2022 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --precision 16 --gpus 4 --log_every_n_steps 1 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version debug \
    --wandb_project wandb_af2gen \
    --deepspeed_config_path deepspeed_config_scc.json \
    --sabdab_summary_file $WORK_DIR/sabdab/info/20220618_Fv_4_All_sabdab_summary.tsv \
    --trunc_antigen true \
