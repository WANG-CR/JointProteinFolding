#! /bin/bash

# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done

EXAMPLE_DIR=$SCRATCH/biofold/example_data
TRAIN_DIR=$EXAMPLE_DIR/training/train
VALID_DIR=$EXAMPLE_DIR/training/valid
OUTPUT_DIR=$SCRATCH/biofold/output
CONFIG_PRESET=vanilla


EXAMPLE_DIR=$SCRATCH/biofold/database
TRAIN_DIR=$EXAMPLE_DIR/pdb/20220319_99_True_All__4_train
VALID_DIR=$EXAMPLE_DIR/pdb/20220319_99_True_All__4_valid
OUTPUT_DIR=$SCRATCH/biofold/output
CONFIG_PRESET=vanilla

python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/$CONFIG_PRESET \
    2021-12-31 \
    --config_preset $CONFIG_PRESET \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --log_lr \
    --checkpoint_every_epoch \
    --seed 2022 \
    --deepspeed_config_path deepspeed_config_scc.json \
    --precision 16 \
    --gpus 1 --replace_sampler_ddp=True \
    --script_modules false \
    --wandb \
    --wandb_entity scc \
    --wandb_id debug \

python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/$CONFIG_PRESET \
    2021-12-31 \
    --config_preset $CONFIG_PRESET \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --log_lr \
    --checkpoint_every_epoch \
    --seed 2022 \
    --precision 16 \
    --gpus 1 --replace_sampler_ddp=True \
    --script_modules false