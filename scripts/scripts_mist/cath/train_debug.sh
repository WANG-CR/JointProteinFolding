#! /bin/bash
# 1. Activate environement
ENV_NAME=cath_gen
module load cuda/11.4
source activate $ENV_NAME
echo env done

# 2. debug with 8 samples.
TRAIN_DIR=$SCRATCH/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=$SCRATCH/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
TEST_DIR=$SCRATCH/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_test
OUTPUT_DIR=$SCRATCH/cath_gen/output

python train_cath.py $TRAIN_DIR $OUTPUT_DIR \
    --ss_file $SCRATCH/structure_datasets/cath/raw/ss_annotation_31885.pkl \
    --val_data_dir $VALID_DIR \
    --seed 2022 \
    --yaml_config_preset yaml_config/default.yml \
    --precision 16 --gpus 1 --log_every_n_steps 1 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version debug \
    --wandb_project cath_gen \
    --deepspeed_config_path deepspeed_config.json \
    --train_epoch_len 100 \

srun python train_cath.py $TEST_DIR $OUTPUT_DIR \
    --ss_file $SCRATCH/structure_datasets/cath/raw/ss_annotation_31885.pkl \
    --val_data_dir $VALID_DIR \
    --seed 2022 \
    --yaml_config_preset yaml_config/default.yml \
    --precision 16 --gpus 4 --log_every_n_steps 1 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version debug \
    --wandb_project cath_gen \
    --deepspeed_config_path deepspeed_config.json \
    --train_epoch_len 1000 \