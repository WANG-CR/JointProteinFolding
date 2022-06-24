#! /bin/bash

# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done
WORK_DIR=$SCRATCH/af2gen

# 3. Biofold
# 3.1 Single fasta inference
YAML_CONFIG_PRESET=gen_abag
VERSION=v3
CKPT_NAME=epoch18-step10715-val_loss=2.355.ckpt
python run_pretrained_af2gen.py \
    $WORK_DIR/sabdab/pdb/20220618_Fv_4_All_cdrh3_test/7t8w.pdb \
    $WORK_DIR/output/wandb_af2gen/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/example_data/inference \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax true \
    --seed 2022 \


YAML_CONFIG_PRESET=gen
VERSION=v3
CKPT_NAME=epoch30-step15065-val_loss=2.125.ckpt
python run_pretrained_biofold.py \
    $WORK_DIR/example_data/inference/7rco.fasta \
    $WORK_DIR/output/wandb_af2gen/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pred_pdb_dir $WORK_DIR/database/pdb/20220603_99_True_All__4_sabdab \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/example_data/inference \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax true \
    --seed 2022 \


