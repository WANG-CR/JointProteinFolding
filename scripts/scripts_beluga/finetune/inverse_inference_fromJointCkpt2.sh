#! /bin/bash


ENV_NAME=cpu1 # your environment name
# module load cuda/11.2 # load cuda modules if neede
source activate $ENV_NAME
echo env done

# 2. set up path
# you can also change the dataset to miniprotein dataset
WORK_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding #current path
DATA_DIR=/home/chuanrui/scratch/database #database path
TRAIN_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
# TEST_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_test
TEST_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/toydata/sample_100
OUTPUT_DIR=$WORK_DIR/output/finetune #path of the output files

# 3. execute training script
# parameter list: 
#   yaml_config_preset: configuration file of the model
#   precision: 16 or 32
#   gpus: 1 or 4 or more
#   train_epoch_len: number of samples in each epoch
CHECKPOINT=output/finetune/debug_beluga/esm3B+8inv_samlldim_4ddp-baseline_finetune_beluga_1node_run3/checkpoints/epoch07-step399-val_loss=2.918.ckpt

python inverseinference.py \
    $TEST_DIR \
    --resume_from_ckpt_backward $CHECKPOINT \
    --is_antibody false \
    --yaml_config_preset yaml_config/joint_finetune/esm3B+8inv_samlldim.yml \
    --output_dir $OUTPUT_DIR/inference/finetuned_model_epoch04-step249-val_loss=2.966 \
    --model_device cpu \
    --no_recycling_iters 3 \
    --relax false \
    --seed 42 \
    --ema false \
    --name_length 7 \
    --is_joint_ckpt true \
