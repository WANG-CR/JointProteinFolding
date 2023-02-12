#! /bin/bash
#SBATCH -D /panfs/users/Xcwang/JointProteinFolding
#SBATCH -J folding
#SBATCH --get-user-env
#SBATCH --partition=extq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=1:00:00
#SBATCH --reservation=25517
#SBATCH --output=log/debug_inverse_1node.out
#SBATCH --error=log/debug_inverse_1node.err

ENV_NAME=cpu1 # your environment name
# module load cuda/11.2 # load cuda modules if neede
source activate $ENV_NAME
echo env done

# 2. debug with 8 samples.
WORK_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding #current path
DATA_DIR=/home/chuanrui/scratch/database #database path
TRAIN_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
OUTPUT_DIR=$WORK_DIR/output_train_esm


# 3. execute training script
# parameter list: 
#   yaml_config_preset: configuration file of the model
#   precision: 16 or 32
#   gpus: 1 or 4 or more
#   train_epoch_len: number of samples in each epoch

python coinference.py \
    $TEST_DIR \
    --resume_from_ckpt_backward /home/Xcwang/scratch/JointProteinFolding/output/inverse/inverse_folding/pretrained_ckpt/epoch203-step50999-val_loss=1.595.ckpt \
    --is_antibody false \
    --yaml_config_preset yaml_config/baseline/baseline_8layer_3B_largebs.yml \
    --output_dir $OUTPUT_DIR/inference/debug-inference \
    --model_device cpu \
    --no_recycling_iters 3 \
    --relax false \
    --seed 42 \
    --ema false \
    --name_length 7 \

# TORCH_DISTRIBUTED_DEBUG=DETAIL srun python train_invfold.py $TRAIN_DIR $OUTPUT_DIR \
#     --val_data_dir $VALID_DIR \
#     --seed 2024 \
#     --yaml_config_preset yaml_config/baseline_inverse/inverse.yml \
#     --precision 32  --log_every_n_steps 50 \
#     --train_epoch_len 100 \
    # --resume_from_ckpt /home/Xcwang/scratch/JointProteinFolding/output_cath/debug_endeavour/baseline_8layer_3B_largebs-cath_baseline_bs16_epochlen7680_step165_P5/checkpoints/epoch216-step2169-val_loss=2.236.ckpt \
