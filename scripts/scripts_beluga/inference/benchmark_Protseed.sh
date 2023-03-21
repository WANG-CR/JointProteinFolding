#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=186G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --exclusive
#SBATCH --output=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/benchmark/Protseedbaseline_domain_epoch12.out
#SBATCH --error=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/benchmark/Protseedbaseline_domain_epoch12.err
#SBATCH --qos=main

ENV_NAME=pf2 # your environment name
# module load cuda/11.2 # load cuda modules if needed
source activate $ENV_NAME
echo env done

# 2. debug with 8 samples.
WORK_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding #current path
DATA_DIR=/home/chuanrui/scratch/database/structure_datasets/cath42/standard
TRAIN_DIR=$DATA_DIR/train
VALID_DIR=$DATA_DIR/validation
TEST_DIR=$DATA_DIR/test
CHECKPOINT=$WORK_DIR/output/inverse/debug_beluga/protseed1-protseed_inverse_standard1_P2/checkpoints/epoch12-step29288-val_loss=1.793.ckpt

python inverseinference.py \
    $TEST_DIR \
    --resume_from_ckpt_backward $CHECKPOINT \
    --is_antibody false \
    --yaml_config_preset yaml_config/baseline_inverse/protseed1.yml \
    --model_device cuda:0 \
    --no_recycling_iters 1 \
    --relax false \
    --seed 42 \
    --ema true \


TEST_DIR=$DATA_DIR/short_94 
python inverseinference.py \
    $TEST_DIR \
    --resume_from_ckpt_backward $CHECKPOINT \
    --is_antibody false \
    --yaml_config_preset yaml_config/baseline_inverse/protseed1.yml \
    --model_device cuda:0 \
    --no_recycling_iters 1 \
    --relax false \
    --seed 42 \
    --ema true \


TEST_DIR=$DATA_DIR/single  
python inverseinference.py \
    $TEST_DIR \
    --resume_from_ckpt_backward $CHECKPOINT \
    --is_antibody false \
    --yaml_config_preset yaml_config/baseline_inverse/protseed1.yml \
    --model_device cuda:0 \
    --no_recycling_iters 1 \
    --relax false \
    --seed 42 \
    --ema true \


