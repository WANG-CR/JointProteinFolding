#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=186G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --exclusive
#SBATCH --output=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/joint_inference.out
#SBATCH --error=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/joint_inference.err
#SBATCH --qos=unkillable

ENV_NAME=cpu1 # your environment name
# module load cuda/11.2 # load cuda modules if neede
source activate $ENV_NAME
echo env done

# 2. set up path
WORK_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding #current path
DATA_DIR=/home/chuanrui/scratch/database #database path
TRAIN_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_train
VALID_DIR=$DATA_DIR/structure_datasets/cath/processed/top_split_512_2023_0.01_0.04_valid
TEST_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/toydata/sample_25
OUTPUT_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/toydata/generated

# 3. execute inference script
python coinference.py \
    $TEST_DIR \
    --resume_from_ckpt_backward /home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/output/inverse/checkpoint/epoch203-step50999-val_loss=1.595.ckpt \
    --is_antibody false \
    --yaml_config_preset yaml_config/joint_finetune/esm3B+8inv_samlldim.yml \
    --output_dir $OUTPUT_DIR/debug-inference \
    --model_device cpu \
    --no_recycling_iters 3 \
    --relax false \
    --seed 42 \
    --ema false \
    --name_length 7 \
