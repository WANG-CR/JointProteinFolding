#! /bin/bash
#SBATCH -D /panfs/users/Xcwang/JointProteinFolding
#SBATCH -J folding
#SBATCH --get-user-env
#SBATCH --partition=extq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=102
#SBATCH --time=1:00:00
#SBATCH --reservation=25496
#SBATCH --output=log/exp1_inference_debug.out
#SBATCH --error=log/exp1_inference_debug.err

source ~/scratch/pf_cpu/bin/activate

# 2. set up path
# you can also change the dataset to miniprotein dataset
WORK_DIR=/home/Xcwang/scratch/JointProteinFolding #current path
DATA_DIR=/home/Xcwang/scratch/database/miniprotein #database path
TRAIN_DIR=$DATA_DIR/train # train data path
VALID_DIR=$DATA_DIR/valid # test data path
TEST_DIR=$DATA_DIR/test # test data path
OUTPUT_DIR=$WORK_DIR/output #path of the output files


# 3. execute training script
# parameter list: 
#   yaml_config_preset: configuration file of the model
#   precision: 16 or 32
#   gpus: 1 or 4 or more
#   train_epoch_len: number of samples in each epoch


python predict_protein.py \
    $TEST_DIR \
    --resume_from_ckpt /home/Xcwang/scratch/JointProteinFolding/output/debug_endeavour/baseline_8layer-miniprotein_baseline_1_P2/checkpoints/epoch34-step4374-val_loss=0.692.ckpt \
    --is_antibody false \
    --yaml_config_preset yaml_config/baseline/baseline_8layer.yml \
    --output_dir $OUTPUT_DIR/inference/debug-inference \
    --model_device cpu \
    --no_recycling_iters 3 \
    --relax false \
    --seed 42 \
    --ema false \
    --name_length 14 \