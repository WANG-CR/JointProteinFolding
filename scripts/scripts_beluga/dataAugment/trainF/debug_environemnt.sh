#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=186G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/trainF_augment/debug_frame.out
#SBATCH --error=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/trainF_augment/debug_frame.err
#SBATCH --qos=main


# ENV_NAME=cpu1 # your environment name
# module load cuda/11.2 # load cuda modules if neede
source activate ~/scratch/envs/cpu_concrete_2/
echo env done

# 2. set up path
WORK_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding #current path
DATA_DIR=/home/chuanrui/scratch/database #database path
TRAIN_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/toydata/multichain_predict
VALID_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/toydata/multichain_predict
OUTPUT_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/output/trainF_augment


# 3. execute inference script
TORCH_DISTRIBUTED_DEBUG=DETAIL srun python train_esmfold_from_pretrained.py $TRAIN_DIR \
    --output_dir $OUTPUT_DIR \
    --val_data_dir $VALID_DIR \
    --seed 2024 \
    --yaml_config_preset yaml_config/joint_finetune/debug_bs1.yml \
    --precision 32  --log_every_n_steps 5 \
    --train_epoch_len 50 \
    --accelerator cpu \
