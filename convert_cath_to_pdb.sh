#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=186G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --output=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/convert_cath_to_pdb.out
#SBATCH --error=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/convert_cath_to_pdb.err
#SBATCH --qos=main

ENV_NAME=pf2 # your environment name
# module load cuda/11.2 # load cuda modules if needed
source activate $ENV_NAME
echo env done

# 2. debug with 8 samples.


DATA1=/home/chuanrui/scratch/database/structure_datasets/cath42/pickle/
DATA2=/home/chuanrui/scratch/database/structure_datasets/cath42/standard/ #database path


INPUT_DIR=$DATA1/short_94.pkl
OUTPUT_DIR=$DATA2/short_94
python convert_cath_to_pdb.py --list_path $INPUT_DIR --output_dir $OUTPUT_DIR \


INPUT_DIR=$DATA1/single_chain_103.pkl
OUTPUT_DIR=$DATA2/single
python convert_cath_to_pdb.py --list_path $INPUT_DIR --output_dir $OUTPUT_DIR \


INPUT_DIR=$DATA1/test_1120.pkl
OUTPUT_DIR=$DATA2/test
python convert_cath_to_pdb.py --list_path $INPUT_DIR --output_dir $OUTPUT_DIR \


INPUT_DIR=$DATA1/train_18024.pkl
OUTPUT_DIR=$DATA2/train
python convert_cath_to_pdb.py --list_path $INPUT_DIR --output_dir $OUTPUT_DIR \


INPUT_DIR=$DATA1/validation_608.pkl
OUTPUT_DIR=$DATA2/validation
python convert_cath_to_pdb.py --list_path $INPUT_DIR --output_dir $OUTPUT_DIR \


INPUT_DIR=$DATA1/cath_nodes_21572.pkl
OUTPUT_DIR=$DATA2/all
python convert_cath_to_pdb.py --list_path $INPUT_DIR --output_dir $OUTPUT_DIR \