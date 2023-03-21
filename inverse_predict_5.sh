#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=20
#SBATCH --mem=186G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/sampling_5.out
#SBATCH --error=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/slurm_log/sampling_5.err
#SBATCH --qos=main

ENV_NAME=pf2 # your environment name
module load cuda/11.2 # load cuda modules if needed
source activate $ENV_NAME
echo env done

WORK_DIR=/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding
cp /home/chuanrui/scratch/database/structure_datasets/PDB/filtered_147724/zipped_data/rep_pdb.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR

mkdir pdb
tar -xzf rep_pdb.tar.gz -C pdb

find pdb/ | wc -l
echo finish untaring

cd $WORK_DIR
TORCH_DISTRIBUTED_DEBUG=DETAIL srun python inverse_predict.py --input_dir $SLURM_TMPDIR/pdb/rep_pdb \
    --output_dir /home/chuanrui/scratch/database/structure_datasets/PDB/filtered_147724/sampling \
    --protein_name "5" \

