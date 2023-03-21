#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=186G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --out=/home/chuanrui/scratch/database/structure_datasets/PDB/filtered_147724/slurm/seperate_and_compress_completefilter.out
#SBATCH --error=/home/chuanrui/scratch/database/structure_datasets/PDB/filtered_147724/slurm/seperate_and_compress_completefilter.err
#SBATCH --qos=main


# 1. Activate environement
ENV_NAME=pf2
source activate $ENV_NAME
echo env done

mkdir $SLURM_TMPDIR/data


# here, pdb_dezip is the database containing .pdb files
python process_pdb.py \
    /home/chuanrui/scratch/database/structure_datasets/PDB/filtered_147724/pdb_dezip \
    $SLURM_TMPDIR/data/seperate \

# generate seperated chains and make a tar
cd $SLURM_TMPDIR/data/seperate_80identity_seperateChains
tar -I pigz -cf seperate_pdb_complete_filter.tar.gz pdb

# move tar from temporary directory to permanent directory
mkdir /home/chuanrui/scratch/data/seperate_complete_filter
cp $SLURM_TMPDIR/data/seperate_80identity_seperateChains/seperate_pdb_complete_filter.tar.gz /home/chuanrui/scratch/data/seperate_complete_filter
cp -r $SLURM_TMPDIR/data/seperate_80identity_seperateChains/fasta /home/chuanrui/scratch/data/seperate_complete_filter