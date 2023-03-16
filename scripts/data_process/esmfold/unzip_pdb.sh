#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=186G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --qos=main


export -f gunzip
parallel -j0 gunzip -k {} -c ">" /home/chuanrui/scratch/database/structure_datasets/PDB/filtered_147724/pdb_dezip/{/.} ::: pdb/*.gz


# for file in pdb/*.gz; do gunzip -k "$file" -c > "/home/chuanrui/scratch/database/structure_datasets/PDB/filtered_147724/pdb_dezip/$(basename "$file" .gz)"; done