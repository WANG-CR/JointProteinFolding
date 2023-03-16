#! /bin/bash
#SBATCH --account=def-bengioy
#SBATCH --cpus-per-task=40
#SBATCH --mem=186G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --qos=main


# ls pdb_dezip/*.pdb | parallel -j 8 'tar -rf pdb_20200501.tar {}'
ls pdb_dezip/*.pdb | parallel -j0 'tar -rf pdb_20200501.tar {}'
# gzip pdb_20200501.tar



## compress remaining pdb files
# ls pdb_dezip/*.pdb > all_pdb_files.txt
# tar -tf pdb_20200501.tar > tar_contents.txt
# grep -vxFf tar_contents.txt all_pdb_files.txt > remaining_pdb_files.txt
# cat remaining_pdb_files.txt | parallel -j0 'tar -rf pdb_20200501.tar {}'
# gzip -f pdb_20200501.tar