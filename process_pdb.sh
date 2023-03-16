#! /bin/bash
# 1. Activate environement
ENV_NAME=pf2
source activate $ENV_NAME
echo env done

""" 
    1. filter out the data that does not have positive labels
    2. filter out the data with huge antigen chain
"""
python process_pdb.py \
    /home/chuanrui/scratch/database/structure_datasets/PDB/filtered_147724/pdb_dezip \
    /home/chuanrui/scratch/database/structure_datasets/PDB/filter2 \