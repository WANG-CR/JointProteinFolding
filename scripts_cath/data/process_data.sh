#! /bin/bash
# 1. Activate environement
ENV_NAME=cath_gen
module load cuda/11.4
source activate $ENV_NAME
echo env done

""" 
    1. filter out the data that does not have positive labels
    2. filter out the data with huge antigen chain
"""
python scripts_cath/data/process_data.py \
    /home/shichenc/scratch/structure_datasets/cath/raw/dompdb \
    /home/shichenc/scratch/structure_datasets/cath/processed/top_split \
    /home/shichenc/scratch/structure_datasets/cath/raw/ss_info_topo_31883.pkl \

