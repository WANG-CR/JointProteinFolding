#! /bin/bash
# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done


#############
##treat h3###
# python save_as_jsonl.py \
#     --pdb_path /home/tjhec/scratch/database/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh3_test/ \
#     --is_antibody true \
#     --output_dir /home/tjhec/scratch/antibody/RefineGNN/data/sabdab_0618/20220618_Fv_4_All_cdrh3/test.jsonl \

python save_as_jsonl.py \
    --pdb_path /home/tjhec/scratch/database/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh3_train/ \
    --is_antibody true \
    --output_dir /home/tjhec/scratch/antibody/RefineGNN/data/sabdab_0618/20220618_Fv_4_All_cdrh3/train.jsonl \

python save_as_jsonl.py \
    --pdb_path /home/tjhec/scratch/database/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh3_valid/ \
    --is_antibody true \
    --output_dir /home/tjhec/scratch/antibody/RefineGNN/data/sabdab_0618/20220618_Fv_4_All_cdrh3/valid.jsonl \
# # /home/tjhec/scratch/database/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh3_test/


#############
##treat h1###
python save_as_jsonl.py \
    --pdb_path /home/tjhec/scratch/database/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh1_test/ \
    --is_antibody true \
    --output_dir /home/tjhec/scratch/antibody/RefineGNN/data/sabdab_0618/20220618_Fv_4_All_cdrh1/test.jsonl \

python save_as_jsonl.py \
    --pdb_path /home/tjhec/scratch/database/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh1_train/ \
    --is_antibody true \
    --output_dir /home/tjhec/scratch/antibody/RefineGNN/data/sabdab_0618/20220618_Fv_4_All_cdrh1/train.jsonl \

python save_as_jsonl.py \
    --pdb_path /home/tjhec/scratch/database/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh1_valid/ \
    --is_antibody true \
    --output_dir /home/tjhec/scratch/antibody/RefineGNN/data/sabdab_0618/20220618_Fv_4_All_cdrh1/valid.jsonl \

#############
##treat h2###
python save_as_jsonl.py \
    --pdb_path /home/tjhec/scratch/database/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh2_test/ \
    --is_antibody true \
    --output_dir /home/tjhec/scratch/antibody/RefineGNN/data/sabdab_0618/20220618_Fv_4_All_cdrh2/test.jsonl \

python save_as_jsonl.py \
    --pdb_path /home/tjhec/scratch/database/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh2_train/ \
    --is_antibody true \
    --output_dir /home/tjhec/scratch/antibody/RefineGNN/data/sabdab_0618/20220618_Fv_4_All_cdrh2/train.jsonl \

python save_as_jsonl.py \
    --pdb_path /home/tjhec/scratch/database/antibody/sabdab/pdb/20220618_Fv_4_All_cdrh2_valid/ \
    --is_antibody true \
    --output_dir /home/tjhec/scratch/antibody/RefineGNN/data/sabdab_0618/20220618_Fv_4_All_cdrh2/valid.jsonl \

