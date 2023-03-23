import argparse
import logging
from tracemalloc import start
from turtle import shape
logging.basicConfig(level=logging.INFO)

import os
# import sys
# import time
# import pickle
import torch
import torch.nn.functional as F
# from openfold.utils.seed import seed_everything
from pytorch_lightning.utilities.seed import seed_everything
import shutil
import debugger

# def gather_job(pdb_dir):
#     pdb_paths = []
#     for f_path in os.listdir(pdb_dir):
#         if f_path.endswith('.pdb'):
#             pdb_path = os.path.join(pdb_dir, f_path)
#             pdb_paths.append(pdb_path)
    
#     return pdb_paths

def gather_job(pdb_dir):
    pdb_names = []
    for f_path in os.listdir(pdb_dir):
        if f_path.endswith('.pdb'):
            pdb_name = f_path[:5]
            if pdb_name not in pdb_names:
                pdb_names.append(pdb_name)
    
    return pdb_names


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')

def main(args):
    if args.seed is not None:
        seed_everything(args.seed)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_test = os.path.join(output_dir,"test")
    output_train = os.path.join(output_dir,"train")
    if not os.path.exists(output_test):
        os.makedirs(output_test)
    if not os.path.exists(output_train):
        os.makedirs(output_train)

    jobs = gather_job(args.pdb_path)
    logging.info(f'got {len(jobs)} jobs...')
    # Get input 


    top_k = 40
    prob = torch.rand(len(jobs))
    _, indexes = torch.topk(prob, top_k)
    indexes = indexes.tolist()
    topk_test = [jobs[i] for i in indexes]
    topk_train = [x for x in jobs if x not in topk_test] 
    post_string = ["_sample0", "_sample1", "_sample2"]
    for job in topk_test:
        for i in range(3):
            name = job+post_string[i]
            src = os.path.join(args.pdb_path,f"{name}.pdb")
            tgt = os.path.join(output_test,f"{name}.pdb")
            
            # print(f"#################")
            # print(f">>> treating: {name}")
        
            shutil.copyfile(src, tgt)
    
    for job in topk_train:
        for i in range(3):
            name = job+post_string[i]
            src = os.path.join(args.pdb_path,f"{name}.pdb")
            tgt = os.path.join(output_train,f"{name}.pdb")
            
            # print(f"#################")
            # print(f">>> treating: {name}")
        
            shutil.copyfile(src, tgt)
        # # 2nd option
        # shutil.copy(src, dst) 

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pdb_path", type=str,
    )    
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--name_length", type=int, default=13,
        help="how many characters are used to name the protein"
    )
    args = parser.parse_args()
    main(args)

    # usage
    # python /home/Xcwang/scratch/beluga/JointProteinFolding/random_test_set_avoidleakage.py /nfs/work04/chuanrui/data/sampled_protein/sampling --output_dir /nfs/work04/chuanrui/data/subset_1200_noleakage
