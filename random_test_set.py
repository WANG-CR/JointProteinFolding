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

def gather_job(pdb_dir):
    pdb_paths = []
    for f_path in os.listdir(pdb_dir):
        if f_path.endswith('.pdb'):
            pdb_path = os.path.join(pdb_dir, f_path)
            pdb_paths.append(pdb_path)
    
    return pdb_paths

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

    jobs = gather_job(args.pdb_path)
    logging.info(f'got {len(jobs)} jobs...')
    # Get input 


    top_k = 400
    prob = torch.rand(len(jobs))
    _, indexes = torch.topk(prob, top_k)
    indexes = indexes.tolist()
    topk_jobs = [jobs[i] for i in indexes]
    for job in topk_jobs:
        f_path = os.path.basename(job)
        name = f_path[:args.name_length]
        
        print(f"#################")
        print(f">>> treating: {name}")
        
        dst = os.path.join(
            output_dir,
            f"{name}.pdb"
        )
        shutil.copyfile(job, dst)

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

    #usage
    # python random_test_set.py /nfs/work04/chuanrui/data/sampled_protein/sampling --output_dir /nfs/work04/chuanrui/data/subset_400