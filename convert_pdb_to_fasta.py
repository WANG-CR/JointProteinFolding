import argparse
import logging
import os
# import sys
# import time
# import pickle
import shutil
import pickle
import re

from openfold.np import protein, residue_constants
from openfold.data import feature_pipeline, data_pipeline
logging.basicConfig(level=logging.INFO)
import numpy as np
from tqdm import tqdm
    
def gather_job(pdb_dir):
    pdb_paths = []
    for f_path in os.listdir(pdb_dir):
        if f_path.endswith('.pdb'):
            pdb_path = os.path.join(pdb_dir, f_path)
            pdb_paths.append(pdb_path)
    
    return pdb_paths

def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]] 
        for i in range(len(aatype))
    ])

def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


def main(args):
    jobs = gather_job(args.input_dir)
    
    directory = args.output_dir
    if not os.path.exists(directory):
        os.makedirs(directory)

    logging.info(f'got {len(jobs)} jobs...')

    for job in tqdm(jobs):
        f_path = os.path.basename(job)
        name = f_path.split(".")[0]
        print(f"name: {name}")
        with open(job, 'r') as f:
            pdb_str = f.read()
        protein_object = protein.from_pdb_string(pdb_str)
        seq = _aatype_to_str_sequence(protein_object.aatype)

        with open(os.path.join(directory, name+".fasta"), 'w') as f:
            f.write(">"+name+os.linesep)
            f.write(seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )

    args = parser.parse_args()
    main(args)