import argparse
import logging
from tracemalloc import start
from turtle import shape
logging.basicConfig(level=logging.INFO)

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch


from openfold.np import residue_constants, protein
from openfold.data import data_pipeline
from tqdm import tqdm
import debugger

def gather_job(pdb_dir):
    pdb_paths = []
    for f_path in os.listdir(pdb_dir):
        if f_path.endswith('.pdb'):
            pdb_path = os.path.join(pdb_dir, f_path)
            pdb_paths.append(pdb_path)
    
    return pdb_paths

def main(args):
    # Prepare data

    data_processor = data_pipeline.DataPipeline(is_antibody=args.is_antibody)

    output_dir = args.output_dir

    jobs = gather_job(args.pdb_path)
    logging.info(f'got {len(jobs)} jobs...')
    # Get input 
    JSON_file = []
    for job in tqdm(jobs):
        f_path = os.path.basename(job)
        name = f_path[:4].lower()

        feature_dict = data_processor.process_pdb(
            pdb_path=job,
        )
        
        dict_pdb = {}
        dict_pdb["pdb"] = name
        dict_pdb["seq"]=feature_dict["sequence"].item().decode('UTF-8')
        cdr_index = feature_dict["loop_index"].astype(str).tolist()
        dict_pdb["cdr"]=[''.join(cdr_index)][0]
        # print(dict_pdb)
        coords = feature_dict["all_atom_positions"]
        coords = np.around(coords, 3)
        # print(coords)
        ca_pos = residue_constants.atom_order["CA"]
        n_pos = residue_constants.atom_order["N"]
        c_pos = residue_constants.atom_order["C"]
        o_pos = residue_constants.atom_order["O"]
        ca_coords = coords[:, ca_pos, :]
        n_coords = coords[:, n_pos, :]
        c_coords = coords[:, c_pos, :]
        o_coords = coords[:, o_pos, :]
        dict_pdb["coords"]={
            "N":np.around(n_coords, 3).tolist(), 
            "CA":np.around(ca_coords, 3).tolist(), 
            "C":np.around(c_coords, 3).tolist(), 
            "O":np.around(o_coords, 3).tolist(), 
        }
        # print(coords[..., n_pos, :])
        JSON_file.append(dict_pdb)
    # print(JSON_file)
    import json
    with open(output_dir, 'w') as outfile:
        for entry in JSON_file:
            json.dump(entry, outfile)
            outfile.write('\n')
    
def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb_path", type=str,
    )
    parser.add_argument(
        "--is_antibody", type=bool, default=None,
        help="training on antibody or not"
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )
    args = parser.parse_args()

    main(args)