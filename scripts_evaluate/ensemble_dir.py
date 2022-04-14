import pyrosetta
import argparse
import os
import numpy as np
import time
import requests
from tqdm import tqdm
import shutil

import logging
logging.basicConfig(level=logging.INFO)
    
def get_plddt(path):
    """1dlf_epoch48_cat_refine_rec3_plddt95.750_lddt93.798_unrelaxed.pdb --> 95.75"""
    # 1dlf_epoch48_cat_refine_rec3_plddt95.750_lddt93.798_unrelaxed
    basename = os.path.splitext(os.path.basename(path))[0]
    index = basename.find('plddt') + len("plddt")
    return float(basename[index: index + 6])
    
def main(args):
    valid_pred_dir = 1
    path_list = [args.pred_dir]
    for path in [args.pred_dir2, args.pred_dir3, args.pred_dir4, args.pred_dir5]:
        if path is not None:
            valid_pred_dir += 1
            path_list.append(path)
    assert valid_pred_dir > 1
    logging.info(f"got {len(path_list)} predictions to ensemble.")

    jobs = []
    for i in range(0, valid_pred_dir):
        jobs.append([x for x in os.listdir(path_list[i]) if x.endswith('.pdb')])

    ensemble_list = []
    for i in range(len(jobs)):
        ensemble_i = {}
        for fname in tqdm(jobs[i]):
            pdbid = fname[:4]
            ensemble_i[pdbid] = os.path.join(path_list[i], fname)
        ensemble_list.append(ensemble_i)

    ensembled_path_list = []
    ensemble_1 = ensemble_list[0]
    for key in ensemble_1: # pdb key
        cur_path = ensemble_1[key]
        best = get_plddt(cur_path)
        for ensemble_i in ensemble_list:
            if get_plddt(ensemble_i[key]) > best:
                best = get_plddt(ensemble_i[key])
                cur_path = ensemble_i[key]
        ensembled_path_list.append(cur_path)

    logging.info("copying ensembled pdbs to target dirs.")
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir, exist_ok=True)
    for fpath in ensembled_path_list:
        tgt_path = os.path.join(args.target_dir, os.path.basename(fpath))
        shutil.copy2(fpath, tgt_path)


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
        "--pred_dir", type=str,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--target_dir", type=str,
        help="Path to the ensembled_pdb"
    )
    parser.add_argument(
        "--pred_dir2", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pred_dir3", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pred_dir4", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pred_dir5", type=str, default=None,
        help="Path to the pred_pdb"
    )            
    args = parser.parse_args()

    main(args)
