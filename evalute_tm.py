import numpy as np
from tmtools import tm_align
from openfold.np import protein, residue_constants
import os
import argparse
import logging
from tqdm import tqdm
from statistics import mean

def tm_score_from_pdbs(fp1, fp2):
    coords1, seq1 = coord_seq_from_file(fp1)
    coords2, seq2 = coord_seq_from_file(fp2)
    res = tm_align(coords1, coords2, seq1, seq2)
    return res.tm_norm_chain1

def coord_seq_from_file(path):
    with open(path, 'r') as f:
        pdb_str = f.read()
    protein_object = protein.from_pdb_string(pdb_str)
    ca_coords = protein_object.atom_positions[:, 1, :]
    seq = _aatype_to_str_sequence(protein_object.aatype)
    return ca_coords, seq

def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]] 
        for i in range(len(aatype))
    ])

def gather_job(gt_dir, predict_dir):
    pdb_paths = []
    for f_path in os.listdir(gt_dir):
        if f_path.endswith('.pdb'):
            pdb_path_to_verify = os.path.join(predict_dir, f_path)
            if os.path.exists(pdb_path_to_verify):
                pdb_paths.append(f_path)
    return pdb_paths

def TMalign_dir(path1, path2):
    jobs = gather_job(path1, path2)    
    logging.info(f'got {len(jobs)} jobs...')

    tm_list = []
    tm_dict = {}
    for job in tqdm(jobs):
        fp1 = os.path.join(path1, job)
        fp2 = os.path.join(path2, job)
        name = job.split(".")[0]
        tm = tm_score_from_pdbs(fp1, fp2)
        tm_list.append(tm)
        tm_dict[name] = tm
    return tm_list, tm_dict, mean(tm_list), len(jobs)

def main(args):
    tm_list, tm_dict, mean, num_jobs = TMalign_dir(args.gt_dir, args.predict_dir)
    with open(args.log_dir, 'w') as f:
        f.write(str(num_jobs) + " jobs: ")
        f.write(str(mean))
    # print(mean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_dir", type=str, default=None,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--predict_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )
    parser.add_argument(
        "--log_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )
    args = parser.parse_args()
    main(args)