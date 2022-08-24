#coding: utf-8

import imp
import os
import logging
import argparse
import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import shutil
import multiprocessing
from functools import partial
import pickle

logging.basicConfig(level=logging.INFO)
from openfold.utils import data_utils
from openfold.utils.seed import seed_everything
from openfold.np import protein, residue_constants
import debugger

""" 
python scripts_cath/data/process_data.py \
    /home/shichenc/scratch/structure_datasets/cath/raw/dompdb \
    /home/shichenc/scratch/structure_datasets/cath/processed/top_split \
    /home/shichenc/scratch/structure_datasets/cath/raw/ss_info_topo_31883.pkl \
"""
def is_same_seq(prot: protein.Protein, seq):
    restypes = residue_constants.restypes + ['X']
    decoded_seq = ""
    aatype = prot.aatype
    for i in range(aatype.shape[0]):
        decoded_seq += restypes[aatype[i]]
    
    if decoded_seq == seq:
        return True
    else:
        logging.warning(
            f"decoded sequence is different from the original sequence\n"
            f"found {decoded_seq} vs. {seq}"
        )
        return False

def is_large_prot(prot: protein.Protein, max_len):
    if prot.aatype.shape[0] <= max_len:
        return False
    else:
        return True

def do(args, src_path, tgt_path, seq):
    with open(src_path, 'r') as f:
        pdb_str = f.read()    
    protein_object = protein.from_pdb_string(pdb_str)
    if is_large_prot(protein_object, args.max_len):
        return 1
    if not is_same_seq(protein_object, seq):
        return 2

    shutil.copy2(src_path, tgt_path)
    return 0

def main(args):
    seed_everything(args.seed)

    tag2seq = {}
    tag2top = {}
    with open(args.ss_file, 'rb') as fin:
        second_structure_data = pickle.load(fin)
    logging.warning(f"get {len(second_structure_data)} second structure data")
    for ss_ in second_structure_data:
        tag2seq[ss_['tag']] = ss_['sequence']
        tag2top[ss_['tag']] = ss_['topology']

    filenames = [x for x in os.listdir(args.input_dir) if len(x) == 7]
    assert len(filenames) == len(tag2top)
    assert len(filenames) == len(set(filenames))

    all_tops = list(set(tag2top.values))

    top2split = {}
    for top in all_tops:
        if np.random.rand() <= 0.05:
            top2split[top] = 'test'
        else:
            top2split[top] = 'train'

    logging.info(f"get {len(filenames)} files.")
    logging.info(f"get {len(top2split)} unique topologies")
    args.output_dir = args.output_dir + f"_{args.max_len}"
    train_output_dir = args.output_dir + "_train"
    test_output_dir = args.output_dir + "_test"
    output_dirs = [train_output_dir, test_output_dir]
    split2output_dir = {
        'train': output_dirs[0],
        'test': output_dirs[1],
    }

    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    src_paths = []
    tgt_paths = []
    seqs = []
    for fname in filenames:
        src_path = os.path.join(args.input_dir, fname)
        tag = fname
        top = tag2top[tag]
        split = top2split[top]
        tgt_path = os.path.join(
            split2output_dir[split],
            fname + '.pdb',
        )
        src_paths.append(src_path)
        tgt_paths.append(tgt_path)
        seqs.append(tag2seq[tag])
    
    job_args = zip(src_paths, tgt_paths, seqs)
    cnt_large = 0
    cnt_error = 0
    cnt = 0
    with multiprocessing.Pool(args.num_workers) as p:
        func = partial(do, args=args)
        for ret_code in p.starmap(func, job_args):
            if ret_code == 1:
                cnt_large += 1
            elif ret_code == 2:
                cnt_error
            else:
                cnt += 1
    logging.info(
        f"{cnt_large+cnt_error} data is excluded.\n"
        f"large: {cnt_large} | error: {cnt_error}\n"
        f"remaining: {cnt}"
    )

def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeoProt')
    parser.add_argument(
        "input_dir", type=str,
        help="input prefix for glob."
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Path to model parameters."
    )
    parser.add_argument(
        "ss_file", type=str,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--num_workers", type=int, default=36,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--max_len", type=int, default=512,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--seed", type=int, default=2022,
        help="Path to model parameters."
    )
    args = parser.parse_args()
    main(args)
