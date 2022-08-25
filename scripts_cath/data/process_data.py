#coding: utf-8
import os
import argparse
import numpy as np
import shutil
import multiprocessing
from functools import partial
import pickle

from openfold.utils.seed import seed_everything
from openfold.np import protein, residue_constants
import debugger

import logging
logging.basicConfig(level=logging.INFO)

""" 
python scripts_cath/data/process_data.py \
    /home/shichenc/scratch/structure_datasets/cath/raw/dompdb \
    /home/shichenc/scratch/structure_datasets/cath/processed/top_split \
    /home/shichenc/scratch/structure_datasets/cath/raw/ss_info_topo_31883.pkl \
"""

def is_same_seq(prot: protein.Protein, seq, fname):
    restypes = residue_constants.restypes + ['X']
    decoded_seq = ""
    aatype = prot.aatype
    for i in range(aatype.shape[0]):
        decoded_seq += restypes[aatype[i]]
    
    if decoded_seq == seq:
        return True
    else:
        # logging.warning(
        #     f"{fname}: decoded sequence is different from the original sequence\n"
        #     f"{decoded_seq} vs.\n"
        #     f"{seq}"
        # )
        return False

def is_large_prot(prot: protein.Protein, max_len, fname):
    if prot.aatype.shape[0] <= max_len:
        return False
    else:
        return True

def do(fname, src_path, tgt_path, seq, max_len):
    with open(src_path, 'r') as f:
        pdb_str = f.read()
    try:
        protein_object = protein.from_pdb_string(pdb_str)
    except ValueError as e:
        # logging.warning(
        #     f"fail to parse {fname}\n"
        #     f"{e}"
        # )
        return 3
    if is_large_prot(protein_object, max_len, fname):
        return 1
    if not is_same_seq(protein_object, seq, fname):
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

    filenames = [x for x in os.listdir(args.input_dir) if x in tag2top]
    assert len(filenames) == len(tag2top)
    assert len(filenames) == len(set(filenames))

    all_tops = list(set(tag2top.values()))

    top2split = {}
    for top in all_tops:
        if np.random.rand() <= args.test_ratio:
            top2split[top] = 'test'
        else:
            top2split[top] = 'train'

    logging.info(f"get {len(filenames)} files.")
    logging.info(f"get {len(top2split)} unique topologies")
    args.output_dir = args.output_dir + \
                      f"_{args.max_len}_{args.seed}_{args.test_ratio}"
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
    
    job_args = zip(filenames, src_paths, tgt_paths, seqs)
    cnt_list = [0,0,0,0]
    with multiprocessing.Pool(args.num_workers) as p:
        func = partial(do, max_len=args.max_len)
        for ret_code in p.starmap(func, job_args):
            assert ret_code in [0, 1, 2, 3]
            cnt_list[ret_code] += 1
    logging.info(
        f"{sum(cnt_list) - cnt_list[0]} data is excluded.\n"
        f"large: {cnt_list[1]} | not the same: {cnt_list[2]}\n"
        f"parse error: {cnt_list[3]} | remaining: {cnt_list[0]}"
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
        "--test_ratio", type=float, default=0.05,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--seed", type=int, default=2023,
        help="Path to model parameters."
    )
    args = parser.parse_args()
    main(args)
