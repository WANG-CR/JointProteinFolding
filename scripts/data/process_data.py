#coding: utf-8
import logging
logging.basicConfig(level=logging.INFO)

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

""" 
python scripts_cath/data/process_data.py \
    /home/shichenc/scratch/structure_datasets/cath/raw/dompdb \
    /home/shichenc/scratch/structure_datasets/cath/processed/top_split \
    /home/shichenc/scratch/structure_datasets/cath/raw/ss_annotation_31885.pkl \
        
# before debug
INFO:root:get 31885 files.
INFO:root:get 1470 unique topologies
INFO:root:859 data is excluded.
large: 116 | not the same: 599
parse error: 144 | remaining: 31026

# after debug
WARNING:root:get 31885 second structure data
WARNING:root:remove bad pdbs: 1alo006, found chain:
WARNING:root:remove bad pdbs: 1baa001, found chain:
WARNING:root:remove bad pdbs: 1bdp001, found chain:
WARNING:root:remove bad pdbs: 1bdp002, found chain:
WARNING:root:remove bad pdbs: 1gep001, found chain:
WARNING:root:remove bad pdbs: 1sil000, found chain:
WARNING:root:remove bad pdbs: 1tbs002, found chain:
WARNING:root:remove bad pdbs: 2mt2000, found chain:
INFO:root:get 31877 files.
INFO:root:get 1469 unique topologies
INFO:root:859 data is excluded.
large: 116 | not the same: 599
parse error: 144 | remaining: 31018
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
        chain_id = fname[4]
        protein_object = protein.from_pdb_string(pdb_str, chain_id)
        assert len(protein_object.atom_positions) > 0
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
        tag_ = ss_['tag']
        chain_ = ss_['chain']
        assert len(set(chain_)) == 1
        if chain_[0] != tag_[4]:
            logging.warning(f"remove bad pdbs: {tag_}, found chain: {chain_}")
            continue
        tag2seq[ss_['tag']] = ss_['sequence']
        tag2top[ss_['tag']] = ss_['topology']

    filenames = [x for x in os.listdir(args.input_dir) if x in tag2top]
    assert len(filenames) == len(tag2top)
    assert len(filenames) == len(set(filenames))

    all_tops = list(set(tag2top.values()))

    top2split = {}
    for top in all_tops:
        rand_ = np.random.rand()
        if rand_ <= args.valid_ratio:
            top2split[top] = 'valid'
        elif rand_ <= (args.valid_ratio + args.test_ratio):
            top2split[top] = 'test'
        else:
            top2split[top] = 'train'

    logging.info(f"get {len(filenames)} files.")
    logging.info(f"get {len(top2split)} unique topologies")
    args.output_dir = args.output_dir + \
                      f"_{args.max_len}_{args.seed}_{args.valid_ratio}_{args.test_ratio}"
    train_output_dir = args.output_dir + "_train"
    valid_output_dir = args.output_dir + "_valid"
    test_output_dir = args.output_dir + "_test"
    output_dirs = [train_output_dir, valid_output_dir, test_output_dir]
    split2output_dir = {
        'train': output_dirs[0],
        'valid': output_dirs[1],
        'test': output_dirs[2],
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
        "--valid_ratio", type=float, default=0.01,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.04,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--seed", type=int, default=2023,
        help="Path to model parameters."
    )
    args = parser.parse_args()
    main(args)
