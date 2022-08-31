import argparse
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
import shutil
import random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

from openfold.utils import data_utils


def parse_sabdab_summary(file_name):
    """
    SAbDab produces a rather unique summary file.
    This function reads that file into a dict with the key being the
    4-letter PDB code.
    
    dict[pdb] = {col1 : value, col2: value, ...}
    """
    sabdab_dict = {}

    with open(file_name, "r") as f:
        # first line is the header, or all the keys in our sub-dict
        header = f.readline().strip().split("\t")
        # next lines are data
        for line in f.readlines():
            split_line = line.strip().split("\t")
            td = {}  # temporary dict of key value pairs for one pdb
            for k, v in zip(header[1:], split_line[1:]):
                # pdb id is first, so we skip that for now
                td[k] = v
            # add temporary dict to sabdab dict at the pdb id
            sabdab_dict[split_line[0]] = td

    return sabdab_dict


def get_split_data(pdb_dir, summary_path, train_ratio=0.9):
    """
    split pdbs in pdb_dir according to released date provided in summary path.

    Args:
        pdb_dir (str)
        summary_path (str)
    """
    # sanity check
    summary_df = pd.read_csv(summary_path, sep='\t')
    pdb_ids = [x[:4] for x in os.listdir(pdb_dir) if x.endswith(".pdb")]
    summary_df = summary_df.loc[summary_df['pdb'].isin(pdb_ids)]
    assert len(sorted(summary_df['pdb'].unique())) == len(pdb_ids),\
        "some pdbs are not included in the summary file"
    
    sabdab_dict = parse_sabdab_summary(summary_path)
    pdb_date_dict = {}
    for pdb_id in pdb_ids:
        pdb_date_dict[pdb_id] = datetime.strptime(sabdab_dict[pdb_id]['date'], '%m/%d/%y')

    pdb_date_dict = sorted(pdb_date_dict.items(), key=lambda x: x[1])
    sorted_pdbs, sorted_dates = zip(*pdb_date_dict)
    len_data = len(sorted_pdbs)
    train_pdbs = sorted_pdbs[:int(len_data * train_ratio)]
    valid_pdbs = sorted_pdbs[int(len_data * train_ratio):]
    logging.info(
        f"{len_data} pdbs are split into a train set with {len(train_pdbs)} pdbs "
        f"and a valid set with {len(valid_pdbs)} pdbs"
    )
    logging.info(
        f"train set contains pdbs released between {sorted_dates[0]} "
        f"and {sorted_dates[int(len_data * train_ratio) - 1]}"
    )
    logging.info(
        f"valid set contains pdbs released between {sorted_dates[int(len_data * train_ratio)]} "
        f"and {sorted_dates[-1]}"
    )
    return train_pdbs, valid_pdbs


def copy_pdb_to_new_dir(
    pdb_id,
    old_pdb_dir,
    new_pdb_dir,
    old_fasta_dir=None,
    new_fasta_dir=None,
):
    """copy pdb and fasta into a new directory"""
    old_pdb_path = os.path.join(old_pdb_dir, pdb_id + '.pdb')
    new_pdb_path = os.path.join(new_pdb_dir, pdb_id + '.pdb')
    shutil.copy2(old_pdb_path, new_pdb_path)
    
    if old_fasta_dir is not None and new_fasta_dir is not None:
        old_fasta_path = os.path.join(old_fasta_dir, pdb_id + '.fasta')
        new_fasta_path = os.path.join(new_fasta_dir, pdb_id + '.fasta')
        shutil.copy2(old_fasta_path, new_fasta_path)


def merge_fasta(fasta_dir, output_path, mode):
    """merge fastas in a directory into a single fasta file"""
    fastas = data_utils.fastas2fasta(data_dir=fasta_dir, mode=mode)
            
    with open(output_path, "w") as fp:
        fp.write('\n'.join(fastas))

def split_by_cluster(pdb_dir, cluster_res, train_ratio=0.8, valid_ratio=0.1, seed=2022):
    assert train_ratio + valid_ratio < 1
    test_ratio = 1 - train_ratio - valid_ratio
    random.seed(seed)

    pdb_ids = [
        os.path.splitext(x)[0] for x in os.listdir(pdb_dir) if x.endswith(".pdb")
    ]

    cluster_dict = defaultdict(list)
    with open(cluster_res, 'r') as f:
        for line in f:
            k, v = line.strip().split('\t')
            k = k.rsplit('_', 1)[0]
            v = v.rsplit('_', 1)[0]
            if v not in pdb_ids:
                logging.warning(f"pair {k}:{v} not found in database")
                continue
            cluster_dict[k].append(v)
    
    cluster_rep = sorted(list(cluster_dict.keys()))
    random.shuffle(cluster_rep)
    len_cluster = len(cluster_rep)
    
    train_rep = cluster_rep[:int(len_cluster * train_ratio)]
    valid_rep  = cluster_rep[int(len_cluster * train_ratio):int(len_cluster * (train_ratio + valid_ratio))]
    test_rep = cluster_rep[int(len_cluster * (train_ratio + valid_ratio)):]
    logging.info(
        f"get {len_cluster} cluster representatives\n"
        f"get {len(train_rep)} train cluster representatives\n"
        f"get {len(valid_rep)} valid cluster representatives\n"
        f"get {len(test_rep)} test cluster representatives\n"
    )
    train_pdbs, valid_pdbs, test_pdbs = [], [], []
    for pdb_id in train_rep:
        train_pdbs.extend(cluster_dict[pdb_id])
    for pdb_id in valid_rep:
        valid_pdbs.extend(cluster_dict[pdb_id])
    for pdb_id in test_rep:
        test_pdbs.extend(cluster_dict[pdb_id])
    logging.info(
        f"get {len(train_pdbs)} train pdbs\n"
        f"get {len(valid_pdbs)} valid pdbs\n"
        f"get {len(test_pdbs)} test pdbs\n"
    )

    return train_pdbs, valid_pdbs, test_pdbs
        


def main(args):

    # 1. split train/valid data, and prepare directory
    logging.info(f"spliting the sabdab database with version: {args.summary_prefix}")
    summary_path = os.path.join(args.info_dir, args.summary_prefix + "_sabdab_summary.tsv")

    old_pdb_dir = os.path.join(args.pdb_dir, args.summary_prefix + "_sabdab")
    all_cdr_names = ['cdrs', 'cdrh1', 'cdrh2', 'cdrh3', 'cdrl1', 'cdrl2', 'cdrl3']
    cdr_name = all_cdr_names[args.cdr_idx]
    train_pdb_dir = os.path.join(args.pdb_dir, args.summary_prefix + f"_{cdr_name}_{args.train_ratio}_{args.valid_ratio}_train")
    valid_pdb_dir = os.path.join(args.pdb_dir, args.summary_prefix + f"_{cdr_name}_{args.train_ratio}_{args.valid_ratio}_valid")
    test_pdb_dir = os.path.join(args.pdb_dir, args.summary_prefix + f"_{cdr_name}_{args.train_ratio}_{args.valid_ratio}_test")

    for dir_ in [train_pdb_dir, valid_pdb_dir, test_pdb_dir]:
        os.makedirs(dir_, exist_ok=True)
    
    train_pdbs, valid_pdbs, test_pdbs = split_by_cluster(
        old_pdb_dir,
        args.cluster_res,
        args.train_ratio,
        args.valid_ratio,
        args.seed,
    )

    # 2. copy data
    logging.info("copying pdbs...")
    for pdb in train_pdbs:
        copy_pdb_to_new_dir(
            pdb, old_pdb_dir, train_pdb_dir
        )
    for pdb in valid_pdbs:
        copy_pdb_to_new_dir(
            pdb, old_pdb_dir, valid_pdb_dir
        )
    for pdb in test_pdbs:
        copy_pdb_to_new_dir(
            pdb, old_pdb_dir, test_pdb_dir
        )


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
        "pdb_dir", type=str,
        help="Path to a directory containing truncated pdbs"
    )
    parser.add_argument(
        "fasta_dir", type=str,
        help="Path to a directory containing truncated fastas"
    )
    parser.add_argument(
        "info_dir", type=str,
        help="Path to a directory containing information about the database, e.g., summary file"
    )
    parser.add_argument(
        "summary_prefix", type=str,
        help="prefix of the sabdab summary file, which uniquely determines the version of the database"
    )
    parser.add_argument(
        "--cdr_idx", type=int,
        help="which cdr to process"
    )
    parser.add_argument(
        "--cluster_res", type=str,
        help="path of the clustering results"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.9,
        help="the proportion of the train data"
    )
    parser.add_argument(
        "--valid_ratio", type=float, default=0.05,
        help="the proportion of the valid data"
    )
    parser.add_argument(
        "--seed", type=int, default=2022,
        help="random shuffle seed"
    )

    args = parser.parse_args()

    main(args)