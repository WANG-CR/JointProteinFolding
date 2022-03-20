import argparse
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
import shutil
import numpy as np
import pandas as pd
from datetime import datetime

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
    old_fasta_dir,
    new_pdb_dir,
    new_fasta_dir,
):
    """copy pdb and fasta into a new directory"""
    old_pdb_path = os.path.join(old_pdb_dir, pdb_id + '.pdb')
    new_pdb_path = os.path.join(new_pdb_dir, pdb_id + '.pdb')
    old_fasta_path = os.path.join(old_fasta_dir, pdb_id + '.fasta')
    new_fasta_path = os.path.join(new_fasta_dir, pdb_id + '.fasta')
    shutil.copy2(old_pdb_path, new_pdb_path)
    shutil.copy2(old_fasta_path, new_fasta_path)


def merge_fasta(fasta_dir, output_path, mode):
    """merge fastas in a directory into a single fasta file"""
    fastas = data_utils.fastas2fasta(data_dir=fasta_dir, mode=mode)
            
    with open(output_path, "w") as fp:
        fp.write('\n'.join(fastas))
        

def main(args):

    # 1. split train/valid data, and prepare directory
    logging.info(f"spliting the sabdab database with version: {args.summary_prefix}")
    summary_path = os.path.join(args.info_dir, args.summary_prefix + "_sabdab_summary.tsv")
    old_pdb_dir = os.path.join(args.pdb_dir, args.summary_prefix + "_sabdab")
    old_fasta_dir = os.path.join(args.fasta_dir, args.summary_prefix + "_sabdab")
    
    train_pdb_dir = os.path.join(args.pdb_dir, args.summary_prefix + "_train")
    train_fasta_dir = os.path.join(args.fasta_dir, args.summary_prefix + "_train")
    valid_pdb_dir = os.path.join(args.pdb_dir, args.summary_prefix + "_valid")
    valid_fasta_dir = os.path.join(args.fasta_dir, args.summary_prefix + "_valid")
    for dir_ in [train_pdb_dir, train_fasta_dir, valid_pdb_dir, valid_fasta_dir]:
        os.makedirs(dir_, exist_ok=True)
    train_pdbs, valid_pdbs = get_split_data(old_pdb_dir, summary_path, args.train_ratio)
    
    # 2. copy data
    logging.info("copying pdbs and fastas...")
    for pdb in train_pdbs:
        copy_pdb_to_new_dir(
            pdb, old_pdb_dir, old_fasta_dir, train_pdb_dir, train_fasta_dir
        )
    for pdb in valid_pdbs:
        copy_pdb_to_new_dir(
            pdb, old_pdb_dir, old_fasta_dir, valid_pdb_dir, valid_fasta_dir
        )
    
    # 3. merge train/valid fastas into the info directory.
    merged_fasta_dir = os.path.join(args.fasta_dir, "merged")
    os.makedirs(merged_fasta_dir, exist_ok=True)
    
    # 3.1 merge train/valid
    logging.info("merging sabdab fastas...")
    train_merged_fasta_path = os.path.join(merged_fasta_dir, args.summary_prefix + "_train.fasta")
    valid_merged_fasta_path = os.path.join(merged_fasta_dir, args.summary_prefix + "_valid.fasta")
    # deepab's format, used for torchfold.
    merge_fasta(fasta_dir=train_fasta_dir, output_path=train_merged_fasta_path, mode=-1)
    merge_fasta(fasta_dir=valid_fasta_dir, output_path=valid_merged_fasta_path, mode=-1)
    
    if args.merge_rosetta:
        logging.info("merging rosetta fastas...")
        rosetta_merged_fasta_path = os.path.join(merged_fasta_dir, "rosetta.fasta")
        rosetta_cat30x_merged_fasta_path = os.path.join(merged_fasta_dir, "rosetta_cat30x.fasta")
        rosetta_fasta_dir = os.path.join(args.fasta_dir, "rosetta")
        merge_fasta(fasta_dir=rosetta_fasta_dir, output_path=rosetta_merged_fasta_path, mode=-1)
        merge_fasta(fasta_dir=rosetta_fasta_dir, output_path=rosetta_cat30x_merged_fasta_path, mode=30)
        
    if args.merge_therapeutics:
        logging.info("merging therapeutics fastas...")
        therapeutics_merged_fasta_path = os.path.join(merged_fasta_dir, "therapeutics.fasta")
        therapeutics_cat30x_merged_fasta_path = os.path.join(merged_fasta_dir, "therapeutics_cat30x.fasta")
        therapeutics_fasta_dir = os.path.join(args.fasta_dir, "therapeutics")
        merge_fasta(fasta_dir=therapeutics_fasta_dir, output_path=therapeutics_merged_fasta_path, mode=-1)
        merge_fasta(fasta_dir=therapeutics_fasta_dir, output_path=therapeutics_cat30x_merged_fasta_path, mode=30)
        

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
        "--train_ratio", type=float, default=0.9,
        help="the proportion of the train data"
    )
    parser.add_argument(
        '--merge_rosetta', type=bool_type, default=False,
        help='whether to merge rosetta fastas'
    )
    parser.add_argument(
        '--merge_therapeutics', type=bool_type, default=False,
        help='whether to merge therapeutics fastas'
    )
    args = parser.parse_args()

    main(args)