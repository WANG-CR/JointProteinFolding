import argparse
import imp
import os
import sys
import logging
from xml.etree.ElementInclude import default_loader
logging.basicConfig(level=logging.INFO)
import shutil
import random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

import multiprocessing
from functools import partial

from openfold.utils import data_utils


def cdrsfasta_from_truncated_pdbs(
    data_dir,
    output_dir,
    save_single=True,
    max_workers=16,
):
    job_args = [x for x in os.listdir(data_dir) if x.endswith(".pdb")]
    job_args = zip(job_args, list(range(len(job_args))))

    with multiprocessing.Pool(max_workers) as p:
        func = partial(data_utils.pdb2cdrsfasta, data_dir=data_dir)
        if save_single:
            fastas = []
            for (ret, basename) in p.starmap(func, job_args):
                fastas.extend(ret)
            trunc_fasta_path = os.path.join(output_dir, "cdrs.fasta")
            with open(trunc_fasta_path, "w") as f:
                f.write('\n'.join(fastas))
        else:
            for (ret, basename) in p.starmap(func, job_args):
                trunc_fasta_path = os.path.join(output_dir, f"{basename}_cdrs.fasta")
                with open(trunc_fasta_path, "w") as f:
                    f.write('\n'.join(ret))


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


def get_cluster_rep(cluster_res):
    cluster_dict = defaultdict(list)
    with open(cluster_res, 'r') as f:
        for line in f:
            k, v = line.strip().split('\t')
            k = k.rsplit('_', 1)[0]
            v = v.rsplit('_', 1)[0]
            cluster_dict[k].append(v)

    cluster_rep = sorted(list(cluster_dict.keys()))
    logging.info(
        f"get {len(cluster_rep)} cluster representatives\n"
    )
    return cluster_rep


def main_get_cdrs(args):

    os.makedirs(args.output_dir, exist_ok=True)

    # 1.
    cdrsfasta_from_truncated_pdbs(args.input_dir, args.output_dir)


def main_copy_pdbs(args):

    os.makedirs(args.output_dir, exist_ok=True)

    # 2.
    cluster_rep = get_cluster_rep(args.cluster)
    logging.info("copying pdbs...")
    for pdb in cluster_rep:
        copy_pdb_to_new_dir(
            pdb, args.input_dir, args.output_dir
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Path to a directory containing truncated pdbs"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Path to a directory containing truncated fastas"
    )
    parser.add_argument(
        "--cluster", type=str, default=None,
        help="Path to a directory containing truncated fastas"
    )
    args = parser.parse_args()

    if args.cluster is None:
        main_get_cdrs(args)
    else:
        main_copy_pdbs(args)