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

def is_same_seq(prot: protein.Protein):
    restypes = residue_constants.restypes + ['X']
    decoded_seq = ""
    aatype = prot.aatype
    for i in range(aatype.shape[0]):
        decoded_seq += restypes[aatype[i]]
    
def convert_to_seq(prot: protein.Protein):
    restypes = residue_constants.restypes + ['X']
    decoded_seq = ""
    aatype = prot.aatype
    for i in range(aatype.shape[0]):
        decoded_seq += restypes[aatype[i]]  
    return decoded_seq

def save_protein(prot: protein.Protein, chain_id, accept, fname, output_dir):
    if accept:
        tgt_path = os.path.join(output_dir, fname+chain_id+".pdb")
        logging.info(f"saving pdb to path {tgt_path}")
        with open(tgt_path, 'w') as f:
            f.write(protein.to_pdb_with_chain_name(prot,chain_id))

# def save_fasta(prot: protein.Protein, chain_id, accept, fname, output_dir):
#     if accept:
#         tgt_path = os.path.join(output_dir, fname+chain_id+".fasta")
#         sequence = convert_to_seq(prot)
#         with open(tgt_path, 'w') as f:
#             f.write(">"+fname+chain_id+os.linesep)
#             f.write(sequence)

# write to unique fasta file
def save_fasta(prot: protein.Protein, chain_id, accept, fname, output_dir):
    if accept:
        tgt_path = os.path.join(output_dir, "all.fasta")
        sequence = convert_to_seq(prot)
        with open(tgt_path, 'a') as f:
            f.write(">"+fname+chain_id+os.linesep)
            f.write(sequence+os.linesep)

def reject_aa_80_len_20(prot: protein.Protein):
    aatype = prot.aatype
    if len(aatype) <= 20:
        return False
    _, counts = np.unique(aatype, return_counts=True)
    fractions = counts/len(aatype)
    for i in range(len(fractions)):
        if fractions[i] >= 0.8:
            return False
    return True

def is_large_prot(prot: protein.Protein, max_len, fname):
    if prot.aatype.shape[0] <= max_len:
        return False
    else:
        return True

def do(fname, src_path, fasta_output_dir, pdb_output_dir):   
    with open(src_path, 'r') as f:
        pdb_str = f.read()
    try:
        protein_objects, unique_chain_ids = protein.from_pdb_string_multichain(pdb_str)
        assert len(protein_objects) == len(unique_chain_ids)
        assert len(protein_objects[0].atom_positions) > 0
    except AssertionError as e:
        return (0, 0, 1, 0)

    accept = list(map(reject_aa_80_len_20, protein_objects))
    count_accept = sum(accept)
    count_reject = len(accept) - sum(accept)

    job_args = zip(protein_objects, unique_chain_ids, accept)
    func1 = partial(save_protein, fname=fname, output_dir=pdb_output_dir)
    func2 = partial(save_fasta, fname=fname, output_dir=fasta_output_dir)
    for prot, chain_id, accept in job_args:
        func1(prot, chain_id, accept)
        func2(prot, chain_id, accept)
    # logging.info(f"executing func1")
    # map(func1, job_args)
    # logging.info(f"executing func2")
    # map(func2, job_args)
    return (count_accept, count_reject, 0, 1)


def main(args):
    filenames = [x[0:4] for x in os.listdir(args.input_dir) if ".pdb" in x]
    assert len(filenames) == len(set(filenames))
    logging.info(f"get {len(filenames)} files.")

    args.output_dir = args.output_dir + \
                      f"_80identity_seperateChains"
    fasta_output_dir = os.path.join(args.output_dir, "fasta")
    pdb_output_dir = os.path.join(args.output_dir, "pdb")
    output_dirs = [fasta_output_dir, pdb_output_dir]
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    open(os.path.join(fasta_output_dir, "all.fasta"), "w").close()
        
    src_paths = []
    for fname in filenames:
        src_path = os.path.join(args.input_dir, fname+".pdb")
        src_paths.append(src_path)
    
    job_args = zip(filenames, src_paths)
    count_accept, count_reject, count_error, count_pdb = 0,0,0,0
    with multiprocessing.Pool(args.num_workers) as p:
        func = partial(do, fasta_output_dir=fasta_output_dir, pdb_output_dir=pdb_output_dir)
        for output in p.starmap(func, job_args):
            count_accept += output[0]
            count_reject += output[1]
            count_error += output[2]
            count_pdb += output[3]
        print(f"processing file successful. No. {count_pdb}")
    logging.info(
        f"parse error: {count_error} pdb files\n"
        f"parse succesful: {count_pdb} pdb files\n"
        f"{count_accept} chains are saved.\n"
        f"{count_reject} chains are exclueded due to 80% identity.\n"
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
        "--num_workers", type=int, default=36,
        help="Path to model parameters."
    )
    args = parser.parse_args()
    main(args)
