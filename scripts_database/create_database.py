# Code adpated from the source code of deepab
# https://github.com/RosettaCommons/DeepAb/blob/main/deepab/preprocess/create_antibody_db.py

# (c) Copyright Rosetta Commons Member Institutions.
# (c) This file is part of the Rosetta software suite and is made available under license.
# (c) The Rosetta software is developed by the contributing members of the Rosetta Commons.
# (c) For more information, see http://www.rosettacommons.org. Questions about this can be
# (c) addressed to University of Washington CoMotion, email: license@uw.edu.

## @details download non-redundant Chothia Abs from SAbDab
## Abs are downloaded by html query (is there a better practice)?
## Abs are Chothia-numbered, though we use Kabat to define CDRs.
## After download, trim Abs to Fv and extract FR and CDR sequences.
## For the purpose of trimming, truncate the heavy @112 and light @109
import argparse
import os
import sys
import numpy as np
from typing import Optional, Any
import logging
logging.basicConfig(level=logging.INFO)

from datetime import datetime
import requests
import pandas as pd
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial 

from openfold.utils import data_utils


def parse_sabdab_summary(file_name):
    """
    SAbDab produces a rather unique summary file.
    This function reads that file into a dict with the key being the
    4-letter PDB code.
    
    dict[pdb] = {col1 : value, col2: value, ...}
    """
    sabdab_dict = {}
    possible_antigen_type = ['protein', 'peptide', 'protein | protein']
    # possible_antigen_type = ['protein', 'peptide']

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
            if td.get("antigen_type", None) in possible_antigen_type:
                # add temporary dict to sabdab dict at the pdb id
                sabdab_dict[split_line[0]] = td

    return sabdab_dict


def download_file(url, output_path):
    with open(output_path, 'w') as f:
        f.write(requests.get(url).content.decode("utf-8"))


def download_chothia_pdb_files(
    pdb_ids,
    output_dir,
    max_workers=16,
):
    """
    Args:
        pdb_ids (set): A set of PDB IDs to download
        output_dir (str): Path to the directory to save the PDB files to.
        max_workers (int, optional): Max number of workers in the thread pool
            while downloading. Defaults to 16.
    """
    
    pdb_file_paths = [
        os.path.join(output_dir, pdb + ".pdb") for pdb in pdb_ids
    ]

    # Download PDBs using multiple threads
    download_url = "http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/pdb/{}/?scheme=chothia"
    urls = [download_url.format(pdb) for pdb in pdb_ids]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [
            executor.submit(lambda a: download_file(*a), args)
            for args in zip(urls, pdb_file_paths)
        ]
        logging.info(
            f"Downloading chothia files to {output_dir} from {download_url} ..."
        )
        for _ in tqdm(as_completed(results), total=len(urls)):
            pass
        

# ununsed. we generate fastas from protein object generated from pdb by deepmind's protein class.
def download_fasta_files(
    pdb_ids,
    output_dir,
    max_workers=16,
):
    """
    Args:
        pdb_ids (set): A set of PDB IDs to download
        output_dir (str): Path to the directory to save the fasta files to.
        max_workers (int, optional): Max number of workers in the thread pool
            while downloading. Defaults to 16.
    """    

    fasta_file_paths = [
        os.path.join(output_dir, pdb + ".fasta") for pdb in pdb_ids
    ]

    # Download fastas using multiple threads
    download_url = "https://www.rcsb.org/fasta/entry/{}"
    urls = [download_url.format(pdb) for pdb in pdb_ids]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [
            executor.submit(lambda a: download_file(*a), args)
            for args in zip(urls, fasta_file_paths)
        ]
        logging.info(
            f"Downloading fasta files to {output_dir} from {download_url} ..."
        )
        for _ in tqdm(as_completed(results), total=len(urls)):
            pass
        
def download_sabdab_summary_file(
    output_info_dir,
    ABtype="Fv",
    method="All",
    species="All",
    resolution=4,
    rfactor='',
    antigen="All",
    ltype="All",
    constantregion="All",
    affinity="All",
    isin_covabdab="All",
    isin_therasabdab="All",
    chothiapos='',
    restype="ALA",
    field_0="Antigens",
    keyword_0='',
    **kwargs,
):
    """Find antibody structures that have been deposited in the PDB.
    ("http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/")

    Args:
        summary_file_path (str): Output directory containing a summary CSV file.
    """
    date_string = datetime.today().strftime("%Y%m%d")
    summary_path = os.path.join(
        output_info_dir, f"{date_string}_{ABtype}_{resolution}_{antigen}_sabdab_summary.tsv"
    )
    if os.path.exists(summary_path):
        logging.info("sabdab summary file exists: {summary_path}")
        return summary_path

    logging.info("start querying sabdab database...")
    base_url = "http://opig.stats.ox.ac.uk"
    search_url = os.path.join(base_url, "webapps/newsabdab/sabdab/search/")
    params = dict(
        ABtype=ABtype,
        method=method,
        species=species,
        resolution=resolution,
        rfactor=rfactor,
        antigen=antigen,
        ltype=ltype,
        constantregion=constantregion,
        affinity=affinity,
        isin_covabdab=isin_covabdab,
        isin_therasabdab=isin_therasabdab,
        chothiapos=chothiapos,
        restype=restype,
        field_0=field_0,
        keyword_0=keyword_0,
    )
    query = requests.get(search_url, params=params)
    html = BeautifulSoup(query.content, "html.parser")
    summary_file_url = base_url + html.find(
        id="downloads").find('a').get("href")
    logging.info(
        f"Downloading sabdab summary to {output_info_dir} from {summary_file_url} ..."
    )
    download_file(summary_file_url, summary_path)
    return summary_path


def _get_HL_chains(pdb_path):
    """Gets the heavy and light chain ID's from a chothia file from SAbDab"""
    # Get the line with the HCHAIN and LCHAIN
    hl_line = ''
    with open(pdb_path) as f:
        for line in f.readlines():
            if 'PAIRED_HL' in line:
                hl_line = line
                break
    if hl_line == '':
        return None, None

    words = hl_line.split(' ')
    h_chain = l_chain = None
    for word in words:
        if word.startswith('HCHAIN'):
            h_chain = word.split('=')[1]
        if word.startswith('LCHAIN'):
            l_chain = word.split('=')[1]
    return h_chain, l_chain


def truncate_chain(pdb_text, chain, resnum, newchain):
    """
    Read PDB line by line and return all lines for a chain,
    with a resnum less than or equal to the input.
    This has to be permissive for insertion codes.
    This will return only a single truncated chain.
    This function can update chain to newchain.
    """
    trunc_text = ""
    for line in pdb_text.split("\n"):
        if (line.startswith("ATOM") and line[21] == chain
                and int(line[22:26]) <= resnum):
            trunc_text += line[:21] + newchain + line[22:]
            trunc_text += "\n"
    return trunc_text


def truncate_antibody_pdb(
    pdb_id, data_dir, output_dir,
    ignore_same_VL_VH_chains,
    warn_pdbs,
    same_chain,
    sabdab_summary_file: Optional[dict] = None,
):
    """
    Args:
        pdb_id (str): The PDB ID of the protein
        data_dir (str): 
            The directory containing all the chothia numbered pdb files ([pdb_id].pdb).
        output_dir (str): 
            The directory containing the truncated antibody pdbs.        
        ignore_same_VL_VH_chains (bool): 
            Whether or not to ignore a PDB file when VH & VL are on the same chain.
        same_chain (list): 
            set of pdb_ids which contains the same chain.
        warn_pdbs (list):
            set of pdb_ids which raise warnings. We append warnings to it.
        sabdab_summary_file (dict): summary file
    """
     
    # check if 'pdb_trunc.pdb' exists, if not then generate it
    pdb_path = os.path.join(data_dir, pdb_id + ".pdb")
    trunc_pdb_path = os.path.join(output_dir, pdb_id + ".pdb")
    if os.path.isfile(trunc_pdb_path):
        logging.info(
            f"a truncated version of {pdb_id} is found. Skipping..."
        )
        os.remove(pdb_path)  # delete old file
        return

    pdb_text = ''
    try:
        with open(pdb_path, 'r') as f:
            pdb_text = f.read()  # want string not list
    except IOError:
        sys.exit(f"Failed to open {pdb_id} in antibody database {pdb_path}")
    if len(pdb_text) == 0:
        sys.exit(f"Nothing parsed for PDB {pdb_id} !")

    # Get the hchain and lchain data from the SAbDab summary file, if given
    hchain_text, lchain_text = '', ''
    antigen_text = ''
    antigen_chain = 'NA'
    if sabdab_summary_file is not None:
        hchain = sabdab_summary_file[pdb_id]["Hchain"] # usually a uppercase letter
        lchain = sabdab_summary_file[pdb_id]["Lchain"]
        antigen_chain = sabdab_summary_file[pdb_id].get("antigen_chain", 'NA')
    else:
        hchain, lchain = _get_HL_chains(pdb_path)

    if ignore_same_VL_VH_chains and hchain == lchain:
        same_chain.append(pdb_id)
        logging.warning(f"{pdb_id} has the VH+VL on a single chain, removing...")
        os.remove(pdb_path)
        return

    if hchain == 'NA' or lchain == 'NA':
        warn_pdbs.append(pdb_id)
        logging.warning(f"one chain of {pdb_id} is NA, H: {hchain}, L: {lchain}, removing...")
        os.remove(pdb_path)
        return
    
    if antigen_chain == 'NA':
        warn_pdbs.append(pdb_id)
        logging.warning(f"missing antigen chain, removing...")
        os.remove(pdb_path)
        return

    antigen_chain = antigen_chain.split('|')
    antigen_chain = [c.strip() for c in antigen_chain]
    new_antigen_chain = ['A', 'B']
    for idx_, antigen_chain_ in enumerate(antigen_chain):
        antigen_text += truncate_chain(
            pdb_text, antigen_chain_, 9999, new_antigen_chain[idx_]
        )
    if len(antigen_text) == 0:
        warn_pdbs.append(pdb_id)
        logging.warning(
            f"could not find antigen chain for {pdb_id}!\n"
            "It was not reported to be NA, so the file may have been altered!"
        )
        os.remove(pdb_path)
        return

    hchain_text = truncate_chain(pdb_text, hchain, 112, 'H')
    if len(hchain_text) == 0:
        warn_pdbs.append(pdb_id)
        logging.warning(
            f"could not find {hchain} chain for {pdb_id}!\n"
            "It was not reported to be NA, so the file may have been altered!"
        )
        os.remove(pdb_path)
        return

    lchain_text = truncate_chain(pdb_text, lchain, 109, 'L')
    if len(lchain_text) == 0:
        warn_pdbs.append(pdb_id)
        logging.warning(
            f"could not find {lchain} chain for {pdb_id}!\n"
            "It was not reported to be NA, so the file may have been altered!"
        )
        os.remove(pdb_path)
        return

    # write new file to avoid bugs from multiple truncations
    with open(trunc_pdb_path, 'w') as f:
        f.write(hchain_text + lchain_text + antigen_text)

    # remove old file
    os.remove(pdb_path)
    

def truncate_antibody_pdbs(
    data_dir,
    output_dir,
    ignore_same_VL_VH_chains=True,
    sabdab_summary_path: Optional[str] = None,
):
    """
    We only use the Fv as a template, so this function loads each pdb
    and deletes excess chains/residues. We define the Fv, under the
    Chothia numbering scheme as H1-H112 and L1-L109.
    """
    warn_pdbs = []  # count warnings
    same_chain = []  # count deleted files (VH+VL are on the same chain)

    # iterate over PDBs in antibody_database and truncate accordingly
    unique_pdbs = set([
        x[:4] for x in os.listdir(data_dir) if x.endswith(".pdb")
    ])

    sabdab_summary_file = None
    if sabdab_summary_path is not None:
        sabdab_summary_file = parse_sabdab_summary(sabdab_summary_path)

    logging.info("Truncating pdb files...")
    for pdb in tqdm(unique_pdbs):
        truncate_antibody_pdb(
            pdb,
            data_dir,
            output_dir,
            ignore_same_VL_VH_chains=ignore_same_VL_VH_chains,
            warn_pdbs=warn_pdbs,
            same_chain=same_chain,
            sabdab_summary_file=sabdab_summary_file,
        )

    logging.info(
        f"Finished truncating!\n"
        f"Deleted {len(same_chain)} pdbs from database because it has VH+VL on the same chain.\n"
        f"Deleted {len(warn_pdbs)} pdbs from database because some errors happen"
    )


def fasta_from_truncated_pdbs(
    data_dir,
    output_dir,
    max_workers=16,
):
    job_args = [x for x in os.listdir(data_dir) if x.endswith(".pdb")]
    job_args = zip(job_args, list(range(len(job_args))))
    
    with multiprocessing.Pool(max_workers) as p:
        func = partial(data_utils.pdb2fasta, data_dir=data_dir)
        for (ret, basename) in p.starmap(func, job_args):
            assert len(ret) % 2 == 0, 'length of the returned fasta should be even'
            trunc_fasta_path = os.path.join(output_dir, basename + ".fasta")
            with open(trunc_fasta_path, "w") as f:
                f.write('\n'.join(ret))

def cdrfasta_from_truncated_pdbs(
    data_dir,
    output_dir,
    cdr_idx,
    save_single=True,
    max_workers=16,
):
    job_args = [x for x in os.listdir(data_dir) if x.endswith(".pdb")]
    job_args = zip(job_args, list(range(len(job_args))))
    name = ['cdrh1', 'cdrh2', 'cdrh3', 'cdrl1', 'cdrl2', 'cdrl3']
    
    with multiprocessing.Pool(max_workers) as p:
        func = partial(data_utils.pdb2cdrfasta, data_dir=data_dir, cdr_idx=cdr_idx)
        if save_single:
            fastas = []
            for (ret, basename) in p.starmap(func, job_args):
                fastas.extend(ret)
            trunc_fasta_path = os.path.join(output_dir, name[cdr_idx - 1] + ".fasta")
            with open(trunc_fasta_path, "w") as f:
                f.write('\n'.join(fastas))
        else:
            for (ret, basename) in p.starmap(func, job_args):
                trunc_fasta_path = os.path.join(output_dir, f"{basename}_{name[cdr_idx - 1]}.fasta")
                with open(trunc_fasta_path, "w") as f:
                    f.write('\n'.join(ret))


def dataset_gen(
    pdb_ids,
    download_dir,
    output_pdb_dir,
    output_fasta_dir,
    output_cdr_fasta_dir=None,
    sabdab_summary_path=None,
    ignore_same_VL_VH_chains=True,
):
    """dataset generator

    Args:
        pdb_ids (set): a set containing pdb_ids for the dataset to generate
        download_dir (str): a temporary diretory containing the raw pdbs downloaded from sabdab database
        output_pdb_dir (str): a directory containing the final truncated pdbs.
        output_fasta_dir (str): a directory containing the final truncated fastas.        
        sabdab_summary_path (str): 
            Path to the summary file, from which we extrac heavy/light chain letters.
            Defaults to None.
        ignore_same_VL_VH_chains (bool, optional):
            Whether or not to ignore a PDB file when VH & VL are on the same chain.. Defaults to True.
    """

    # 1. mkdirs
    for dir_ in [
        download_dir, output_pdb_dir, output_cdr_fasta_dir
    ]:
        os.makedirs(dir_, exist_ok=True)

    # 2. download
    download_chothia_pdb_files(
        pdb_ids=pdb_ids,
        output_dir=download_dir,
    )
    # 3. truncate pdbs
    truncate_antibody_pdbs(
        data_dir=download_dir,
        output_dir=output_pdb_dir,
        sabdab_summary_path=sabdab_summary_path,
        ignore_same_VL_VH_chains=ignore_same_VL_VH_chains,
    )
    
    # 4. get cdr fasta
    if output_cdr_fasta_dir is not None:
        for cdr_idx in range(1, 6 + 1):
            cdrfasta_from_truncated_pdbs(
                data_dir=output_pdb_dir,
                output_dir=output_cdr_fasta_dir,
                cdr_idx=cdr_idx,
            )
    # 5. truncate antigen
    # we do it on the fly
    
    # 6. get trucated fastas from truncated pdbs
    # we do not need it as we will provide the framework structure during inference
    # fasta_from_truncated_pdbs(
    #     data_dir=output_pdb_dir,
    #     output_dir=output_fasta_dir,
    # )


def main(args):

    # 1. make dir
    os.makedirs(args.output_info_dir, exist_ok=True)
      
    # 3. fetch summary file if not exists
    summary_path = download_sabdab_summary_file(**vars(args))
    # *_sabdab_summary.tsv
    summary_prefix, _ = os.path.splitext(os.path.basename(summary_path))
    summary_prefix = '_'.join(summary_prefix.split('_')[:-2])
    
    summary_df = pd.read_csv(summary_path, sep='\t')
    
    is_protein = summary_df['antigen_type'].isin(['protein', 'peptide', 'protein | protein'])
    summary_df = summary_df[is_protein]
    # only protein or peptide antigen is accepted
    sabdab_pdb_ids = set(summary_df['pdb'].unique())
    logging.info(f"prefiltering {len(sabdab_pdb_ids)} pdbs")

    # 5. gen sabdab database
    logging.info(f"generating sabdab database with prefix: {summary_prefix}...")
    sabdab_download_dir = os.path.join(args.download_dir, f"{summary_prefix}_sabdab")
    sabdab_pdb_dir = os.path.join(args.output_pdb_dir, f"{summary_prefix}_sabdab")
    sabdab_fasta_dir = os.path.join(args.output_fasta_dir, f"{summary_prefix}_sabdab")
    output_cdr_fasta_dir = os.path.join(args.output_fasta_dir, f"{summary_prefix}_cdr")
    dataset_gen(
        pdb_ids=sabdab_pdb_ids,
        download_dir=sabdab_download_dir,
        output_pdb_dir=sabdab_pdb_dir,
        output_fasta_dir=sabdab_fasta_dir,
        output_cdr_fasta_dir=output_cdr_fasta_dir,
        sabdab_summary_path=summary_path,
        ignore_same_VL_VH_chains=True,
    )
 
    # 8. remove the download file
    if os.path.exists(args.download_dir):
        try:
            os.rmdir(args.download_dir)
        except:
            logging.warning(
                f"the download directory: {args.download_dir} is not empty!"
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
        "download_dir", type=str,
        help="Path to a directory containing downlowned raw pdbs"
    )
    parser.add_argument(
        "output_pdb_dir", type=str,
        help="Path to a directory containing truncated pdbs"
    )
    parser.add_argument(
        "output_fasta_dir", type=str,
        help="Path to a directory containing truncated fastas"
    )
    parser.add_argument(
        "output_info_dir", type=str,
        help="Path to a directory containing information about the database, e.g., summary file"
    )
    parser.add_argument(
        '--resolution', type=int, default=4,
        help='Resolution cutoff'
    )
    parser.add_argument(
        "--n_job", type=int, default=16,
        help="number of cpu jobs"
    )
    args = parser.parse_args()

    main(args)
