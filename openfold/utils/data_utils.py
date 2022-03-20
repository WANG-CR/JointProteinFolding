import argparse
import os
import numpy as np

from openfold.data import parsers
from openfold.np import protein, residue_constants

import multiprocessing
from functools import partial 


ROSETTA_ANTIBODY_BENCHMARK = [
    "1dlf", "1fns", "1gig", "1jfq", "1jpt", "1mfa", "1mlb", "1mqk", "1nlb", "1oaq",
    "1seq", "2adf", "2d7t", "2e27", "2fb4", "2fbj", "2r8s", "2v17", "2vxv", "2w60", 
    "2xwt", "2ypv", "3e8u", "3eo9", "3g5y", "3giz", "3gnm", "3go1", "3hc4", "3hnt",
    "3i9g", "3liz", "3lmj", "3m8o", "3mlr", "3mxw", "3nps", "3oz9", "3p0y", "3t65",
    "3umt", "3v0w", "4f57", "4h0h", "4h20", "4hpy", "4nzu",
]
THERAPEUTICS_BENCHMARK = [
    "1bey", "1cz8", "1mim", "1sy6", "1yy8", "2hwz", "3eo0", "3gkw", "3nfs", "3o2d",
    "3pp3", "3qwo", "3u0t", "4cni", "4dn3", "4g5z", "4g6k", "4hkz", "4i77", "4irz",
    "4kaq", "4m6n", "4nyl", "4od2", "4ojf", "4qxg", "4x7s", "4ypg", "5csz", "5dk3",
    "5ggq", "5ggu", "5i5k", "5jxe", "5kmv", "5l6y", "5n2k", "5nhw", "5sx4", "5tru",
    "5vh3", "5wuv", "5xxy", "5y9k", "6and",
]
LEN_ROSETTA = len(ROSETTA_ANTIBODY_BENCHMARK)
LEN_THERAPEUTICS = len(THERAPEUTICS_BENCHMARK)
assert len(np.intersect1d(ROSETTA_ANTIBODY_BENCHMARK, THERAPEUTICS_BENCHMARK)) == 0
assert len(np.unique(ROSETTA_ANTIBODY_BENCHMARK)) == LEN_ROSETTA
assert len(np.unique(THERAPEUTICS_BENCHMARK)) == LEN_THERAPEUTICS


def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]] 
        for i in range(len(aatype))
    ])
    
      
def _string_index_select(str, bool_index):
    str = np.array(list(str))
    str = list(str[bool_index])
    return ''.join(str)
    

def pdb2fasta(fname, idx, data_dir):
    basename, ext = os.path.splitext(fname)
    fpath = os.path.join(data_dir, fname)
    ret = []
    if ext == '.pdb':
        with open(fpath, 'r') as f:
            pdb_str = f.read()
        protein_object = protein.from_pdb_string_antibody(pdb_str)
        seq = _aatype_to_str_sequence(protein_object.aatype)
        
        ret.append(f">{basename}_H")
        ret.append(_string_index_select(seq, protein_object.chain_index==0))
        ret.append(f">{basename}_L")
        ret.append(_string_index_select(seq, protein_object.chain_index==1))
           
    else:
        raise ValueError(f'ext is invalid, should be either pdb of fasta, found {ext}')
    return ret, basename


def fastas2fasta(data_dir, mode=-1):
    """merge fastas in a directory into a single fasta
    mode: format of the output fasta
        mode==-1: deepab's format: e.g.,
            >1ay1_H
            EVQLQESGPGLVKPYQSLSLSCTVTGYSITSDYAWNWIRQFPGNKLEWMGYITYSGTTDYNPSLKSRISITRDTSKNQFFLQLNSVTTEDTATYYCARYYYGYWYFDVWGQGTTLTVS
            >1ay1_L
            DIQMTQSPAIMSASPGEKVTMTCSASSSVSYMYWYQQKPGSSPRLLIYDSTNLASGVPVRFSGSGSGTSYSLTISRMEAEDAATYYCQQWSTYPLTFGAGTKLELKRA
        mode>=0: 2-line fasta, with ``mode'' unknown residues (X) inserted between heavy ang light chains.
            e.g., mode=10
            >5ggu
            QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVIWYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDPRGATLYYYYYGMDVWGQGTTVTVS\
            XXXXXXXXXX
            DIQMTQSPSSLSASVGDRVTITCRASQSINSYLDWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYYSTPFTFGPGTKVEIKRT
    """
    fnames = [x for x in os.listdir(data_dir) if x.endswith('.fasta')]
    fastas = []
    for fname in fnames:
        fpath = os.path.join(data_dir, fname)
        basename, ext = os.path.splitext(fname)
        with open(fpath, 'r') as f:
            fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(fasta_str)
        if mode >= 0:
            assert len(input_seqs) == 2
            seq = input_seqs[0] + 'X' * mode + input_seqs[1]
            fastas.append(f">{basename}_cat{mode}x")
            fastas.append(seq)
        elif mode == -1:
            for (seq, desc) in zip(input_seqs, input_descs):
                fastas.append(f">{desc}")
                fastas.append(seq)
        else:
            raise ValueError(f"unsupported mode: {mode}!")
    return fastas

        
def main(args):
    fasta = []
    job_args = os.listdir(args.data_dir)
    job_args = zip(job_args, list(range(len(job_args))))
    with multiprocessing.Pool(args.n_job) as p:
        func = partial(pdb2fasta, data_dir=args.data_dir)
        for (ret, basename) in p.starmap(func, job_args):
            assert len(ret) % 2 == 0, 'length of the returned fasta should be even'
            fasta.extend(ret)
            
    with open(args.output_path, "w") as fp:
        fp.write('\n'.join(fasta))        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=str,
        help="Path to a directory containing mmCIF or .core files"
    )
    parser.add_argument(
        "output_path", type=str,
        help="Path to output FASTA file"
    )
    parser.add_argument(
        "--n_job", type=int, default=48,
        help="number of cpu jobs"
    )
    args = parser.parse_args()

    main(args)
    
    # python scripts_narval/pdb_to_fasta.py $SCRATCH/dataset/SAbDab_database/train/ $SCRATCH/dataset/SAbDab_database/fasta_from_pdb/train_HL.fasta --n_job 12
    # python scripts_narval/pdb_to_fasta.py $SCRATCH/dataset/SAbDab_database/valid/ $SCRATCH/dataset/SAbDab_database/fasta_from_pdb/valid_HL.fasta --n_job 12    
    # python scripts_narval/pdb_to_fasta.py $SCRATCH/dataset/SAbDab_database/rosetta_antibody_benchmark/ $SCRATCH/dataset/SAbDab_database/fasta_from_pdb/rosetta_HL.fasta --n_job 12        
    # python scripts_narval/pdb_to_fasta.py $SCRATCH/dataset/SAbDab_database/therapeutics/ $SCRATCH/dataset/SAbDab_database/fasta_from_pdb/therapeutics_HL.fasta --n_job 12            