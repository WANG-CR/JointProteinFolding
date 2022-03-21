import argparse
import os
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_max, scatter_min
from torchdrug import data, models
from torchdrug.data import feature
from torchdrug.utils import comm

from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence descriptions taken from the comment lines. In the
            same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions

          
def main(args):

    # 1. get fastas and prepare data
    with open(args.fasta_path, "r") as fp:
        fasta_str = fp.read()
    input_seqs, input_descs = parse_fasta(fasta_str)
    batch_size = 100
    num_batches = (len(input_seqs) + batch_size - 1) // batch_size
    final_output = []
    
    # 2. init model.
    sd = torch.load(args.ckpt_path, map_location='cpu')["model"]
    sd = {k[len("model."):]:v for k,v in sd.items() if k[:3]!='mlp'} 
    
    additional_vocab = ["UNK", "<pad>", "<mask>", "<cls>", "<sep>", "<unk>"]
    vocab = feature.residue_vocab + additional_vocab    
    model = models.BERT(len(vocab))
    model.load_state_dict(sd) 
    if args.gpu:
        model = model.cuda(0)   
    model.eval()
    
    for batch_id in range(num_batches):
        logging.info(f"processing {batch_id}-th batch...")
        proteins = []
        batch_seqs = input_seqs[batch_id * batch_size: (batch_id+1) * batch_size]
        batch_descs = input_descs[batch_id * batch_size: (batch_id+1) * batch_size]
        
        for i, (seq, tag) in enumerate(zip(batch_seqs, batch_descs)):
            protein = data.Protein.from_sequence(seq, residue_feature="default")
            proteins.append(protein)
        proteins = data.Protein.pack(proteins)
        

        if proteins.residue_feature.shape[-1] < len(vocab):
            append_residue_feature = torch.zeros(
                proteins.residue_feature.shape[:1] + (len(vocab) - proteins.residue_feature.shape[-1],),
                device=proteins.device)
            proteins.residue_feature = torch.cat([proteins.residue_feature, append_residue_feature], dim=-1) 
                
        if args.gpu:
            proteins = proteins.cuda(0)

        with torch.no_grad():
            output  = model(proteins, proteins.residue_feature.float())
        residue_feature = output["residue_feature"].cpu()
        
        offset = proteins.num_cum_residues - proteins.num_residues
        for i in range(len(proteins)):
            final_output.append(residue_feature[offset[i] : proteins.num_cum_residues[i]])
            
    logging.info(f"saving {len(final_output)} embeddings into {args.output_dir}...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    for tag, emb in zip(input_descs, final_output):
        output_path = os.path.join(args.output_dir, tag + '.oaspt')
        torch.save(emb.clone(), output_path)


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
        "fasta_path", type=str,
        help="Path to the fasta file"
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Path to a output directory containing merged esm representations"
    )
    
    parser.add_argument(
        "ckpt_path", type=str,
        help="Path to a ckpt"
    )  
    parser.add_argument(
        "--gpu", type=bool_type, default=True,
        help="""Whether to use gpu"""
    )      

    args = parser.parse_args()

    main(args)
    
    # python scripts_narval/extract_oas.py $SCRATCH/dataset/SAbDab_database/fasta_from_pdb/rosetta_HL.fasta $SCRATCH/dataset/SAbDab_database/oas_unpaired/rosetta $SCRATCH/pretrained_model/oas/oasunpaired_bertbase_epoch_25.pth
    # python scripts_narval/extract_oas.py $SCRATCH/dataset/SAbDab_database/fasta_from_pdb/therapeutics_HL.fasta $SCRATCH/dataset/SAbDab_database/oas_unpaired/therapeutics $SCRATCH/pretrained_model/oas/oasunpaired_bertbase_epoch_25.pth
    # python scripts_narval/extract_oas.py $SCRATCH/dataset/SAbDab_database/fasta_from_pdb/valid_HL.fasta $SCRATCH/dataset/SAbDab_database/oas_unpaired/valid $SCRATCH/pretrained_model/oas/oasunpaired_bertbase_epoch_25.pth
    # python scripts_narval/extract_oas.py $SCRATCH/dataset/SAbDab_database/fasta_from_pdb/train_HL.fasta $SCRATCH/dataset/SAbDab_database/oas_unpaired/train $SCRATCH/pretrained_model/oas/oasunpaired_bertbase_epoch_25.pth