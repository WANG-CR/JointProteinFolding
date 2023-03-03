import argparse
import logging
import os
# import sys
# import time
# import pickle
import shutil
import pickle
import re

from openfold.np import protein, residue_constants
logging.basicConfig(level=logging.INFO)
import numpy as np


# def process_name(list_names):
#     output_names = []
#     for name in list_names:
#         output_names.append(re.sub('.', '', name))
    

def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')

def convert_to_37(aatype, atom_positions):
    aatype = residue_constants.sequence_to_index(aatype)
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]
    # 37 atom types
    # 0,1,2,4 is N, CA, C, O
    length = atom_positions.shape[0]
    atom_positions1 = atom_positions[:, 0:3, :]
    atom_positions2 = np.zeros((length,1,3))
    atom_positions3 = atom_positions[:, None, 3, :]
    atom_positions4 = np.zeros((length,32,3))
    atom_positions = np.concatenate((atom_positions1,atom_positions2,atom_positions3,atom_positions4), axis=1)
    print(atom_positions.shape)

    atom_mask1 = np.ones((length,3))
    atom_mask2 = np.zeros((length,1))
    atom_mask3 = np.ones((length,1))
    atom_mask4 = np.zeros((length,32))
    atom_mask = np.concatenate((atom_mask1, atom_mask2, atom_mask3, atom_mask4), axis=1)

    print(atom_mask.shape)
    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    # atom_mask: np.ndarray  # [num_res, num_atom_type]
    return aatype, atom_positions, atom_mask




def main(args):
    with open(args.list_path, "rb") as f:
        data = pickle.load(f)
    
    directory = args.output_dir
    if not os.path.exists(directory):
        os.makedirs(directory)

    for samples in data:
        tag = samples["tag"]
        tag2 = tag[0:4]+tag[5] 
        length = samples["length"]
        sequence = samples["sequence"]
        coords = samples["coords"]
        residue_index = np.arange(length, dtype=int)
        assert length == coords.shape[0]
    
        # sequence is "A"-format
        # coords is numpy array of shape [length, 4, 3]
        # in order N, CA, C, O
        aatype, atom_positions, atom_mask = convert_to_37(sequence, coords)

        unrelaxed_protein = protein.from_cath(aatype, atom_positions, atom_mask, residue_index)
        unrelaxed_output_path = os.path.join(directory, f"{tag2}.pdb")
        with open(unrelaxed_output_path, 'w') as f:
            f.write(protein.to_pdb(unrelaxed_protein))

    # samples = data[0]
    # tag = samples["tag"]
    # length = samples["length"]
    # sequence = samples["sequence"]
    # coords = samples["coords"]
    # residue_index = np.arange(length, dtype=int)
    # assert length == coords.shape[0]


    # aatype, atom_positions, atom_mask = convert_to_37(sequence, coords)

    # unrelaxed_protein = protein.from_cath(aatype, atom_positions, atom_mask, residue_index)
    # unrelaxed_output_path = os.path.join(directory, f"{tag}.pdb")
    # with open(unrelaxed_output_path, 'w') as f:
    #     f.write(protein.to_pdb(unrelaxed_protein))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_path", type=str, default=None,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )

    args = parser.parse_args()
    main(args)