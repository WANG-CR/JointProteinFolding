# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from datetime import date
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import os

import random
import sys
import time
import torch

from openfold.np import protein
import openfold.np.relax.relax as relax

def relax_pdb(old_pdb, relaxed_pdb, relaxer):
    with open(old_pdb, 'r') as f:
        pdb_str = f.read()    
    unrelaxed_protein = protein.from_pdb_string_antibody(pdb_str)
    t = time.perf_counter()
    relaxed_pdb_str, _, _ = relaxer.process(prot=unrelaxed_protein)
    logging.info(f"Relaxation time: {time.perf_counter() - t}")
    with open(relaxed_pdb, 'w') as fp:
        fp.write(relaxed_pdb_str)     
   
            
def main(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir, exist_ok=True)
    logging.info(f"target dir: {args.target_dir}")
    logging.info("start relaxation")
    amber_relaxer = relax.AmberRelaxation(
        **{
            "max_iterations": 0,  # no max
            "tolerance": 2.39,
            "stiffness": 10.0,
            "max_outer_iterations": 20,
            "exclude_residues": [],
            "use_gpu": True,
        }
    )
    jobs = [x for x in os.listdir(args.source_dir) if x.endswith('.pdb')]
    for fname in jobs:
        pdbid = fname[:4]
        #if pdbid in test_list:
        logging.info(f"relaxing {pdbid}...")
        src_path = os.path.join(args.source_dir, fname)
        tgt_path = os.path.join(args.target_dir, fname)
        relax_pdb(src_path, tgt_path, amber_relaxer)
     
    
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
        "source_dir", type=str,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "target_dir", type=str,
        help="Path to the native_pdb"
    )
    args = parser.parse_args()

    main(args)
    
