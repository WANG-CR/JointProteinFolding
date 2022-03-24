import argparse
import os
import numpy as np
import time
import requests

import logging
logging.basicConfig(level=logging.INFO)
import pyrosetta

test_list = ['6u6o', '6xm2', '6ztd', '7b0b', '7daa', '7e86', '7kba', '7m7w', '7n3g', '7vux']

def renumber_pdb(old_pdb, renum_pdb):
    success = False
    time.sleep(3)
    for i in range(10):
        try:
            with open(old_pdb, 'rb') as f:
                response = requests.post(
                    "http://www.bioinf.org.uk/abs/abnum/abnumpdb.cgi",
                    params={
                        "plain": "1",
                        "output": "-HL",
                        "scheme": "-c"
                    },
                    files={"pdb": f})

            success = response.status_code == 200 and not ("<html>"
                                                           in response.text)

            if success:
                break
            else:
                time.sleep((i + 1) * 3)
        except requests.exceptions.ConnectionError:
            time.sleep(60)

    # if success:
    print('success state', success)
    new_pdb_data = response.text
    with open(renum_pdb, "w") as f:
        f.write(new_pdb_data)
        
        



def main(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir, exist_ok=True)
    
    jobs = [x for x in os.listdir(args.source_dir) if x.endswith('.pdb')]
    for fname in jobs:
        pdbid = fname[:4]
        #if pdbid in test_list:
        if 1:            
            logging.info(f"renumbering {fname}...")
            src_path = os.path.join(args.source_dir, fname)
            tgt_path = os.path.join(args.target_dir, fname)
            try:
                renumber_pdb(src_path, tgt_path)
            except:
                logging.info(f"{pdbid} failed...")
     
        
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