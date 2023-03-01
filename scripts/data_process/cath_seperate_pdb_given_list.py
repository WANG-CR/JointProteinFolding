import argparse
import logging
import os
# import sys
# import time
# import pickle
import shutil
import json
import re
logging.basicConfig(level=logging.INFO)


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

def main(args):
    with open(args.list_path, "rb") as f:
        data = json.load(f)
    
    train = data["train"]
    test = data["test"]
    valid = data["validation"]

    train = [re.sub('\.', '', name) for name in train]
    test = [re.sub('\.', '', name) for name in test]
    valid = [re.sub('\.', '', name) for name in valid]

    print(test[0:5])

    output_dir = args.output_dir
    train_dir = os.path.join(output_dir, "train")
    valid_dir = os.path.join(output_dir, "valid")
    test_dir = os.path.join(output_dir, "test")

    for directory in [train_dir, valid_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    input_dir = args.pdb_path
    for f_path in os.listdir(input_dir):
        if f_path.endswith('.pdb'):
            pdb_path = os.path.join(input_dir, f_path)
            if f_path[0:5] in train:
                pdb_dir = os.path.join(train_dir, f_path)
            elif f_path[0:5] in valid:
                pdb_dir = os.path.join(valid_dir, f_path)
            elif f_path[0:5] in test:
                pdb_dir = os.path.join(test_dir, f_path)
            else:
                pdb_dir = None
            
            if pdb_dir is not None:
                shutil.copyfile(pdb_path, pdb_dir)


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb_path", type=str,
    )    
    parser.add_argument(
        "--list_path", type=str, default=None,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--name_length", type=int, default=5,
        help="how many characters are used to name the protein"
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )

    args = parser.parse_args()
    main(args)