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
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_dir = args.pdb_path
    for f_path in os.listdir(input_dir):
        input_path = os.path.join(input_dir, f_path)
        output_path = os.path.join(output_dir, f_path+".pdb")
        shutil.copyfile(input_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb_path", type=str,
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )

    args = parser.parse_args()
    main(args)