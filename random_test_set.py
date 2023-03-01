import argparse
import logging
from tracemalloc import start
from turtle import shape
logging.basicConfig(level=logging.INFO)

import os
# import sys
# import time
# import pickle
import torch
import torch.nn.functional as F
# from openfold.utils.seed import seed_everything
from pytorch_lightning.utilities.seed import seed_everything
import shutil
import debugger

def gather_job(pdb_dir):
    pdb_paths = []
    for f_path in os.listdir(pdb_dir):
        if f_path.endswith('.pdb'):
            pdb_path = os.path.join(pdb_dir, f_path)
            pdb_paths.append(pdb_path)
    
    return pdb_paths

def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')

def main(args):
    if args.seed is not None:
        seed_everything(args.seed)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    jobs = gather_job(args.pdb_path)
    logging.info(f'got {len(jobs)} jobs...')
    # Get input 


    top_k = 25
    prob = torch.rand(len(jobs))
    _, indexes = torch.topk(prob, top_k)
    indexes = indexes.tolist()
    output_dir = "/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/toydata/valid_25"
    topk_jobs = [jobs[i] for i in indexes]
    for job in topk_jobs:
        f_path = os.path.basename(job)
        name = f_path[:args.name_length].lower()
        
        print(f"#################")
        print(f">>> treating: {name}")
        
        dst = os.path.join(
            output_dir,
            f"{name}.pdb"
        )
        shutil.copyfile(job, dst)

        # # 2nd option
        # shutil.copy(src, dst) 

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pdb_path", type=str,
    )    
    parser.add_argument(
        "--resume_from_ckpt_forward", type=str, default=None,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--resume_from_ckpt_backward", type=str, default=None,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--prediction_without_groundtruth", type=bool_type, default=False,
        help="evaluation or prediction"
    )
    parser.add_argument(
        "--version", type=str, default=None,
    )
    parser.add_argument(
        "--is_antibody", type=bool_type, default=False,
        help="training on antibody or not"
    )
    parser.add_argument(
        "--name_length", type=int, default=4,
        help="how many characters are used to name the protein"
    )
    parser.add_argument(
        "--relax", type=bool_type, default=True,
        help="Whether to perform the relaxation"
    )
    parser.add_argument(
        "--ema", type=bool_type, default=True,
        help="Whether to use ema model parameters"
    )
    parser.add_argument(
        "--no_recycling_iters", type=int, default=3,
        help="number of recycling iterations"
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )
    parser.add_argument(
        "--model_device", type=str, default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--config_preset", type=str, default=None,
        help=(
            "Config setting. Choose e.g. 'initial_training', 'finetuning', "
            "'model_1', etc. By default, the actual values in the config are "
            "used."
        )
    )
    parser.add_argument(
        "--yaml_config_preset", type=str, default=None,
        help=(
            "A path to a yaml file that contains the updated config setting. "
            "If it is set, the config_preset will be overwrriten as the basename "
            "of the yaml_config_preset."
        )
    )
    parser.add_argument(
        "--cpus", type=int, default=10,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    if args.model_device == "cpu" and torch.cuda.is_available():
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    if args.config_preset is None and args.yaml_config_preset is None:
        raise ValueError(
            "Either --config_preset or --yaml_config_preset should be specified."
        )

    if args.yaml_config_preset is not None:
        if not os.path.exists(args.yaml_config_preset):
            raise FileNotFoundError(
                f"{os.path.abspath(args.yaml_config_preset)}")
        args.config_preset = os.path.splitext(
            os.path.basename(args.yaml_config_preset)
        )[0]
        logging.info(
            f"the config_preset is set as {args.config_preset} by yaml_config_preset.")

    main(args)