
from openfold.model.esm.esmfold import ESMFold, ESMFoldConfig, constructConfigFromYAML
from openfold.model.esm.trunk import StructureModuleConfig, FoldingTrunkConfig
import torch
import numpy as np
import os
from openfold.config import model_config
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.np import residue_constants, protein
from openfold.data import parsers
from tqdm import tqdm
import time
import argparse

def save_esm_protein(output, output_dir, name, postfix):
    b_factors = np.zeros_like(output["final_atom_mask"])
    unrelaxed_protein = protein.from_esm_prediction(
        result=output,
        b_factors=b_factors
    )        
    unrelaxed_output_path = os.path.join(
        output_dir,
        f"{name}_{postfix}.pdb"
    )
    with open(unrelaxed_output_path, 'w') as f:
        f.write(protein.to_pdb(unrelaxed_protein))

def gather_job(fasta_dir):
    fasta_paths = []
    for f_path in os.listdir(fasta_dir):
        if f_path.endswith('.fasta'):
            fasta_path = os.path.join(fasta_dir, f_path)
            fasta_paths.append(fasta_path)
    return fasta_paths

def main(args):
    if args.model=="esm2_v1":
        model_data = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt", progress=False, map_location="cpu")
    elif args.model=="esm2_v0":
        model_data = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v0.pt", progress=False, map_location="cpu")
    
    config = model_config(
        name='3B+48L',
        yaml_config_preset='/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/yaml_config/joint_finetune/esm3B+8inv_samlldim.yml',
        train=False,
        low_prec=False,
    )
    cfg = constructConfigFromYAML(config)

    # cfg = ESMFoldConfig()
    model_state = model_data["model"]
    model = ESMFold(esmfold_config=cfg, using_fair=True)
    model.load_state_dict(model_state, strict=False)
    print(f"load succesfully")

    chunk_size = args.chunk_size
    model.set_chunk_size(chunk_size)

    fasta_dir = args.fasta_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    jobs = gather_job(fasta_dir)
    index = 0
    skiplist = {}
    for job in tqdm(jobs):
        f_path = os.path.basename(job)
        name = f_path.split(".")[0]
        with open(job, 'r') as f:
            fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(fasta_str)
        sequence = input_seqs[0]
        print(f">>>>>>>>>job {index}>>>>>>>>>>")
        print(f"treating protein: {name}")
        print(f"sequence length: {len(sequence)}")
        
        if len(sequence)>args.max_length:
            skiplist[name] = sequence
            continue

        index += 1
        start = time.time()
        with torch.no_grad():
            output = model.infer_pdb(sequence, num_recycles=3, cpu_only=True)
        end = time.time()
        print(f"time elapsed: {end - start}")

        output_path = os.path.join(output_dir, name+".pdb")
        with open(output_path, "w") as f:
            f.write(output)

    print(f"we succesfully predicted {len(skiplist)} proteins, which are shorter than {args.max_length}")


    # print(skiplist)

    # for key, sequence in tqdm(skiplist.items()):
    #     print(f">>>>>>>>>job {index}>>>>>>>>>>")
    #     print(f"treating protein: {name}")
    #     print(f"sequence length: {len(sequence)}")
    #     print(f"time elapsed: {end - start}")
    #     index += 1
    #     start = time.time()
    #     with torch.no_grad():
    #         output = model.infer_pdb(sequence, num_recycles=3, cpu_only=True)
    #     end = time.time()
    #     output_path = os.path.join(output_dir, key+".pdb")
    #     with open(output_path, "w") as f:
    #         f.write(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta_dir", type=str, default=None,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )
    parser.add_argument(
        "--model", type=str, default="esm2_v1",
        help="Name of the directory in which to output the prediction",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=512,
        help="chunk size",
    )
    parser.add_argument(
        "--max_length", type=int, default=1024,
        help="max_length",
    )
    

    args = parser.parse_args()
    main(args)