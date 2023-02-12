import argparse
import logging
from tracemalloc import start
from turtle import shape
logging.basicConfig(level=logging.INFO)

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.feats import atom14_to_atom37
from openfold.utils.seed import seed_everything
from openfold.utils.superimposition import superimpose
from openfold.np import residue_constants, protein
from openfold.model.model import AlphaFold
from openfold.model.model_inv import AlphaFoldInverse
from openfold.data import feature_pipeline, data_pipeline, data_transforms
from openfold.config import model_config
from openfold.utils.validation_metrics import gdt_ts
from openfold.model.esm.esmfold import ESMFold, ESMFoldConfig, constructConfigFromYAML

import debugger

def gather_job(pdb_dir):
    pdb_paths = []
    for f_path in os.listdir(pdb_dir):
        if f_path.endswith('.pdb'):
            pdb_path = os.path.join(pdb_dir, f_path)
            pdb_paths.append(pdb_path)
    
    return pdb_paths

def print_mean_metric(metric):
    mean_value = 0
    for i in metric:
        mean_value = i + mean_value
    mean_value = mean_value / len(metric)
    return mean_value

def save_protein(batch, output, output_dir, name, postfix):
    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=output,
        chain_index=batch["chain_index"],
    )        
    unrelaxed_output_path = os.path.join(
        output_dir,
        f"{name}_{postfix}.pdb"
    )
    with open(unrelaxed_output_path, 'w') as f:
        f.write(protein.to_pdb(unrelaxed_protein))

def compute_perplexity(gt_aatype_one_hot, pred_aatype_dist, loop_index):
    #pred_aatype_dist is a distribution
    #gt_aatype [*, Nres]
    #turn gt_aatype into one-hot
    assert gt_aatype_one_hot.shape == pred_aatype_dist.shape
    ppl = pred_aatype_dist[gt_aatype_one_hot.to(torch.bool)]
    assert ppl.shape == loop_index.shape

    ppl = ppl[loop_index]
    ppl = torch.log(ppl)
    ppl = -1 / len(ppl) * torch.sum(ppl)
    ppl = torch.exp(ppl)
    return ppl.item()

##for each sample, compute Sequence RMSD for different loops region
def calculate_rmsd_ca(p1, p2, mask):
    """
        Compute GDT between two structures.
        (Global Distance Test under specified distance cutoff)
        Args:
            p1:
                [*, N, 3] superimposed predicted (ca) coordinate tensor
            p2:
                [*, N, 3] ground-truth (ca) coordinate tensor
            mask:
                [*, N] residue masks
            cutoffs:
                A tuple of size 4, which contains distance cutoffs.
        Returns:
            A [*] tensor contains the final GDT score.
    """
    n = torch.sum(mask, dim=-1) # [*]
    
    p1 = p1.float()
    p2 = p2.float()
    distance = torch.sum((p1 - p2)**2, dim=-1)    # [*, N]
    rmsd = torch.sqrt(torch.sum(distance * mask, dim=-1)/ (mask.sum()+ 1e-6))
    return rmsd.item()

def calculate_structure_score(gt_coords_masked_ca, pred_coords_masked_ca, residue_mask, all_atom_mask_ca):   
    superimposed_pred, _ = superimpose(
        gt_coords_masked_ca, pred_coords_masked_ca
        ) # [*, N, 3]
    rmsd_ca = calculate_rmsd_ca(
        superimposed_pred, gt_coords_masked_ca, residue_mask,
    )
    gdt_ts_score = gdt_ts(
        superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
    )
    return rmsd_ca, gdt_ts_score

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

    config = model_config(
        name=args.config_preset,
        yaml_config_preset=args.yaml_config_preset,
        train=False,
        low_prec=False,
    )
    
    # Loading forward model'scheckpoint
    model_data = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt", progress=False, map_location="cpu")
    cfg = constructConfigFromYAML(config)
    model_state = model_data["model"]
    f_model = ESMFold(esmfold_config=cfg, using_fair=True)
    f_model.load_state_dict(model_state, strict=False)
    g_model = AlphaFoldInverse(config)
    if args.resume_from_ckpt_backward is not None:
        # Loading backward model'scheckpoint
        sd = torch.load(args.resume_from_ckpt_backward, map_location=torch.device('cpu'))
        logging.info("printing loaded state dict for model")
        stat_dict_g = {k[len("model."):]:v for k,v in sd["state_dict"].items()}
        stat_dict_m2s = {}
        for k,v in stat_dict_g.items():
            if k in ["evoformer.linear.weight", "evoformer.linear.bias"]:
                stat_dict_m2s[k[len("evoformer.linear."):]] = v
        g_model.load_state_dict(stat_dict_g, strict=False)
        g_model.linear_m2s.load_state_dict(stat_dict_m2s)
        g_model = g_model.eval()
        logging.info("Successfully loaded backward model weights...")

    print(f">>> printing forward model:")
    print(f_model)
    print(f">>> printing backward model:")
    print(g_model)

    logging.info(f">>> is using antibody: {args.is_antibody} ...")
    # Prepare data
    data_processor = data_pipeline.DataPipeline(is_antibody=args.is_antibody)
    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    jobs = gather_job(args.pdb_path)
    logging.info(f'got {len(jobs)} jobs...')
    # Get input 
    # metrics = []
    # list_rmsd = []
    # list_gdt_ts = []
    # list_tm = []
    # list_mean_plddt = []
    logging.info(f'predicting with {args.no_recycling_iters} recycling iterations...')
    for job in jobs:
        f_path = os.path.basename(job)
        name = f_path[:args.name_length].lower()

        # process pdb feature
        feature_dict = data_processor.process_pdb(
            pdb_path=job,
        )
        feature_dict["no_recycling_iters"] = args.no_recycling_iters
        batch = feature_processor.process_features(
            feature_dict,
            mode="predict",
        )
        batch = {
            k: torch.as_tensor(v, device=args.model_device).unsqueeze_(0)
            for k, v in batch.items()
        }
        sequence1 = feature_dict["sequence"][0].decode("utf-8") 
        sequence_test = residue_constants.aatype_to_sequence(batch["aatype"][0, ..., -1])
        print(f'>>>sequence is: {sequence1}')
        print(f'>>>sequence test is: {sequence_test}')

        ######## begin inference #########
        ######## model1 #########
        with torch.no_grad():
            output1 = f_model.infer_bb(sequence1, num_recycles=3, cpu_only=True)
        bb_coords_1 = output1['bb_coords']
        all_coords_1 = output1['final_atom_positions']
        print(f">>> output of infer_bb is {bb_coords_1.shape}")

        ######## model2 #########
        with torch.no_grad():
            output2 = g_model.forward_h(batch, bb_coords_1)
        output2 = tensor_tree_map(lambda x: x[0, ...].cpu(), output2)
        final_pred_aatype_dist = output2["sm"]["seqs_logits"][-1]
        final_pred_aatype_dist[..., -1] = -9999 # zero out UNK.
        sampled_seqs = final_pred_aatype_dist.argmax(dim=-1)    # greedy sampling
        sequence2 = residue_constants.aatype_to_sequence(sampled_seqs)

        ######## model3 #########
        with torch.no_grad():
            output3 = f_model.infer_bb(sequence2, num_recycles=3, cpu_only=True)
        bb_coords_3 = output3['bb_coords']
        all_coords_3 = output3['final_atom_positions']

        ######## model4 #########
        # inverse folding from native bb
        with torch.no_grad():
            output4 = g_model.forward(batch)
        output4 = tensor_tree_map(lambda x: x[0, ...].cpu(), output4)
        final_pred_aatype_dist2 = output4["sm"]["seqs_logits"][-1]
        final_pred_aatype_dist2[..., -1] = -9999 # zero out UNK.
        sampled_seqs2 = final_pred_aatype_dist2.argmax(dim=-1)    # greedy sampling

        
        ## plddt
        ## temperature sampling
    
        ###### evaluation #######
        batch = tensor_tree_map(lambda x: x[0, ..., -1].cpu(), batch)
        gt_aatype = batch["aatype"]
        ce = F.cross_entropy(final_pred_aatype_dist, gt_aatype)
        ppl = ce.exp()
        aars = sampled_seqs.eq(gt_aatype).float().mean()
        print(f">>> ppl from predcited bb is {ppl}")
        print(f">>> aars from predcited bb is {aars}")
        print(f">>> sampled_seqs from predcited bb is: {sampled_seqs}")

        ce2 = F.cross_entropy(final_pred_aatype_dist2, gt_aatype)
        ppl2 = ce2.exp()
        aars2 = sampled_seqs2.eq(gt_aatype).float().mean()
        print(f">>> ppl from native bb is {ppl2}")
        print(f">>> aars from native bb is {aars2}")
        print(f">>> sampled_seqs from native bb is: {sampled_seqs2}")

        if not args.prediction_without_groundtruth:
            # align coordinates and compute rmsd_ca
            gt_coords = batch["all_atom_positions"] # [*, N, 37, 3]
            all_atom_mask = batch["all_atom_mask"] # [*, N, 37]
            gt_coords_masked = gt_coords * all_atom_mask[..., None] # [*, N, 37, 3]
            residue_mask = torch.sum(all_atom_mask, dim=-1)
            residue_mask = torch.where(residue_mask>0,1,0)
            ca_pos = residue_constants.atom_order["CA"]
            gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :] # [*, N, 3]
            all_atom_mask_ca = all_atom_mask[..., ca_pos] # [*, N]
            
            pred_coords1 = all_coords_1 # [*, N, 37, 3]
            pred_coords_masked1 = pred_coords1 * all_atom_mask[..., None] # [*, N, 37, 3]
            pred_coords_masked_ca1 = pred_coords_masked1[..., ca_pos, :] # [*, N, 3]
            rmsd_ca1, gdt_ts_score1 = calculate_structure_score(gt_coords_masked_ca, pred_coords_masked_ca1, residue_mask, all_atom_mask_ca)
            # logging.info(f">>> residue mask is {residue_mask}")
            print(f">>> rmsd_ca of direct prediction is {rmsd_ca1}")
            print(f">>> gdt_ts of direct prediction is {gdt_ts_score1}")

            pred_coords3 = all_coords_3 # [*, N, 37, 3]
            pred_coords_masked3 = pred_coords3 * all_atom_mask[..., None] # [*, N, 37, 3]
            pred_coords_masked_ca3 = pred_coords_masked3[..., ca_pos, :] # [*, N, 3]
            rmsd_ca3, gdt_ts_score3 = calculate_structure_score(gt_coords_masked_ca, pred_coords_masked_ca3, residue_mask, all_atom_mask_ca)
            # logging.info(f">>> residue mask is {residue_mask}")
            print(f">>> rmsd_ca of indirect prediction is {rmsd_ca3}")
            print(f">>> gdt_ts of indirect prediction is {gdt_ts_score3}")

            ## protein object saving & relaxation
            output1 = tensor_tree_map(lambda x: np.array(x[0, ...]), output1)
            output3 = tensor_tree_map(lambda x: np.array(x[0, ...]), output3)
            batch = tensor_tree_map(lambda x: np.array(x), batch)
            save_protein(batch, output1, output_dir, name, "direct_structure_prediction")
            save_protein(batch, output3, output_dir, name, "indirect_structure_prediction")

    

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