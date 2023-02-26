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
import torch.nn.functional as F

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
    g_model = AlphaFoldInverse(config)

    if args.resume_from_ckpt_backward is not None:
        # Loading backward model'scheckpoint
        if not args.is_joint_ckpt:
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
        elif args.is_joint_ckpt:
            sd = torch.load(args.resume_from_ckpt_backward, map_location=torch.device('cpu'))
            logging.info("printing loaded state dict for model")
            stat_dict_g2 = {}
            for k, v in sd["state_dict"].items():
                if "g_model." in k:
                    stat_dict_g2[k] = v
            # stat_dict_g = {k[len("g_model."):]:v for k,v in sd["state_dict"].items()}
            stat_dict_g = {k[len("g_model."):]:v for k,v in stat_dict_g2.items()}
            # stat_dict_m2s = {}
            # for k,v in stat_dict_g.items():
            #     if k in ["evoformer.linear.weight", "evoformer.linear.bias"]:
            #         stat_dict_m2s[k[len("evoformer.linear."):]] = v
            g_model.load_state_dict(stat_dict_g)
            # g_model.linear_m2s.load_state_dict(stat_dict_m2s)
            g_model = g_model.eval()
            g_model.requires_grad_(False)
            logging.info("Successfully loaded backward model weights...")
    
    print(f">>> printing backward model:")
    print(g_model)

    logging.info(f">>> is using antibody: {args.is_antibody} ...")
    # Prepare data
    data_processor = data_pipeline.DataPipeline(is_antibody=args.is_antibody)
    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    output_dir = args.output_dir
    # predict_dir = os.path.join(output_dir, 'pdb')
    predict_unrelaxed_dir = os.path.join(output_dir, f'{args.config_preset}_rec{args.no_recycling_iters}_s{args.seed}', 'unrelaxed_pdb')
    predict_relaxed_dir = os.path.join(output_dir, f'{args.config_preset}_rec{args.no_recycling_iters}_s{args.seed}', 'relaxed_pdb')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # if not os.path.exists(predict_dir):
    #     os.makedirs(predict_dir)
    if not os.path.exists(predict_unrelaxed_dir):
        os.makedirs(predict_unrelaxed_dir)
    if not os.path.exists(predict_relaxed_dir):
        os.makedirs(predict_relaxed_dir)

    jobs = gather_job(args.pdb_path)
    logging.info(f'got {len(jobs)} jobs...')
    # Get input 
    # metrics = []
    # list_rmsd = []
    # list_gdt_ts = []
    # list_tm = []
    # list_mean_plddt = []
    list_aar = []
    list_ppl = []
    num_treated = 0
    logging.info(f'predicting with {args.no_recycling_iters} recycling iterations...')
    for job in jobs:
        num_treated += 1
        logging.info(f">>> num_treated: {num_treated}") 
        f_path = os.path.basename(job)
        name = f_path[:args.name_length].lower()

        feature_dict = data_processor.process_pdb(
            pdb_path=job,
        )
        feature_dict["no_recycling_iters"] = args.no_recycling_iters
        processed_feature_dict = feature_processor.process_features(
            feature_dict,
            mode="predict",
        )

        logging.info("Executing model...")
        batch = processed_feature_dict
        with torch.no_grad():
            batch = {
                k: torch.as_tensor(v, device=args.model_device).unsqueeze_(0)
                for k, v in batch.items()
            }
            out = g_model(batch)


        # Toss out the recycling dimensions --- we don't need them anymore
        # batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)
        batch = tensor_tree_map(lambda x: x[0, ..., -1].cpu(), batch)
        out = tensor_tree_map(lambda x: np.array(x[0, ...].cpu()), out)
        final_pred_aatype_dist = out["sm"]["seqs_logits"][-1]
        final_pred_aatype_dist = torch.from_numpy(final_pred_aatype_dist)
        gt_aatype = batch["aatype"]
        # masked_pred = logits.masked_select(batch["seq_mask"].unsqueeze(-1).to(torch.bool)).view(-1, residue_constants.restype_num+1) # Nl x 21
        # masked_target = aatype.masked_select(batch["seq_mask"].to(torch.bool)).view(-1)    # Nl
        ce = F.cross_entropy(final_pred_aatype_dist, gt_aatype)
        ppl = ce.exp()
        list_ppl.append(ppl)
        #
        final_pred_aatype_dist[..., -1] = -9999 # zero out UNK.
        sampled_seqs = final_pred_aatype_dist.argmax(dim=-1)    # greedy sampling
        # masked_sampled_seqs = sampled_seqs.masked_select(batch["seq_mask"].to(torch.bool)).view(-1) # N x Nl
        aars = sampled_seqs.eq(gt_aatype).float().mean()
        print(f">>> aars is {aars}")
        list_aar.append(aars)

        # multinomial sampling
        # final_pred_aatype = torch.argmax(final_pred_aatype_dist, dim=-1)    
        # gt_aatype_one_hot = data_transforms.make_one_hot(gt_aatype, 21)
        # loop_index = batch["loop_index"]
        # num_samples = 10000
        # num_k = 100
        # aatype_sample10000 = torch.multinomial(final_pred_aatype_dist, num_samples, replacement=True).permute(1,0)
        # aatype_sample_onehot = data_transforms.make_one_hot(aatype_sample10000, 21)
        # likelihood = (final_pred_aatype_dist[None, ...].repeat(num_samples,1,1))[aatype_sample_onehot.to(torch.bool)].view(num_samples, -1)
        # log_likelihood = torch.log(likelihood)
        # log_likelihood = torch.sum(log_likelihood, -1)
        # _, indice = torch.sort(log_likelihood,descending=True)
        # aatype_sample100 = aatype_sample10000[indice[:num_k], :]
        # loop_index100 = loop_index.expand(num_k,-1)
        # # calculate AAR of sampled sequence
        # for i in range(1,7):
        #     metric["sampleAAR"+str(i)] = ((gt_aatype[loop_index==i].repeat(num_k) == aatype_sample100[loop_index100==i]).sum() / len(gt_aatype[loop_index==i]) / num_k).item()
        # logging.info(f"loop H3 sample aar of {name}: {metric['sampleAAR3']}")
            
        # # calculate AAR of argmax predicted sequence
        # for i in range(1,7):
        #     metric["aar"+str(i)] = ((final_pred_aatype[loop_index==i] == gt_aatype[loop_index==i]).sum() / len(gt_aatype[loop_index==i])).item()
        # logging.info(f"loop H3 aar of {name}: {metric['aar3']}")

        # # for predicted aatype distribution, compute perplexity
        # for i in range(1,7):
        #     metric["ppl"+str(i)] = compute_perplexity(gt_aatype_one_hot,final_pred_aatype_dist,loop_index==i)

        # # align coordinates and compute rmsd_ca
        # gt_coords = batch["all_atom_positions"] # [*, N, 37, 3]
        # pred_coords = out["final_atom_positions"] # [*, N, 37, 3]
        # all_atom_mask = batch["all_atom_mask"] # [*, N, 37]
        # # This is super janky for superimposition. Fix later
        # pred_coords = torch.from_numpy(pred_coords)
        # gt_coords_masked = gt_coords * all_atom_mask[..., None] # [*, N, 37, 3]
        # pred_coords_masked = pred_coords * all_atom_mask[..., None] # [*, N, 37, 3]
        # ca_pos = residue_constants.atom_order["CA"]
        # gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :] # [*, N, 3]
        # pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :] # [*, N, 3]
        # all_atom_mask_ca = all_atom_mask[..., ca_pos] # [*, N]
        # superimposed_pred, _ = superimpose(
        #     gt_coords_masked_ca, pred_coords_masked_ca
        #     ) # [*, N, 3]
        
        # for i in range(1,7):
        #     metric["rmsd"+str(i)] = calculate_rmsd_ca(
        #         superimposed_pred, gt_coords_masked_ca, loop_index==i
        #     )

        # metrics.append(metric)


            

    logging.info(f">>>>>> final mean ppl is {print_mean_metric(list_ppl)}")
    logging.info(f">>>>>> final mean aar is {print_mean_metric(list_aar)}")
    # logging.info(f">>>>>> final mean gdt_ts is {print_mean_metric(list_gdt_ts)}")
    # logging.info(f">>>>>> final mean tm is {print_mean_metric(list_tm)}")


    # after predicting all samples, compute the metric's statistic

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pdb_path", type=str,
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
        "--is_joint_ckpt", type=bool_type, default=False,
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