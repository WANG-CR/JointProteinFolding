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
from openfold.data import feature_pipeline, data_pipeline, data_transforms
from openfold.config import model_config

import debugger

def gather_job(pdb_dir):
    pdb_paths = []
    for f_path in os.listdir(pdb_dir):
        if f_path.endswith('.pdb'):
            pdb_path = os.path.join(pdb_dir, f_path)
            pdb_paths.append(pdb_path)
    
    return pdb_paths

def main(args):
    if args.seed is not None:
        seed_everything(args.seed)

    config = model_config(
        name=args.config_preset,
        yaml_config_preset=args.yaml_config_preset,
        train=False,
        low_prec=False,
    )
    model = AlphaFold(config)
    model = model.eval()

    # Load the checkpoint
    latest_path = os.path.join(args.resume_from_ckpt, 'latest')
    if os.path.isfile(latest_path):
        with open(latest_path, 'r') as fd:
            tag_ = fd.read().strip()
    else:
        raise ValueError(f"Unable to find 'latest' file at {latest_path}")
    ckpt_path = os.path.join(args.resume_from_ckpt,
                             tag_, "mp_rank_00_model_states.pt")
    ckpt_epoch = os.path.basename(args.resume_from_ckpt).split('-')[0]
    if args.ema:
        state_dict = torch.load(ckpt_path, map_location="cpu")["ema"]["params"]
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")["module"]
        state_dict = {k[len("module.model."):]: v for k,
                      v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.model_device)
    logging.info(f"Successfully loaded model weights from {ckpt_path}...")

    # Prepare data

    data_processor = data_pipeline.DataPipeline(is_antibody=args.is_antibody)
    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    output_dir = args.output_dir
    predict_dir = os.path.join(output_dir, 'pdb')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)

    jobs = gather_job(args.pdb_path)
    logging.info(f'got {len(jobs)} jobs...')
    # Get input 
    metrics = []
    for job in jobs:
        f_path = os.path.basename(job)
        name = f_path[:4].lower()

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
                k: torch.as_tensor(v, device=args.model_device)
                for k, v in batch.items()
            }
            t = time.perf_counter()
            out = model(batch)
            logging.info(f"Inference time: {time.perf_counter() - t}")

        # Toss out the recycling dimensions --- we don't need them anymore
        # batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)
        batch = tensor_tree_map(lambda x: x[..., -1].cpu(), batch)

        # handle the discrepancy caused by predicted aatype.
        if "final_aatype" in out:
            fake_batch = {"aatype": out["final_aatype"]}
            fake_batch = data_transforms.make_atom14_masks(fake_batch)
            out["final_atom_positions"] = atom14_to_atom37(
                out["sm"]["positions"][-1], fake_batch
            )
            out["residx_atom14_to_atom37"] = fake_batch["residx_atom14_to_atom37"]
            out["residx_atom37_to_atom14"] = fake_batch["residx_atom37_to_atom14"]
            out["atom14_atom_exists"] = fake_batch["atom14_atom_exists"]
            out["atom37_atom_exists"] = fake_batch["atom37_atom_exists"]
            out["final_atom_mask"] = fake_batch["atom37_atom_exists"]
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

        plddt = out["plddt"] # [*, N]
        # mean_plddt = np.mean(plddt) # [*, ]
        # [*, N, 37]
        plddt_b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )


        # saving whole metric file
        metric = {}
        metric["name"] = name

        final_pred_aatype_dist = out["final_aatype_dist"]
        final_pred_aatype_dist = torch.from_numpy(final_pred_aatype_dist)
        final_pred_aatype = torch.argmax(final_pred_aatype_dist, dim=-1)
        gt_aatype = batch["aatype"]
        gt_aatype_one_hot = data_transforms.make_one_hot(gt_aatype, 21)
        loop_index = batch["loop_index"]

        start = time.time()
        #sampling 10K samples
        num_samples = 10000
        num_k = 100
        #using multinomial to sample
        # print(f"size of final_pred_aatype_dist is {final_pred_aatype_dist.shape}")

        aatype_sample10000 = torch.multinomial(final_pred_aatype_dist, num_samples, replacement=True).permute(1,0)
        # print(f"size of aatype_sample100 is {aatype_sample10000.shape}")
        # top k sampling
        aatype_sample_onehot = data_transforms.make_one_hot(aatype_sample10000, 21)
        # print(f"size of augmented final_pred_aatype_dist is {final_pred_aatype_dist[None, ...].repeat(num_samples,1,1).shape}")
        likelihood = (final_pred_aatype_dist[None, ...].repeat(num_samples,1,1))[aatype_sample_onehot.to(torch.bool)].view(num_samples, -1)
        log_likelihood = torch.log(likelihood)
        log_likelihood = torch.sum(log_likelihood, -1)
        # print(f"size of loglikelihood is {log_likelihood.shape}")
        _, indice = torch.sort(log_likelihood,descending=True)
        aatype_sample100 = aatype_sample10000[indice[:num_k], :]
        
        # print(f"size of likelihood is {likelihood.shape}")
        # print(f"time used is {time.time()-start}")
        loop_index100 = loop_index.expand(num_k,-1)
        # print(f"size of loop_index 100 is {loop_index100.shape}")
        for i in range(1,7):
            metric["sampleAAR"+str(i)] = ((gt_aatype[loop_index==i].repeat(num_k) == aatype_sample100[loop_index100==i]).sum() / len(gt_aatype[loop_index==i]) / num_k).item()
        logging.info(f"loop H3 sample aar of {name}: {metric['sampleAAR3']}")
        #for each sample, calculate AAR




        # loop_pred = []
        # loop_gt = []
        # aar = []
        for i in range(1,7):
            # loop_pred.append(final_pred_aatype[loop_index==i])
            # loop_gt.append(gt_aatype[loop_index==i])
            # aar.append((loop_pred[i-1] == loop_gt[i-1]).sum() / len(loop_gt[i-1]))
            metric["aar"+str(i)] = ((final_pred_aatype[loop_index==i] == gt_aatype[loop_index==i]).sum() / len(gt_aatype[loop_index==i])).item()
        # metric["aar"] = aar
        # logging.info(f"loop H3 aar of {name}: {aar[2]}")
        logging.info(f"loop H3 aar of {name}: {metric['aar3']}")
        unrelaxed_protein = protein.from_prediction(
            features=batch,
            result=out,
            b_factors=plddt_b_factors,
            chain_index=batch["chain_index"],
        )

        ##for each sample, compute perplexity
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

        for i in range(1,7):
            metric["ppl"+str(i)] = compute_perplexity(gt_aatype_one_hot,final_pred_aatype_dist,loop_index==i)
            # metric["ppl"+str(i)] = compute_perplexity(gt_aatype[loop_index==i],final_pred_aatype_dist[loop_index_broadcast==i])
            # logging.info(f"loop {i} ppl of {name}: {metric['ppl'+str(i)]}")
        


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
       
       
        gt_coords = batch["all_atom_positions"] # [*, N, 37, 3]
        pred_coords = out["final_atom_positions"] # [*, N, 37, 3]
        all_atom_mask = batch["all_atom_mask"] # [*, N, 37]
        # This is super janky for superimposition. Fix later

        pred_coords = torch.from_numpy(pred_coords)

        gt_coords_masked = gt_coords * all_atom_mask[..., None] # [*, N, 37, 3]
        pred_coords_masked = pred_coords * all_atom_mask[..., None] # [*, N, 37, 3]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :] # [*, N, 3]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :] # [*, N, 3]
        all_atom_mask_ca = all_atom_mask[..., ca_pos] # [*, N]
        
        superimposed_pred, _ = superimpose(
            gt_coords_masked_ca, pred_coords_masked_ca
            ) # [*, N, 3]
        
        for i in range(1,7):
            metric["rmsd"+str(i)] = calculate_rmsd_ca(
                superimposed_pred, gt_coords_masked_ca, loop_index==i
            )

        metrics.append(metric)
        
        # unrelaxed_protein = protein.from_prediction(
        #     features=batch,
        #     result=out,
        #     b_factors=plddt_b_factors,
        #     chain_index=batch["chain_index"],
        # )

        # # Save the unrelaxed PDB.
        # unrelaxed_output_path = os.path.join(
        #     predict_dir,
        #     f"{name}_{ckpt_epoch}_{args.config_preset}_rec{args.no_recycling_iters}_s{args.seed}_unrelaxed.pdb"
        # )
        # with open(unrelaxed_output_path, 'w') as f:
        #     f.write(protein.to_pdb(unrelaxed_protein))

        # if args.relax:
        #     if "relax" not in sys.modules:
        #         import openfold.np.relax.relax as relax

        #     logging.info("start relaxation")
        #     amber_relaxer = relax.AmberRelaxation(
        #         use_gpu=(args.model_device != "cpu"),
        #         **config.relax,
        #     )
        #     try:
        #         # Relax the prediction.
        #         t = time.perf_counter()
        #         visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        #         if("cuda" in args.model_device):
        #             device_no = args.model_device.split(":")[-1]
        #             os.environ["CUDA_VISIBLE_DEVICES"] = device_no
        #         relaxed_pdb_str, _, _ = amber_relaxer.process(
        #             prot=unrelaxed_protein)
        #         os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        #         logging.info(f"Relaxation time: {time.perf_counter() - t}")

        #         # Save the relaxed PDB.
        #         relaxed_output_path = os.path.join(
        #             predict_dir,
        #             f"{name}_{ckpt_epoch}_{args.config_preset}_rec{args.no_recycling_iters}_s{args.seed}_relaxed.pdb"
        #         )
        #         with open(relaxed_output_path, 'w') as f:
        #             f.write(relaxed_pdb_str)
        #     except Exception as e:
        #         logging.warning(e)
        #         logging.warning("relaxation failed...")

    sum_aar = 0
    for metric in metrics:
        sum_aar += metric["aar3"]
    avg_aar = sum_aar / len(metrics)
    logging.info(f"avg aar3s : {avg_aar}")

    sum_ppl = 0
    for metric in metrics:
        sum_ppl += metric["ppl3"]
    avg_ppl = sum_ppl / len(metrics)
    logging.info(f"avg ppl3s : {avg_ppl}") 

    print(f"{args.version}")
    pd.DataFrame(metrics).to_csv(args.version + 'top_100_sample.csv')
    
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
        "pdb_path", type=str,
    )
    parser.add_argument(
        "resume_from_ckpt", type=str,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--version", type=str, default=None,
    )
    parser.add_argument(
        "--is_antibody", type=bool, default=None,
        help="training on antibody or not"
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