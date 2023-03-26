import numpy as np
import torch
from tmtools import tm_align
from openfold.np import protein, residue_constants
from openfold.utils.loss import lddt_ca, compute_drmsd, compute_drmsd_np
from openfold.utils.superimposition import superimpose
from openfold.utils.validation_metrics import gdt_ts, gdt_ha
import os
import argparse
import logging
from tqdm import tqdm
from statistics import mean

def tm_score_from_pdbs(fp1, fp2):
    coords1, seq1 = coord_seq_from_file(fp1)
    coords2, seq2 = coord_seq_from_file(fp2)
    res = tm_align(coords1, coords2, seq1, seq2)
    return res.tm_norm_chain1

def drmsd_score_from_pdbs(fp1, fp2):
    coords_ca1, mask_ca = coord_ca_from_file(fp1)
    coords_ca2, _ = coord_ca_from_file(fp2)
    drmsd_ca_score = compute_drmsd_np(
        coords_ca2,
        coords_ca1,
        mask=mask_ca,
    ) # [*]

    coords_ca1 = torch.from_numpy(coords_ca1)
    coords_ca2 = torch.from_numpy(coords_ca2)
    mask_ca = torch.from_numpy(mask_ca)
    superimposed_pred, _ = superimpose(
        coords_ca1, coords_ca2
    ) # [*, N, 3]
    gdt_ts_score = gdt_ts(
        superimposed_pred, coords_ca1, mask_ca
    )
    gdt_ha_score = gdt_ha(
        superimposed_pred, coords_ca1, mask_ca
    )

    return drmsd_ca_score.item(), gdt_ts_score.item(), gdt_ha_score.item()

def coord_ca_from_file(path):
    with open(path, 'r') as f:
        pdb_str = f.read()
    protein_object = protein.from_pdb_string(pdb_str)
    all_coords = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask.astype(float)

    all_coords_masked = all_coords * all_atom_mask[..., None]
    ca_pos = residue_constants.atom_order["CA"]
    coords_masked_ca = all_coords_masked[..., ca_pos, :]
    all_atom_mask_ca = all_atom_mask[..., ca_pos]
    return coords_masked_ca, all_atom_mask_ca


def coord_seq_from_file(path):
    with open(path, 'r') as f:
        pdb_str = f.read()
    protein_object = protein.from_pdb_string(pdb_str)
    ca_coords = protein_object.atom_positions[:, 1, :]
    seq = _aatype_to_str_sequence(protein_object.aatype)
    return ca_coords, seq

def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]] 
        for i in range(len(aatype))
    ])

def gather_job(gt_dir, predict_dir):
    pdb_paths = []
    for f_path in os.listdir(gt_dir):
        if f_path.endswith('.pdb'):
            pdb_path_to_verify = os.path.join(predict_dir, f_path)
            if os.path.exists(pdb_path_to_verify):
                pdb_paths.append(f_path)
    return pdb_paths

def TMalign_dir(path1, path2):
    jobs = gather_job(path1, path2)    
    logging.info(f'got {len(jobs)} jobs...')

    drmsd_list = []
    metric_dict = {}
    gdt_ts_list = []
    gdt_ha_list = []
    
    # lddt_ca_score = lddt_ca(
    #     pred_coords,
    #     gt_coords,
    #     all_atom_mask,
    #     eps=self.config.globals.eps,
    #     per_residue=False,
    # ) # [*]

    for job in tqdm(jobs):
        fp1 = os.path.join(path1, job)
        fp2 = os.path.join(path2, job)
        name = job.split(".")[0]
        drmsd, gdt_ts, gdt_ha = drmsd_score_from_pdbs(fp1, fp2)
        drmsd_list.append(drmsd)
        gdt_ts_list.append(gdt_ts)
        gdt_ha_list.append(gdt_ha)
        # print(drmsd_list)
        # metric_dict[name] = drmsd
    return {
        "mean_drmsd": mean(drmsd_list),
        "mean_gdt_ts": mean(gdt_ts_list),
        "mean_gdt_ha": mean(gdt_ha_list),
        "len_job": len(jobs),
    }
    # drmsd_list, metric_dict, mean(drmsd_list), mean(gdt_ts_list), mean(gdt_ha_list), len(jobs)

def main(args):
    dictionary = TMalign_dir(args.gt_dir, args.predict_dir)
    with open(args.log_dir, 'w') as f:
        f.write(str(dictionary["len_job"]) + " jobs: " + "\n")
        f.write("mean_drmsd: " + str(dictionary["mean_drmsd"]) + "\n")
        f.write("mean_gdt_ts: " + str(dictionary["mean_gdt_ts"]) + "\n")
        f.write("mean_gdt_ha: " + str(dictionary["mean_gdt_ha"]) + "\n")
    # print(mean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_dir", type=str, default=None,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--predict_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )
    parser.add_argument(
        "--log_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )
    args = parser.parse_args()
    main(args)


    # python verify_drmsd.py --gt_dir /home/chuanrui/scratch/database/structure_datasets/CASP14_esm/pdb --predict_dir /home/chuanrui/scratch/database/structure_datasets/CASP14_esm/predict_esm_v0 --log_dir debug_drmsd.txt