import pyrosetta
import argparse
import os
import numpy as np
import time
import requests
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)


def get_ab_metrics(pose_1, pose_2):
    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_1)
    pose_i2 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_2)

    results = pyrosetta.rosetta.protocols.antibody.cdr_backbone_rmsds(
        pose_1, pose_2, pose_i1, pose_i2)

    results_labels = [
        'ocd', 'frh_rms', 'h1_rms', 'h2_rms', 'h3_rms', 'frl_rms', 'l1_rms',
        'l2_rms', 'l3_rms'
    ]
    results_dict = {}
    for i in range(9):
        results_dict[results_labels[i]] = results[i + 1]

    return results_dict


def get_pair_results(pred_pdb, native_pdb):
    pose = pyrosetta.pose_from_pdb(pred_pdb)
    native_pose = pyrosetta.pose_from_pdb(native_pdb)

    metrics = get_ab_metrics(pose, native_pose)
    return metrics


def main(args):

    init_string = "-mute all -check_cdr_chainbreaks false -detect_disulf true"
    pyrosetta.init(init_string)

    test_list = []
    metrics = []
    bad_pdbs = []
    results_labels = [
        'ocd', 'frh_rms', 'h1_rms', 'h2_rms', 'h3_rms', 'frl_rms', 'l1_rms',
        'l2_rms', 'l3_rms'
    ]
    results_dict = {}
    for label in results_labels:
        results_dict[label] = []

    jobs = [x for x in os.listdir(args.pred_dir) if x.endswith('.pdb')]
    for fname in tqdm(jobs):
        pdbid = fname[:4]
        test_list.append(pdbid)
        pred_pdb_path = os.path.join(args.pred_dir, fname)
        native_pdb_path = os.path.join(args.target_dir, pdbid + '.pdb')
        # logging.info(f'calculating {pdbid}...')
        try:
            metrics.append(get_pair_results(pred_pdb_path, native_pdb_path))
        except Exception as exception:
            bad_pdbs.append(pdbid)
            logging.warning(f"{pdbid} failed: {exception}")

    logging.info(f"got {len(test_list)} pdbs to compare.")
    logging.info(test_list)
    for res in metrics:
        for metric_name, metric_value in res.items():
            results_dict[metric_name].append(metric_value)
    for metric_name, metric_value in results_dict.items():
        print("{}{}".format(metric_name.ljust(12), round(
            sum(metric_value)/len(metric_value), 3)))
    logging.info(f"bad pdbs: {bad_pdbs}")
    

def main2(args):
    logging.info("calculating the upper bound of the metrics using two prediction...")
    init_string = "-mute all -check_cdr_chainbreaks false -detect_disulf true"
    pyrosetta.init(init_string)

    bad_pdbs = []
    results_labels = [
        'ocd', 'frh_rms', 'h1_rms', 'h2_rms', 'h3_rms', 'frl_rms', 'l1_rms',
        'l2_rms', 'l3_rms'
    ]
    results_dict = {}
    for label in results_labels:
        results_dict[label] = []

    valid_pred_dir = 1
    path_list = [args.pred_dir]
    for path in [args.pred_dir2, args.pred_dir3, args.pred_dir4, args.pred_dir5]:
        if path is not None:
            valid_pred_dir += 1
            path_list.append(path)
    logging.info(f"got {len(path_list)} predictions to ensemble.")

    jobs = []
    for i in range(0, valid_pred_dir):
        jobs.append([x for x in os.listdir(path_list[i]) if x.endswith('.pdb')])

    metrics_list = []
    for i in range(len(jobs)):
        metrics_i = {}
        test_list = []
        for fname in tqdm(jobs[i]):
            pdbid = fname[:4]
            test_list.append(pdbid)
            pred_pdb_path = os.path.join(path_list[i], fname)
            native_pdb_path = os.path.join(args.target_dir, pdbid + '.pdb')
            # logging.info(f'calculating {pdbid}...')
            try:
                metrics_i[pdbid] = get_pair_results(pred_pdb_path, native_pdb_path)
            except Exception as exception:
                bad_pdbs.append(pdbid)
                logging.warning(f"{pdbid} failed: {exception}")
        metrics_list.append(metrics_i)

    metrics = []
    metrics1 = metrics_list[0]
    for key in metrics1.keys(): # pdb key
        new_metric = {}
        v = metrics1[key]
        best = v["h3_rms"]
        for metrics_i in metrics_list:
            if metrics_i[key]["h3_rms"] < best:
                best = metrics_i[key]["h3_rms"]
                v = metrics_i[key]
        
        for k in v.keys():
            new_metric[k] = v[k]
        metrics.append(new_metric)

    logging.info(f"got {len(test_list)} pdbs to compare.")
    logging.info(test_list)
    for res in metrics:
        for metric_name, metric_value in res.items():
            results_dict[metric_name].append(metric_value)
    for metric_name, metric_value in results_dict.items():
        print("{}{}".format(metric_name.ljust(12), round(
            sum(metric_value)/len(metric_value), 3)))
    logging.info(f"bad pdbs: {bad_pdbs}")


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
        "pred_dir", type=str,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "target_dir", type=str,
        help="Path to the native_pdb"
    )
    parser.add_argument(
        "--pred_dir2", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pred_dir3", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pred_dir4", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pred_dir5", type=str, default=None,
        help="Path to the pred_pdb"
    )            
    args = parser.parse_args()

    main2(args)
