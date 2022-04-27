from openfold.np import protein, residue_constants
import shutil
from tqdm import tqdm
import os
import numpy as np
import argparse
import logging
logging.basicConfig(level=logging.INFO)


def get_distance(pdb, h3_only=False):
    with open(pdb, 'r') as f:
        pdb_str = f.read()
    protein_obj = protein.from_pdb_string_antibody(pdb_str)
    all_atom_positions = protein_obj.atom_positions
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_positions = all_atom_positions[..., ca_pos, :]  # [num_res, 3]
    if h3_only:
        all_atom_positions = all_atom_positions[protein_obj.loop_index == 3]

    d = all_atom_positions[..., :, None, :] - \
        all_atom_positions[..., None, :, :]
    d = d ** 2
    d = np.sqrt(np.sum(d, axis=-1))  # [*, num_res, num_res]
    return d


def ensemble_pdbs_by_distance(pdbs, h3_only=False):
    distances = []
    for pdb in pdbs:
        distances.append(get_distance(pdb, h3_only))
    distances = np.stack(distances)  # [num_pdbs, num_res, num_res]

    # ensembled distance!
    # [1, num_res, num_res]
    mean_distance = np.mean(distances, axis=0, keepdims=True)

    # calculate deviations from mean distance (drmsd).
    drmsd = distances - mean_distance
    drmsd = drmsd ** 2
    drmsd = np.sum(drmsd, axis=(-1, -2))  # [num_pdbs, ]
    denom = distances.shape[-1] * (distances.shape[-1] - 1)
    drmsd = np.sqrt(drmsd / denom)  # [num_pdbs, ]

    return pdbs[np.argmin(drmsd)], drmsd


def main(args):
    valid_pred_dir = 1
    path_list = [args.pdb_dir1]
    for path in [
        args.pdb_dir2, args.pdb_dir3, args.pdb_dir4,
        args.pdb_dir5, args.pdb_dir6, args.pdb_dir7,
        args.pdb_dir8, args.pdb_dir9, args.pdb_dir9,
    ]:
        if path is not None:
            valid_pred_dir += 1
            path_list.append(path)
    assert valid_pred_dir > 1
    logging.info(f"got {len(path_list)} predictions to ensemble.")

    jobs = []
    for i in range(0, valid_pred_dir):
        jobs.append([x for x in os.listdir(
            path_list[i]) if x.endswith('.pdb')])

    ensemble_list = []
    for i in range(len(jobs)):
        ensemble_i = {}
        for fname in tqdm(jobs[i]):
            pdbid = fname[:4]
            ensemble_i[pdbid] = os.path.join(path_list[i], fname)
        ensemble_list.append(ensemble_i)

    ensembled_path_list = []
    ensemble_1 = ensemble_list[0]
    for key in ensemble_1:  # pdb key
        pdb_paths = [x[key] for x in ensemble_list]
        ensembled_pdb_path, drmsd = ensemble_pdbs_by_distance(
            pdb_paths,
            h3_only=False,
        )
        logging.info(f'ensembling {key}, drmsd: {drmsd}.')
        ensembled_path_list.append(ensembled_pdb_path)

    logging.info("copying ensembled pdbs to target dirs.")
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir, exist_ok=True)
    for fpath in ensembled_path_list:
        tgt_path = os.path.join(args.target_dir, os.path.basename(fpath))
        shutil.copy2(fpath, tgt_path)


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
        "--pdb_dir1", type=str,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pdb_dir2", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pdb_dir3", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pdb_dir4", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pdb_dir5", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pdb_dir6", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pdb_dir7", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pdb_dir8", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pdb_dir9", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--pdb_dir10", type=str, default=None,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "--target_dir", type=str,
        help="Path to the native_pdb"
    )
    args = parser.parse_args()

    main(args)
