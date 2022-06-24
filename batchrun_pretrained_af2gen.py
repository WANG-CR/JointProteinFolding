import logging
import pdb
logging.basicConfig(level=logging.INFO)
import debugger
import torch
from matplotlib import ticker
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import sys
import time
import argparse

from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.feats import atom14_to_atom37
from openfold.utils.seed import seed_everything
from openfold.utils.loss import lddt_ca
from openfold.np import residue_constants, protein
from openfold.model.model import AlphaFold
from openfold.data import feature_pipeline, data_pipeline, data_transforms, parsers
from openfold.config import model_config


def gather_job(pdb_dir):
    pdb_paths = []
    for f_path in os.listdir(pdb_dir):
        if f_path.endswith('.pdb'):
            pdb_path = os.path.join(pdb_dir, f_path)
            pdb_paths.append(pdb_path)
    
    return pdb_paths


def main(args):
    if(args.seed is not None):
        seed_everything(args.seed)

    config = model_config(
        name=args.config_preset,
        yaml_config_preset=args.yaml_config_preset,
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
    template_featurizer = None
    logging.warning(
        "'template_featurizer' is set as None."
    )

    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )
    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    output_dir_base = args.output_dir
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    if(args.use_precomputed_alignments is None):
        alignment_dir = output_dir_base
    else:
        alignment_dir = args.use_precomputed_alignments

    jobs = gather_job(args.pdb_dir)
    logging.info(f'got {len(jobs)} jobs...')

    aars = []
    for job in jobs:
        f_path = os.path.basename(job)
        tag = f_path[:4].lower()

        if args.residue_embedding_dir is not None:
            local_residue_embedding_dir = os.path.join(
                args.residue_embedding_dir, tag)
        else:
            local_residue_embedding_dir = None
        local_alignment_dir = os.path.join(alignment_dir, tag)
        if not os.path.exists(local_alignment_dir):
            local_alignment_dir = alignment_dir

        if args.residue_attn_dir is not None:
            attn_path_H = os.path.join(
                args.residue_attn_dir, tag + '_H.oaspt'
            )
            attn_path_L = os.path.join(
                args.residue_attn_dir, tag + '_L.oaspt'
            )
        else:
            attn_path_H = attn_path_L = None

        feature_dict = data_processor.process_pdb(
            pdb_path=job,
            alignment_dir=local_alignment_dir,
            embedding_dir=local_residue_embedding_dir,
            _alignment_index=None,
            is_antibody=True,
            trunc_antigen=True,
        )
    

        if args.pred_pdb_dir:
            pdb_pattern = os.path.join(args.pred_pdb_dir, f"{tag}*.pdb")
            pdb_files = glob.glob(pdb_pattern)
            assert len(pdb_files) == 1, f"{len(pdb_files)} pdbs found for {tag}..."
            pdb_file = pdb_files[0]
            with open(pdb_file, "r") as fin:
                pdb_str = fin.read()
            protein_object = protein.from_pdb_string_antibody(pdb_str)
            pdb_feats = data_pipeline.make_pdb_features(
                protein_object,
                tag.upper(),
            )
            # copy whatever features that are not in the evaluation batch
            for key, value in pdb_feats.items():
                if key not in feature_dict:
                    feature_dict[key] = value
            feature_dict["pred_atom_positions"] = feature_dict["all_atom_positions"]
            feature_dict["loop_index"] = pdb_feats["loop_index"]
            feature_dict["aatype"] = pdb_feats["aatype"]

        feature_dict["no_recycling_iters"] = args.no_recycling_iters
        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode="predict",
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
        batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)

        # patch the loop design case
        if "final_pred_aatype" in out:
            fake_batch = {"aatype": out["final_pred_aatype"]}
            fake_batch = data_transforms.make_atom14_masks(fake_batch)
            out["final_atom_positions"] = atom14_to_atom37(
                out["sm"]["positions"][-1], fake_batch
            )
            out["residx_atom37_to_atom14"] = fake_batch["residx_atom37_to_atom14"]
            out["atom37_atom_exists"] = fake_batch["atom37_atom_exists"]
            out["final_atom_mask"] = fake_batch["atom37_atom_exists"]
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

        plddt = out["plddt"]
        # 1. naive mean
        # mean_plddt = np.mean(plddt)
        # 2. Following IgFold, we use 90-th percentile error
        mean_plddt = np.percentile(plddt, 100 - 90)
        # 3. Average plddt on CDR h3 region.
        # mean_plddt = np.mean(plddt[loop_index == 3])
        mean_lddt = 0.
        
        plddt_b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )
        
        final_pred_aatype = out["final_pred_aatype"]
        gt_aatype = batch["aatype"]
        loop_index = batch["loop_index"]
        final_pred_aatype = final_pred_aatype[loop_index==3]
        gt_aatype = gt_aatype[loop_index==3]
        aar = (final_pred_aatype == gt_aatype).sum() / len(gt_aatype)
        aars.append(aar)
        logging.info(f"aar of {tag}: {aar}")

        unrelaxed_protein = protein.from_prediction(
            features=batch,
            result=out,
            b_factors=plddt_b_factors,
            chain_index=batch["chain_index"],
        )

        # Save the unrelaxed PDB.
        unrelaxed_output_path = os.path.join(
            args.output_dir,
            f"{tag}_{ckpt_epoch}_{args.config_preset}_" \
            f"rec{args.no_recycling_iters}_plddt{mean_plddt:.3f}_" \
            f"lddt{mean_lddt:.3f}_unrelaxed.pdb"
        )
        with open(unrelaxed_output_path, 'w') as f:
            f.write(protein.to_pdb(unrelaxed_protein))

        if args.relax:
            if "relax" not in sys.modules:
                import openfold.np.relax.relax as relax

            logging.info("start relaxation")
            amber_relaxer = relax.AmberRelaxation(
                use_gpu=(args.model_device != "cpu"),
                **config.relax,
            )
            try:
                # Relax the prediction.
                t = time.perf_counter()
                visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
                if("cuda" in args.model_device):
                    device_no = args.model_device.split(":")[-1]
                    os.environ["CUDA_VISIBLE_DEVICES"] = device_no
                relaxed_pdb_str, _, _ = amber_relaxer.process(
                    prot=unrelaxed_protein)
                os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
                logging.info(f"Relaxation time: {time.perf_counter() - t}")

                # Save the relaxed PDB.
                relaxed_output_path = os.path.join(
                    args.output_dir,
                    f"{tag}_{ckpt_epoch}_{args.config_preset}_" \
                    f"rec{args.no_recycling_iters}_plddt{mean_plddt:.3f}_" \
                    f"lddt{mean_lddt:.3f}_relaxed.pdb"                    
                )
                with open(relaxed_output_path, 'w') as f:
                    f.write(relaxed_pdb_str)
            except Exception as e:
                logging.warning(e)
                logging.warning("relaxation failed...")

    avg_aar = sum(aars) / len(aars)
    logging.info(f"avg aars : {avg_aar}")

    if os.path.exists(os.path.join(args.output_dir, "tmp.fasta")):
        os.remove(os.path.join(args.output_dir, "tmp.fasta"))


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
        "resume_from_ckpt", type=str,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--pdb_dir", type=str, default=None,
        help="Path to ground truth pdb data"
    )
    parser.add_argument(
        "--use_precomputed_alignments", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--residue_embedding_dir", type=str, default=None,
        help="""Path to pre-trained residue embedding directory. If not provided, 
                the model will ignore the feature."""
    )
    parser.add_argument(
        "--pred_pdb_dir", type=str, default=None,
        help="Directory containing predicted pdb structures"
    )
    parser.add_argument(
        "--residue_attn_dir", type=str, default=None,
        help="Directory containing residue attention features"
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
        "--cpus", type=int, default=12,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        '--seed', type=str, default=None,
        help="Random seed"
    )
    args = parser.parse_args()

    if(args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    if(args.config_preset is None and args.yaml_config_preset is None):
        raise ValueError(
            "Either --config_preset or --yaml_config_preset should be specified."
        )

    if(args.yaml_config_preset is not None):
        if not os.path.exists(args.yaml_config_preset):
            raise FileNotFoundError(
                f"{os.path.abspath(args.yaml_config_preset)}")
        args.config_preset = os.path.splitext(
            os.path.basename(args.yaml_config_preset)
        )[0]
        logging.info(
            f"the config_preset is set as {args.config_preset} by yaml_config_preset.")

    main(args)
