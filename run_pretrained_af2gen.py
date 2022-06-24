import debugger
import logging
logging.basicConfig(level=logging.INFO)
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

    # Get input pdb
    f_path = os.path.basename(args.pdb_path)
    tag = f_path[:4].lower()
    # with open(args.fasta_path, "r") as fp:
    #     fasta_str = fp.read()
    # input_seqs, input_descs = parsers.parse_fasta(fasta_str)
    # tag = input_descs[0][:4].lower()
    # chain_index = np.array([0] * len(input_seqs[0]) + [1] * len(input_seqs[1]))
    # h_len, l_len = len(input_seqs[0]), len(input_seqs[1])

    output_dir_base = args.output_dir
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    if(args.use_precomputed_alignments is None):
        alignment_dir = output_dir_base
    else:
        alignment_dir = os.path.join(args.use_precomputed_alignments, tag)
    if(args.residue_embedding_dir is None):
        residue_embedding_dir = None
    else:
        residue_embedding_dir = os.path.join(
            args.residue_embedding_dir, tag
        )

    feature_dict = data_processor.process_pdb(
        pdb_path=args.pdb_path,
        alignment_dir=alignment_dir,
        embedding_dir=residue_embedding_dir,
        _alignment_index=None,
        is_antibody=True,
        trunc_antigen=True,
    )

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
    mean_plddt = np.mean(plddt)



    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors,
        chain_index=batch["chain_index"],
    )

    # Save the unrelaxed PDB.
    unrelaxed_output_path = os.path.join(
        args.output_dir,
        f"{tag}_{ckpt_epoch}_{args.config_preset}_rec{args.no_recycling_iters}_unrelaxed.pdb"
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
                f"{tag}_{ckpt_epoch}_{args.config_preset}_rec{args.no_recycling_iters}_relaxed.pdb"
            )
            with open(relaxed_output_path, 'w') as f:
                f.write(relaxed_pdb_str)
        except Exception as e:
            logging.warning(e)
            logging.warning("relaxation failed...")


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
