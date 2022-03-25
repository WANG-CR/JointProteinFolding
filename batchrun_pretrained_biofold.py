import logging
logging.basicConfig(level=logging.INFO)
import debugger
import torch
from matplotlib import ticker
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import time
import argparse

from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.feats import atom14_to_atom37
from openfold.utils.seed import seed_everything
from openfold.utils.loss import lddt_ca
from openfold.np import residue_constants, protein
from openfold.model.model import AlphaFold
from openfold.data import feature_pipeline, data_pipeline, parsers
from openfold.config import model_config


def gather_jobs(fasta_path, mode=0):
    """gather fastas from the fasta directory

    Args:
        fasta_path ([str]): The directory containing all fastas.
        mode (int, optional): 
            0: all jobs in one fasta, 
            1: jobs in separate fastas. 
            2: jobs in separate pdbs. Only used in loop refinement
            Defaults to 0.
    Return:
        list of tuples [(name, h_len, l_len, seq)] if mode in [0, 1]
        list of tuples [(name, pdb_path)] if mode is 2
    """
    if mode == 0:
        jobs = []
        fasta_dict = {}
        chain2id = {'H': 0, 'L': 1}

        with open(fasta_path, "r") as fp:
            fasta_str = fp.read()
        input_seqs, input_descs = parsers.parse_fasta(fasta_str)
        for i, (seq, desc) in enumerate(zip(input_seqs, input_descs)):
            tag = desc[:4].lower()
            chain = desc[-1].upper()
            assert chain in [
                'H', 'L'], f'only accept H chain and L chain, found {chain}'
            if tag not in fasta_dict:
                fasta_dict[tag] = ['', '']
            fasta_dict[tag][chain2id[chain]] = seq
        for k, v in fasta_dict.items():
            job = [k, len(v[0]), len(v[1]), ''.join(v)]
            assert job[1] > 0, 'missing heavy chain'
            assert job[2] > 0, 'missing light chain'
            jobs.append(job)
        return jobs

    elif mode == 1:
        jobs = []

        for f in os.listdir(fasta_path):
            path = os.path.join(fasta_path, f)
            ext = os.path.splitext(f)[-1]

            if (ext == ".fasta"):
                with open(path, "r") as fp:
                    fasta_str = fp.read()
                input_seqs, input_descs = parsers.parse_fasta(fasta_str)
                tag = input_descs[0][:4].lower()
                jobs.append([tag, len(input_seqs[0]), len(
                    input_seqs[1]), input_seqs[0] + input_seqs[1]])
            else:
                continue

        return jobs

    elif mode == 2:
        jobs = []

        for f in os.listdir(fasta_path):
            path = os.path.join(fasta_path, f)
            ext = os.path.splitext(f)[-1]

            if (ext == ".pdb"):
                tag = f[:4].lower()
                jobs.append([tag, path])
            else:
                continue

        return jobs

    else:
        raise ValueError(f'expect mode to be either 0, 1, or 2, found {mode}')


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

    if args.pdb_path:
        try:
            import pyrosetta
            from pyrosetta.rosetta.protocols.antibody import CDRNameEnum
            pyrosetta.init(
                "-mute all -check_cdr_chainbreaks false -detect_disulf true")
        except Exception as err:
            logging.warning(err)

    jobs = gather_jobs(args.fasta_path, mode=args.fasta_mode)
    logging.info(f'got {len(jobs)} jobs...')

    for job in jobs:
        tag, h_len, l_len, seq = job
        assert h_len > 0, "missing heavy chain"
        assert l_len > 0, "missing light chain"
        chain_index = np.array([0] * h_len + [1] * l_len)
        logging.info("tag: %s" % tag)
        logging.info("H length: %d, L length: %d" % (h_len, l_len))
        logging.info("H seq: %s" % seq[:h_len])
        logging.info("L seq: %s" % seq[h_len:])
        fasta_path = os.path.join(args.output_dir, "tmp.fasta")
        with open(fasta_path, "w") as fp:
            fp.write(f">{tag}_H\n{seq[:h_len]}\n")
            fp.write(f">{tag}_L\n{seq[h_len:]}")

        if args.residue_embedding_dir is not None:
            local_residue_embedding_dir = os.path.join(
                args.residue_embedding_dir, tag)
        else:
            local_residue_embedding_dir = None
        local_alignment_dir = os.path.join(alignment_dir, tag)
        if not os.path.exists(local_alignment_dir):
            local_alignment_dir = alignment_dir

        feature_dict = data_processor.process_fasta(
            fasta_path=fasta_path,
            alignment_dir=local_alignment_dir,
            embedding_dir=local_residue_embedding_dir,
            _alignment_index=None,
            is_antibody=True,
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

        if args.pdb_path:
            pdb_file = os.path.join(args.pdb_path, f"{tag}.pdb")
            with open(pdb_file, "r") as fin:
                pdb_str = fin.read()
            protein_object = protein.from_pdb_string_antibody(pdb_str)
            loop_index = np.copy(protein_object.loop_index)

            starts = []
            ends = []

            if loop_index is None:
                gt_pose = pyrosetta.pose_from_pdb(pdb_file)
                gt_ab = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(
                    gt_pose)
                assert gt_pose.total_residue() == len(chain_index.shape[0])

                for loop in ["h1", "h2", "h3"]:
                    loop = getattr(CDRNameEnum, loop)
                    # CDR loop is by default [start, end]
                    start = gt_ab.get_CDR_start(loop, gt_pose)
                    end = gt_ab.get_CDR_end(loop, gt_pose)
                    starts.append(start)
                    ends.append(end)
                for loop in ["l1", "l2", "l3"]:
                    loop = getattr(CDRNameEnum, loop)
                    # CDR loop is by default [start, end]
                    start = gt_ab.get_CDR_start(loop, gt_pose) - h_len
                    end = gt_ab.get_CDR_end(loop, gt_pose) - h_len
                    starts.append(start)
                    ends.append(end)
            else:
                # 123456 for CDR H123, CDR L123, respectively
                for loop_idx in range(1, 6 + 1):
                    assert loop_idx in loop_index, f"loop id {loop_idx} is not in loop_index"
                    # + 1: residue index starts from 1 in plot.
                    # +- 2: exclude anchor residues on both sides.
                    starts.append(
                        np.where(loop_index == loop_idx)[
                            0][0] + 1 + 2 - h_len * (loop_idx >= 4)
                    )
                    ends.append(
                        np.where(loop_index == loop_idx)[
                            0][-1] + 1 - 2 - h_len * (loop_idx >= 4)
                    )

            # lddt at each step in the last recycle, step_size = 8
            # [8, *, N, 14, 3] --> [8, N, 14, 3]
            all_atom_pred_pos = out["sm"]["positions"]
            all_atom_pred_pos = atom14_to_atom37(
                all_atom_pred_pos,
                tensor_tree_map(lambda t: t[..., -1][None], batch)
            )  # [8, N, 37, 3]
            all_atom_positions = torch.tensor(protein_object.atom_positions).to(
                all_atom_pred_pos)  # [N, 37, 3]
            all_atom_mask = torch.tensor(protein_object.atom_mask).to(
                all_atom_pred_pos)  # [N, 37]
            sm_lddt = lddt_ca(
                all_atom_pred_pos, all_atom_positions[None], all_atom_mask[None],
                eps=config.globals.eps,
                per_residue=True) * 100  # [8, N]

            # lddt at each final step in each recycle, no_recycle = 3
            all_atom_pred_pos = torch.stack(
                [out_["final_atom_positions"] for out_ in out["recycle_outputs"]], dim=0
            )  # [4, *, N, 37, 3]
            recycle_lddt = lddt_ca(
                all_atom_pred_pos, all_atom_positions[None], all_atom_mask[None],
                eps=config.globals.eps,
                per_residue=True,
            ) * 100  # [4, *, N]

            step_lddts = [np.array(sm_lddt.cpu()),
                          np.array(recycle_lddt.cpu())]

        # Toss out the recycling dimensions --- we don't need them anymore
        batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

        plddt = out["plddt"]
        mean_plddt = np.mean(plddt)

        if args.pdb_path:
            # plot plddt curve w.r.t. residue & step
            step_plddts = [
                out["plddt_by_sm_step"],  # by step (last recyle)
                np.stack(
                    [out_["plddt"] for out_ in out["recycle_outputs"]], axis=0
                )  # by recycle (last step)
            ]
            legends = [
                ["step %d" % i for i in range(len(out["plddt_by_sm_step"]))]
            ]
            paths = ["plddt_by_sm_step"]

            legends.append(
                ["recycle %d" % i for i in range(len(out["recycle_outputs"]))])
            paths.append("plddt_by_recycle")

            max_lines = 10

            for i in range(2):
                step_plddt = step_plddts[i]
                step_lddt = step_lddts[i]
                if len(step_plddt) > max_lines:
                    step = len(step_plddt) // max_lines
                    if len(step_plddt) % max_lines > 0:
                        step += 1
                    steps = range(0, len(step_plddt), step)
                else:
                    steps = range(0, len(step_plddt))
                path = paths[i]
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(2, 1, 1)
                x = np.arange(1, h_len + 1)
                lines = []
                legend = [legends[i][s] for s in steps]
                for j in range(len(steps)):
                    line = ax.plot(
                        x, step_plddt[steps[j], :h_len], linewidth=1, color="C%d" % j)[0]
                    ax.plot(x, step_lddt[steps[j], :h_len],
                            linestyle="dotted", linewidth=1, color="C%d" % j)
                    lines.append(line)
                ax.legend(lines, legend, loc="lower left")
                ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
                ax.set_xlabel("residue index")
                ax.set_ylabel("pLDDT (solid) / LDDT (dotted)")
                ax.set_title("%s heavy chain" % tag)
                # y_min, y_max = ax.get_ylim()
                y_min, y_max = 50, 105
                y = np.linspace(y_min, y_max, 10)
                for start, end in zip(starts[:3], ends[:3]):
                    plt.fill_betweenx(
                        y, start, end, color="tab:red", alpha=0.3)
                ax.set_ylim(y_min, y_max)

                ax = fig.add_subplot(2, 1, 2)
                x = np.arange(1, l_len + 1)
                lines = []
                for j in range(len(steps)):
                    line = ax.plot(
                        x, step_plddt[steps[j], h_len:], linewidth=1, color="C%d" % j)[0]
                    ax.plot(x, step_lddt[steps[j], h_len:],
                            linestyle="dotted", linewidth=1, color="C%d" % j)
                    lines.append(line)
                ax.legend(lines, legend, loc="lower left")
                ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
                ax.set_xlabel("residue index")
                ax.set_ylabel("pLDDT (solid) / LDDT (dotted)")
                ax.set_title("%s light chain" % tag)
                # y_min, y_max = ax.get_ylim()
                y_min, y_max = 50, 105
                y = np.linspace(y_min, y_max, 10)
                for start, end in zip(starts[3:], ends[3:]):
                    plt.fill_betweenx(
                        y, start, end, color="tab:red", alpha=0.3)
                ax.set_ylim(y_min, y_max)

                fig.subplots_adjust(hspace=0.3)
                fig.tight_layout()
                if not os.path.exists(os.path.join(args.output_dir, path)):
                    os.makedirs(os.path.join(args.output_dir, path))
                fig.savefig(
                    os.path.join(
                        args.output_dir,
                        path,
                        f"{tag}_{ckpt_epoch}_{args.config_preset}_rec{args.no_recycling_iters}.png",
                    ),
                    dpi=150,
                )

        plddt_b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )

        unrelaxed_protein = protein.from_prediction(
            features=batch,
            result=out,
            b_factors=plddt_b_factors,
            chain_index=chain_index
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
            except:
                logging.warning("relaxation failed...")

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
        "fasta_path", type=str,
    )
    parser.add_argument(
        "fasta_mode", type=int,
        help="""0: all jobs in one fasta, 1: jobs in separate fastas, only used when `from_fasta` is True"""
    )
    parser.add_argument(
        "resume_from_ckpt", type=str,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--pdb_path", type=str, default=None,
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
