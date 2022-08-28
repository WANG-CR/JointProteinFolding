import logging
logging.basicConfig(level=logging.WARNING)
import copy
import yaml
import ml_collections as mlc
from typing import Optional


def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf


def recursive_set(target, src):
    """
        Recursively set target dict using src dict by matching keys and depths
    """
    for (k, v) in src.items():
        if not isinstance(v, dict):
            target[k] = v
        else:
            if k not in target:
                raise ValueError(
                    f"Key error. Target dict does not contain the key {k}. "
                    "Please check your source (yaml) dict."
                )
            recursive_set(target[k], v)


def model_config(
    name,
    yaml_config_preset: Optional[str] = None,
    train: bool = False,
    low_prec: bool = False,
):
    c = copy.deepcopy(config)

    if yaml_config_preset is not None:
        with open(yaml_config_preset, 'r') as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config is not None:
            recursive_set(c, yaml_config)
        else:
            logging.warning("The yaml config is empty!")
    else:
        if name == "initial_training":
            pass
        else:
            raise ValueError("Invalid model name")

    if train:
        c.globals.blocks_per_ckpt = 1
        c.globals.chunk_size = None

    if low_prec:
        c.globals.eps = 1e-4
        c.globals.low_prec = True
        # If we want exact numerical parity with the original, inf can't be
        # a global constant
        set_inf(c, 1e4)
    
    if c.data.data_module.data_loaders.batch_size > 1:
        # We assume the training crop_size is always bigger
        # than the sequence length during the validation.
        c.data.eval.crop_size = c.data.train.crop_size

    return c


# loss weight
fape_weight = mlc.FieldReference(1.0, field_type=float)
seqs_weight = mlc.FieldReference(1.0, field_type=float)
supervised_chi_weight = mlc.FieldReference(1.0, field_type=float)
lddt_weight = mlc.FieldReference(0.01, field_type=float)

## not used in cath_gen
distogram_weight = mlc.FieldReference(0.0, field_type=float)
experimentally_resolved_weight = mlc.FieldReference(0.0, field_type=float)
violation_weight = mlc.FieldReference(0.0, field_type=float)
tm_weight = mlc.FieldReference(0.0, field_type=float)

c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(256, field_type=int)
c_s = mlc.FieldReference(384, field_type=int)
blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
tm_enabled = mlc.FieldReference(False, field_type=bool)
eps = mlc.FieldReference(1e-8, field_type=float)

NUM_RES = "num residues placeholder"


config = mlc.ConfigDict(
    {
        "data": {
            "common": {
                "feat": {
                    "aatype": [NUM_RES],
                    "sstype": [NUM_RES],
                    "ss_feat": [NUM_RES, None],
                    "all_atom_mask": [NUM_RES, None],
                    "all_atom_positions": [NUM_RES, None, None],
                    "alt_chi_angles": [NUM_RES, None],
                    "atom14_alt_gt_exists": [NUM_RES, None],
                    "atom14_alt_gt_positions": [NUM_RES, None, None],
                    "atom14_atom_exists": [NUM_RES, None],
                    "atom14_atom_is_ambiguous": [NUM_RES, None],
                    "atom14_gt_exists": [NUM_RES, None],
                    "atom14_gt_positions": [NUM_RES, None, None],
                    "atom37_atom_exists": [NUM_RES, None],
                    "backbone_rigid_mask": [NUM_RES],
                    "backbone_rigid_tensor": [NUM_RES, None, None],
                    "backbone_rigid_tensor_7s": [NUM_RES, None],
                    "chi_angles_sin_cos": [NUM_RES, None, None],
                    "torsion_angles_sin_cos": [NUM_RES, None, None],
                    "chi_mask": [NUM_RES, None],
                    "no_recycling_iters": [],
                    "pseudo_beta": [NUM_RES, None],
                    "pseudo_beta_mask": [NUM_RES],
                    "residue_index": [NUM_RES],
                    "chain_index": [NUM_RES],
                    "residx_atom14_to_atom37": [NUM_RES, None],
                    "residx_atom37_to_atom14": [NUM_RES, None],
                    "rigidgroups_alt_gt_frames": [NUM_RES, None, None, None],
                    "rigidgroups_group_exists": [NUM_RES, None],
                    "rigidgroups_group_is_ambiguous": [NUM_RES, None],
                    "rigidgroups_gt_exists": [NUM_RES, None],
                    "rigidgroups_gt_frames": [NUM_RES, None, None, None],
                    "seq_length": [],
                    "resolution": [],
                    "seq_mask": [NUM_RES],
                    "target_feat": [NUM_RES, None],
                    "use_clamped_fape": [],
                },
                "max_recycling_iters": 3,
                "unsupervised_features": [
                    "aatype",
                    "sstype",
                    "residue_index",
                    "chain_index",
                    "seq_length",
                    "between_segment_residues",
                    "no_recycling_iters",
                ],
            },
            "supervised": {
                "clamp_prob": 0.9,
                "supervised_features": [
                    "all_atom_mask",
                    "all_atom_positions",
                    "resolution",
                    "use_clamped_fape",
                ],
            },
            "predict": {
                "fixed_size": True,
                "crop": False,
                "crop_size": None,
                "supervised": False,
                "uniform_recycling": False,
            },
            "eval": {
                "fixed_size": True,
                "crop": False,
                "crop_size": None, # necessary for batch_size >= 2
                "supervised": True,
                "uniform_recycling": False,
            },
            "train": {
                "fixed_size": True,
                "crop": True,
                "crop_size": 256,
                "supervised": True,
                "clamp_prob": 0.9,
                "uniform_recycling": True,
            },
            "data_module": {
                "data_loaders": {
                    "batch_size": 1,
                    "num_workers": 10,
                },
            },
        },
        # Recurring FieldReferences that can be changed globally here
        "globals": {
            "blocks_per_ckpt": blocks_per_ckpt,
            "chunk_size": chunk_size,
            "c_z": c_z,
            "c_m": c_m,
            "c_s": c_s,
            "eps": eps,
            "low_prec": False,
        },
        "optimizer": {
            "lr": 0.001,
            "eps": 1e-5,
        },
        "scheduler": {
            "warmup_no_steps": 5000,
            "start_decay_after_n_steps": 50000,
            "decay_every_n_steps": 2000,
            "decay_factor": 0.95,
        },
        "model": {
            "_mask_trans": False,
            "input_embedder": {
                "tf_dim": 3,
                "c_z": c_z,
                "c_m": c_m,
                "relpos_k": 32,
            },
            "recycling_embedder": {
                "c_z": c_z,
                "c_m": c_m,
                "min_bin": 3.25,
                "max_bin": 20.75,
                "no_bins": 15,
                "inf": 1e8,
            },
            "evoformer_stack": {
                "c_m": c_m,
                "c_z": c_z,
                "c_hidden_seq_att": 32,
                "c_hidden_opm": 32,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "c_s": c_s,
                "no_heads_seq": 8,
                "no_heads_pair": 4,
                "no_blocks": 48,
                "transition_n": 4,
                "seq_dropout": 0.15,
                "pair_dropout": 0.25,
                "blocks_per_ckpt": blocks_per_ckpt,
                "clear_cache_between_blocks": False,
                "inf": 1e9,
            },
            "structure_module": {
                "c_s": c_s,
                "c_z": c_z,
                "c_ipa": 16,
                "c_resnet": 128,
                "no_heads_ipa": 12,
                "no_qk_points": 4,
                "no_v_points": 8,
                "dropout_rate": 0.1,
                "no_blocks": 8,
                "no_transition_layers": 1,
                "no_resnet_blocks": 2,
                "no_angles": 7,
                "trans_scale_factor": 10,
                "epsilon": eps,  # 1e-12,
                "inf": 1e5,
            },
            "heads": {
                "lddt": {
                    "no_bins": 50,
                    "c_in": c_s,
                    "c_hidden": 128,
                    "weight": lddt_weight,
                },
                "distogram": {
                    "c_z": c_z,
                    "no_bins": aux_distogram_bins,
                    "weight": distogram_weight,
                },
                "tm": {
                    "c_z": c_z,
                    "no_bins": aux_distogram_bins,
                    "enabled": tm_enabled,
                    "weight": tm_weight,
                },
                "experimentally_resolved": {
                    "c_s": c_s,
                    "c_out": 37,
                    "weight": experimentally_resolved_weight,
                },
            },
            "contact": {
                "cutoff": 8,
                "eps": eps,
            },
        },
        "relax": {
            "max_iterations": 0,  # no max
            "tolerance": 2.39,
            "stiffness": 10.0,
            "max_outer_iterations": 20,
            "exclude_residues": [],
        },
        "loss": {
            "distogram": {
                "min_bin": 2.3125,
                "max_bin": 21.6875,
                "no_bins": 64,
                "eps": eps,  # 1e-6,
                "weight": distogram_weight,
            },
            "experimentally_resolved": {
                "eps": eps,  # 1e-8,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "weight": experimentally_resolved_weight,
            },
            "fape": {
                "backbone": {
                    "clamp_distance": 10.0,
                    "loss_unit_distance": 10.0,
                    "weight": 0.5,
                },
                "sidechain": {
                    "clamp_distance": 10.0,
                    "length_scale": 10.0,
                    "weight": 0.5,
                },
                "eps": 1e-4,
                "weight": fape_weight,
            },
            "lddt": {
                "min_resolution": 0.0,
                "max_resolution": 3.0,
                "cutoff": 15.0,
                "no_bins": 50,
                "eps": eps,  # 1e-10,
                "weight": lddt_weight,
            },
            "seqs": {
                "eps": eps,  # 1e-8,
                "weight": seqs_weight,
            },
            "supervised_chi": {
                "chi_weight": 0.5,
                "angle_norm_weight": 0.01,
                "eps": eps,  # 1e-6,
                "weight": supervised_chi_weight,
            },
            "violation": {
                "violation_tolerance_factor": 12.0,
                "clash_overlap_tolerance": 1.5,
                "eps": eps,  # 1e-6,
                "weight": violation_weight,
            },
            "tm": {
                "max_bin": 31,
                "no_bins": 64,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "eps": eps,  # 1e-8,
                "weight": tm_weight,
                "enabled": tm_enabled,
            },
            "eps": eps,
        },
        "ema": {"decay": 0.999},
    }
)


# def test_config(yaml_dir):
#     jobs = [
#         "vanilla", "warmup0", "warmup2000", "warmup5000", "warmup10000",
#         "bs2", "esm1b_cat", "debug",
#     ]
#     jobs_openfold = [
#         "initial_training", "model_1", "model_2", "model_3", "model_4",
#         "model_5", "model_1_ptm", "model_2_ptm", "model_3_ptm", "model_4_ptm",
#         "model_5_ptm", "finetuning",
#     ]
#     jobs = jobs + jobs_openfold
    
#     for job in jobs:
#         logging.warning(f"testing config={job}")
#         yaml_dir_ = os.path.join(yaml_dir, "openfold") if job in jobs_openfold else yaml_dir
#         yaml_path = os.path.join(yaml_dir_, job + ".yml")

#         with open(yaml_path, 'r') as f:
#             yaml_config = yaml.safe_load(f)
#         config_1 = model_config(job)
#         config_2 = model_config("initial_training")

#         assert not config_1 == config_2 or job == "initial_training"
#         if yaml_config is not None:
#             recursive_set(config_2, yaml_config)
#         else:
#             logging.warning("The yaml config is empty!")
#         if config_2.data.data_module.data_loaders.batch_size > 1:
#             config_2.data.eval.crop_size = config_2.data.train.crop_size
#         assert config_1 == config_2