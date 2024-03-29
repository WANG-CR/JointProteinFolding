from functools import partial

import torch
import logging
from openfold.data import data_transforms


@data_transforms.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x


def map_fn(fun, x):
    ensembles = [fun(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = torch.stack(
            [dict_i[feat] for dict_i in ensembles], dim=-1
        )
    return ensembled_dict


def nonensembled_transform_fns(mode_cfg):
    """Input pipeline data transformers that are not ensembled."""
    transforms = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.squeeze_features,
        data_transforms.get_backbone_coords,
        data_transforms.make_seq_mask,
        data_transforms.make_seq_feat,
        data_transforms.make_atom14_masks,
    ] 

    # print(f"mode_cfg bb only is: {mode_cfg.bb_only}")
    # logging.info(f"mode_cfg bb only is: {mode_cfg.bb_only}")
    if not mode_cfg.bb_only:

        transforms.extend(
            [
                data_transforms.make_atom14_positions,
                data_transforms.atom37_to_frames,
                data_transforms.atom37_to_torsion_angles(""),
                data_transforms.make_pseudo_beta(""),
                data_transforms.get_backbone_frames,
                data_transforms.get_chi_angles,
            ]
        )

    return transforms


def ensembled_transform_fns(common_cfg, mode_cfg, ensemble_seed):
    """Input pipeline data transformers that can be ensembled and averaged."""
    transforms = []

    crop_feats = dict(common_cfg.feat)

    if mode_cfg.fixed_size:
        transforms.append(data_transforms.select_feat(list(crop_feats)))
        transforms.append(
            data_transforms.random_crop_to_size(
                mode_cfg.crop,
                mode_cfg.crop_size,
                crop_feats,
                seed=ensemble_seed + 1,
            )
        )
        transforms.append(
            data_transforms.make_fixed_size(
                crop_feats,
                mode_cfg.crop_size,
            )
        )

    return transforms


def process_tensors_from_config(tensors, common_cfg, mode_cfg):
    """Based on the config, apply filters and transformations to the data."""

    ensemble_seed = torch.Generator().seed()

    def wrap_ensemble_fn(data, i):
        """Function to be mapped over the ensemble dimension."""
        d = data.copy()
        fns = ensembled_transform_fns(
            common_cfg, 
            mode_cfg, 
            ensemble_seed,
        )
        fn = compose(fns)
        d["ensemble_index"] = i
        return fn(d)

    nonensembled = nonensembled_transform_fns(mode_cfg)

    tensors = compose(nonensembled)(tensors)
    if("no_recycling_iters" in tensors):
        num_recycling = int(tensors["no_recycling_iters"])
    else:
        num_recycling = common_cfg.max_recycling_iters

    # the input tensor is wrapped with an additional dimension [no_recycle_iters + 1]
    tensors = map_fn(
        lambda x: wrap_ensemble_fn(tensors, x), torch.arange(num_recycling + 1)
    )
    return tensors
