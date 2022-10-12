# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import torch
import torch.nn as nn
from typing import Dict

from openfold.np import protein
import openfold.np.residue_constants as rc
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import (
    batched_gather,
    one_hot,
    tree_map,
    tensor_tree_map,
)


def rbf(values, v_min=2., v_max=22., n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    v_expand = torch.unsqueeze(values, -1)
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)

def rbf_from_two_array(a, b):
    """
    Return:
        (B, N, N, i_z)
    """
    dist = torch.sqrt(torch.sum((a[:, :, None, :] - b[:, None, :, :])**2, dim=-1) + 1e-6)
    return rbf(dist)

def compute_pair_rbf(
    backbone_positions: torch.Tensor,
    atom_mask: torch.Tensor,
):
    """
        Compute the contact map of alpha carbons
        Args:
            backbone_positions: (B, N, 4, 3)
            atom_mask: (B, N, )
        Returns:
            [*, N, N, i_z] pair rbf feature
    """
    X = backbone_positions
    b = X[:,:,1,:] - X[:,:,0,:]
    c = X[:,:,2,:] - X[:,:,1,:]
    a = torch.cross(b, c, dim=-1)

    Cb = - 0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:,:,1,:]
    Ca = X[:,:,1,:]
    N = X[:,:,0,:]
    C = X[:,:,2,:]
    O = X[:,:,3,:]
    # print(f"Ca is {Ca[0, 0, ...]}")
    # print(f"N is {N[0, 0, ...]}")
    # print(f"C is {C[0, 0, ...]}")
    # print(f"O is {O[0, 0, ...]}")
    RBF_all = []
    RBF_all.append(rbf_from_two_array(Ca, Ca)) #Ca-Ca
    RBF_all.append(rbf_from_two_array(N, N)) #N-N
    RBF_all.append(rbf_from_two_array(C, C)) #C-C
    RBF_all.append(rbf_from_two_array(O, O)) #O-O
    RBF_all.append(rbf_from_two_array(Cb, Cb)) #Cb-Cb
    RBF_all.append(rbf_from_two_array(Ca, N)) #Ca-N
    RBF_all.append(rbf_from_two_array(Ca, C)) #Ca-C
    RBF_all.append(rbf_from_two_array(Ca, O)) #Ca-O
    RBF_all.append(rbf_from_two_array(Ca, Cb )) #Ca-Cb
    RBF_all.append(rbf_from_two_array(N, C )) #N-C
    RBF_all.append(rbf_from_two_array(N, O )) #N-O
    RBF_all.append(rbf_from_two_array(N, Cb )) #N-Cb
    RBF_all.append(rbf_from_two_array(Cb, C )) #Cb-C
    RBF_all.append(rbf_from_two_array(Cb, O )) #Cb-O
    RBF_all.append(rbf_from_two_array(O, C )) #O-C
    RBF_all.append(rbf_from_two_array(N, Ca )) #N-Ca
    RBF_all.append(rbf_from_two_array(C, Ca )) #C-Ca
    RBF_all.append(rbf_from_two_array(O, Ca )) #O-Ca
    RBF_all.append(rbf_from_two_array(Cb, Ca )) #Cb-Ca
    RBF_all.append(rbf_from_two_array(C, N )) #C-N
    RBF_all.append(rbf_from_two_array(O, N )) #O-N
    RBF_all.append(rbf_from_two_array(Cb, N )) #Cb-N
    RBF_all.append(rbf_from_two_array(C, Cb )) #C-Cb
    RBF_all.append(rbf_from_two_array(O, Cb )) #O-Cb
    RBF_all.append(rbf_from_two_array(C, O )) #C-O

    RBF_all = torch.cat(RBF_all, dim=-1)    # 25 x 16 = 400

    # RBF_all = mask...
    return RBF_all


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    is_gly = aatype == rc.restype_order["G"]
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    pseudo_beta = torch.where(
        is_gly[..., None].expand(*((-1,) * len(is_gly.shape)), 3),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_masks[..., ca_idx],
            all_atom_masks[..., cb_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


def atom14_to_atom37(atom14, batch):
    atom37_data = batched_gather(
        atom14,
        batch["residx_atom37_to_atom14"],
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )

    atom37_data = atom37_data * batch["atom37_atom_exists"][..., None]

    return atom37_data


def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
):
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat(
        [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
    )

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 14, 4, 4]
    default_4x4 = default_frames[aatype, ...]

    # [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions
