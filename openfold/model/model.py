import torch
import torch.nn as nn
import copy

from openfold.utils.feats import (
    pseudo_beta_fn,
    atom14_to_atom37,
)
from openfold.model.embedders import (
    InputEmbedder,
    RecyclingEmbedder,
)
from openfold.model.evoformer import EvoformerStack
from openfold.model.heads import AuxiliaryHeads
import openfold.np.residue_constants as residue_constants
from openfold.model.structure_module import StructureModule
from openfold.utils.loss import compute_contact_ca
from openfold.utils.tensor_utils import dict_multimap, tensor_tree_map
from openfold.utils.rigid_utils import Rigid
from openfold.utils.loss import check_inf_nan

def _nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)

class AlphaFold(nn.Module):
    """
    Alphafold 2.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(AlphaFold, self).__init__()

        self.globals = config.globals
        self.config = config.model

        # Main trunk + structure module
        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"],
            residue_emb_cfg=self.config["residue_emb"],
            residue_attn_cfg=self.config["residue_attn"],
        )
        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )

        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )
        self.structure_module = StructureModule(
            **self.config["structure_module"],
        )

        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )

    def iteration(
        self, feats, m_1_prev, z_prev, x_prev, seqs_prev,
        initial_rigids=None,
        initial_seqs=None,
        _recycle=True,
    ):
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        n = feats["target_feat"].shape[-2]

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        # Initialize the seq and pair representations
        # m: [*, N, C_m]
        # z: [*, N, N, C_z]
        m, z = self.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            seqs_prev,
            
        )
        # print(f"checking m, 1")

        # print(f"checking z, 2")
        if check_inf_nan([m,z]):
            m, z = _nan_to_num(m), _nan_to_num(z)
        # Initialize the recycling embeddings, if needs be
        if None in [m_1_prev, z_prev]:
            # [*, N, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N, N, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n, n, self.config.input_embedder.c_z),
                requires_grad=False,
            )

        if x_prev is None:
            # [*, N, 37, 3]
            x_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )

        # [*, N, 3]
        x_prev = pseudo_beta_fn(
            feats["aatype"], x_prev, None
        ).to(z.dtype)
        # possible leakage?

        if seqs_prev is None:
            # [*, N, 21]
            seqs_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.restype_num + 1),
                requires_grad=False,
            )
            seqs_prev[..., -1] = 1.0

        # m_1_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            x_prev,
            seqs_prev,
        )
        # If the number of recycling iterations is 0, skip recycling
        # altogether. We zero them this way instead of computing them
        # conditionally to avoid leaving parameters unused, which has annoying
        # implications for DDP training.
        if not _recycle:
            m_1_prev_emb = m_1_prev_emb * 0
            z_prev_emb = z_prev_emb * 0

        # [*, N, C_m]
        m = m + m_1_prev_emb

        # [*, N, N, C_z]
        z = z + z_prev_emb

        # Possibly prevents memory fragmentation
        del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb


        # Run sequence + pair embeddings through the trunk of the network
        # m: [*, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]
        m, z, s = self.evoformer(
            m,
            z,
            seq_mask=seq_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            chunk_size=self.globals.chunk_size,
            _mask_trans=self.config._mask_trans,
        )


        # print(f"checking s, 3")
        # print(f"checking z, 4")
        if check_inf_nan([s,z]):
            s, z = _nan_to_num(s), _nan_to_num(z)
        outputs["pair"] = z
        outputs["single"] = s


        # Predict 3D structure
        # gt_angles = feats["torsion_angles_sin_cos"]
        # Possible leakage

        outputs["sm"] = self.structure_module(
            s,
            z,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            initial_rigids=initial_rigids,
            initial_seqs=initial_seqs,
            # gt_angles=gt_angles,
        )

        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        if "atom37_atom_exists" in feats:
            outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]
        outputs["final_aatype"] = outputs["sm"]["aatype_"][-1]
        outputs["final_aatype_dist"] = outputs["sm"]["aatype_dist"][-1] 
        # Save embeddings for use during the next recycling iteration

        # [*, N, C_m]
        m_1_prev = m

        # [*, N, N, C_z]
        z_prev = z

        # [*, N, 37, 3]
        x_prev = outputs["final_atom_positions"]
        
        seqs_prev = outputs["sm"]["seqs"][-1]

        return outputs, m_1_prev, z_prev, x_prev, seqs_prev

    def _disable_activation_checkpointing(self):
        self.evoformer.blocks_per_ckpt = None

    def _enable_activation_checkpointing(self):
        self.evoformer.blocks_per_ckpt = (
            self.config.evoformer_stack.blocks_per_ckpt
        )

    def forward(self, batch):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        seqs_prev = None

        # Disable activation checkpointing for the first few recycling iters
        is_grad_enabled = torch.is_grad_enabled()
        self._disable_activation_checkpointing()

        # Main recycling loop
        num_iters = batch["aatype"].shape[-1]
        recycle_outputs = []

        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)
            initial_rigids = None
            #problem: initial rigids always None?
            
            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    self._enable_activation_checkpointing()
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev, seqs_prev = self.iteration(
                    feats,
                    m_1_prev,
                    z_prev,
                    x_prev,
                    seqs_prev,
                    initial_rigids=initial_rigids,
                    initial_seqs=None,
                    _recycle=(num_iters > 1)
                )
                outputs.update(self.aux_heads(outputs))
                recycle_outputs.append(outputs)

        # Run auxiliary heads
        outputs = copy.copy(outputs)
        # outputs.update(self.aux_heads(outputs)) done in the loop
        outputs["recycle_outputs"] = recycle_outputs

        return outputs
