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
        self, feats, m_1_prev, z_prev, x_prev,
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
            feats["loop_mask"],
            initial_seqs = initial_seqs,
        )

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

        # m_1_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            x_prev,
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

        outputs["pair"] = z
        outputs["single"] = s

        # Predict 3D structure
        gt_angles = feats["torsion_angles_sin_cos"]

        outputs["sm"] = self.structure_module(
            s,
            z,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            initial_rigids=initial_rigids,
            initial_seqs=initial_seqs,
            loop_mask=feats["loop_mask"],
            gt_angles=gt_angles,
        )

        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]
        outputs["final_aatype"] = outputs["sm"]["aatype_"][-1]
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

            if "backbone_rigid_tensor_7s" in feats:
                gt_aff_7s = feats["backbone_rigid_tensor_7s"] # [*, N, 7]
                gt_aff_7s[..., 4:] = gt_aff_7s[..., 4:] * (
                    1.0 / self.structure_module.trans_scale_factor
                )
                identity_rigid_7s = torch.zeros_like(gt_aff_7s)
                identity_rigid_7s[..., 0] = 1
                loop_mask_expand = feats["loop_mask"][..., None].expand_as(gt_aff_7s) # [*, N, 7]
                # only predict the loop
                initial_rigids = loop_mask_expand * identity_rigid_7s + (1 - loop_mask_expand) * gt_aff_7s
                initial_rigids = Rigid.from_tensor_7(initial_rigids)
                if x_prev is None:
                    # [*, N, 37, 3]
                    x_prev = feats["all_atom_positions"]
                    x_prev_zero = torch.zeros_like(x_prev)
                    loop_mask_expand = feats["loop_mask"][..., None, None].expand_as(x_prev) # [*, N, 37, 3]
                    x_prev = loop_mask_expand * x_prev_zero + (1 - loop_mask_expand) * x_prev
            else:
                initial_rigids = None

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
                    initial_rigids=None,
                    initial_seqs=seqs_prev,
                    _recycle=(num_iters > 1)
                )
                outputs.update(self.aux_heads(outputs))
                recycle_outputs.append(outputs)

        # Run auxiliary heads
        outputs = copy.copy(outputs)
        # outputs.update(self.aux_heads(outputs)) done in the loop
        outputs["recycle_outputs"] = recycle_outputs

        return outputs
