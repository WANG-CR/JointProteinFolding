import math
import copy
from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from openfold.model.primitives import Linear

from openfold.utils.feats import (
    pseudo_beta_fn,
    atom14_to_atom37,
)
from openfold.model.embedders import (
    InputEmbedder,
    RecyclingEmbedder,
    GVPEmbedder,
)
from openfold.model.evoformer import EvoformerStack
from openfold.model.heads import AuxiliaryHeads
import openfold.np.residue_constants as residue_constants
from openfold.np.residue_constants import restype_num
from openfold.model.structure_module import StructureModule, SeqResnet
from openfold.model.gvp.gvp_gnn_encoder import GVPGNNEncoder
from openfold.utils.loss import check_inf_nan
from openfold.utils.feats import compute_pair_rbf
from openfold.utils.tensor_utils import tensor_tree_map, permute_final_dims
from openfold.utils.rigid_utils import Rotation, Rigid
import logging

def _nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)

class MultiLayerPerceptron(nn.Module):
    """ Partly taken from torchdrug.
        Note there is no batch normalization, activation or dropout in the last layer.
    """
    def __init__(self, 
        input_dim, 
        hidden_dims, 
        short_cut: bool = False, 
        batch_norm: bool = False, 
        activation: str = "gelu", 
        dropout: float = 0.,
    ):
        """
        Args:
            input_dim (int): input dimension
            hidden_dim (list of int): hidden dimensions
            short_cut (bool, optional): use short cut or not
            batch_norm (bool, optional): apply batch normalization or not
            activation (str or function, optional): activation function
            dropout (float, optional): dropout rate
        """
        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList([])
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        """"""
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden

class AlphaFoldInverse(nn.Module):
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
        super(AlphaFoldInverse, self).__init__()

        self.globals = config.globals
        self.config = config.model

        # Main trunk + structure module
        self.input_embedder = GVPEmbedder(
            **self.config["input_embedder"],
        )

        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )

        self.use_mlp = True
        self.mlp = MultiLayerPerceptron(
            input_dim=self.config["structure_module"]["c_s"],
            hidden_dims=[self.config["structure_module"]["c_s"], restype_num + 1],
        )

        # if "use_mlp" in self.config and self.config["use_mlp"]:
        #     self.mlp = MultiLayerPerceptron(
        #         input_dim=self.config["structure_module"]["c_s"],
        #         hidden_dims=[self.config["structure_module"]["c_s"], restype_num + 1],
        #     )
        #     self.use_mlp = True
        # else:
        #     self.structure_module = StructureModule(
        #         **self.config["structure_module"],
        #     )
        #     self.use_mlp = False
        
        self.aux_heads = None
        # AuxiliaryHeads(
            # self.config["heads"],
        # )

    def iteration(
        self, feats, m_1_prev, z_prev, x_prev, seqs_prev,
        initial_rigids=None,
        initial_seqs=None,
        _recycle=False,
        denoise_feats=None,
    ):
        """
        Required 'feats' keys w/ shape:
            - seq_mask: B, N
            - coords: B, N, 4, 3
            - ss_feat: B, N, C
            - residue_index: B, N
            - aatype: B, N (which is not used during forward)
        """
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        # Ignore it if you are using FP32.
        dtype = next(self.parameters()).dtype
        #print('model dtype', dtype)
        for k in feats:
            #print(feats[k].dtype)
            if not torch.is_tensor(feats[k]):
                continue
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        ##### Grab some data about the input: Disabled now

        # Prep some features
        
        seq_mask = feats["seq_mask"][..., -1]
        # logging.info(f'feats["coords"] size during iteration is {feats["coords"].shape}')
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        # logging.info(f"pair_mask size during iteration is {pair_mask.shape}")
        # # check_inf_nan(feats["coords"])
        # logging.info(f'feats["target_feat"] size during iteration is {feats["target_feat"].shape}')
        # logging.info(f'feats["residue_index"] size during iteration is {feats["residue_index"].shape}')
        # logging.info(f'feats["aatype"] size during iteration is {feats["aatype"].shape}')
        # logging.info(f"seq mask size during iteration is {feats['seq_mask'][..., -1].shape}")

        ## Calculate contact
        ## [*, N, N]
        contact = None
        pair_rbf = compute_pair_rbf(
            feats["coords"][..., 0],    # B, N, 4, 3
            feats["seq_mask"][..., -1],  # B, N
        )
        # check_inf_nan(pair_rbf)

        # pair_rbf = _nan_to_num(pair_rbf)
        # Initialize the seq and pair representations
        # m: [*, N, C_m]
        # z: [*, N, N, C_z]
        m, z = self.input_embedder( # use gvp or not
            feats["target_feat"][..., 0],
            feats["residue_index"][..., 0],
            pair_rbf,
            coords=feats["coords"][..., 0],
            mask=seq_mask
        )
        # m, z = _nan_to_num(m), _nan_to_num(z)

        # check_inf_nan([m,z])

        ######################################
        # Thats' all for predicting aatype
        # No recycle, so comment the rest out
        ######################################

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
        # m, z, s = _nan_to_num(m), _nan_to_num(z), _nan_to_num(s)

        # check_inf_nan([m, z, s])
        # print('m,z, s', m.dtype, z.dtype, s.dtype)

        # outputs["pair"] = z
        # outputs["single"] = s

        ###### Never compute latent
       
        # Predict 3D structure
        if self.use_mlp:
            logits = self.mlp(s)
            outputs["sm"] = {}
            outputs["sm"]["seqs_logits"] = logits.unsqueeze(0)
        else:
            outputs["sm"] = self.structure_module(
                s,
                z,
                feats["aatype"][..., 0],    # never mind that
                mask=feats["seq_mask"][..., -1].to(dtype=s.dtype),
                initial_rigids=initial_rigids,
                initial_seqs=initial_seqs,
                denoise_feats=denoise_feats,
            )

        # [*, N, C_m]
        m_1_prev = m

        # [*, N, N, C_z]
        z_prev = z

        # [*, N, 37, 3]
        x_prev = None   #outputs["final_atom_positions"]
        
        seqs_prev = None #outputs["sm"]["seqs"][-1]

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
        
        # denoise feats
        # it is empty if denoise_enabled is False
        denoise_feats = {}

        # Disable activation checkpointing for the first few recycling iters
        is_grad_enabled = torch.is_grad_enabled()
        self._disable_activation_checkpointing()

        # Main recycling loop
        # num_iters = batch["aatype"].shape[-1]
        recycle_outputs = []
        num_iters = 1

        # dummy disable recycling
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            
            # fetch_cur_batch = lambda t: t[..., cycle_no]
            # feats = tensor_tree_map(fetch_cur_batch, batch)
            feats = batch
            
            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    self._enable_activation_checkpointing()
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()
                
                # identity rigid.
                initial_rigids = None

                # Run the next iteration of the model
                # only run once
                outputs, m_1_prev, z_prev, x_prev, seqs_prev = self.iteration(
                    feats,
                    m_1_prev,
                    z_prev,
                    x_prev,
                    seqs_prev,
                    initial_rigids=initial_rigids,
                    initial_seqs=None,
                    _recycle=(num_iters > 1),
                    denoise_feats=denoise_feats,
                )
                # outputs.update(self.aux_heads(outputs))
                recycle_outputs.append(outputs)

        outputs = copy.copy(outputs)
        # outputs.update(self.aux_heads(outputs)) # done in the loop
        # outputs["recycle_outputs"] = recycle_outputs

        # logits = outputs["sm"]["seqs_logits"]
        return outputs

    def denoise_inference_forward(
        self,
        batch,
        sigmas_trans,
        sigmas_rot,
        step_lr=1e-4,
        rot_step_lr=None,
        n_steps_each=10,
        step_schedule="squared",
        init="identity",
    ):
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        seqs_prev = None
        if rot_step_lr is None:
            rot_step_lr = step_lr
        denoise_feats = {}

        # Disable activation checkpointing for the first few recycling iters
        is_grad_enabled = torch.is_grad_enabled()
        self._disable_activation_checkpointing()

        # Main recycling loop
        num_iters = batch["aatype"].shape[-1]
        assert num_iters == 1, "denoising rigid should use `no_recycle = 0`"

        cycle_no = 0
        # Select the features for the current recycling cycle
        fetch_cur_batch = lambda t: t[..., cycle_no]
        feats = tensor_tree_map(fetch_cur_batch, batch)

        initial_rigids = None

        if step_schedule == "linear":
            total_size_trans = step_lr * sigmas_trans.sum() / sigmas_trans.max() * n_steps_each
            total_size_rot = rot_step_lr * sigmas_rot.sum() / sigmas_rot.max() * n_steps_each
        elif step_schedule == "squared":
            total_size_trans = (step_lr * (sigmas_trans / sigmas_trans.min()) ** 2).sum() / sigmas_trans.max() * n_steps_each
            total_size_rot = (rot_step_lr * (sigmas_rot / sigmas_rot.min()) ** 2).sum() / sigmas_rot.max() * n_steps_each
        else:
            raise ValueError
        print('total step size (w.r.t. one training step): translation = %g, rotation = %g' % (total_size_trans, total_size_rot))

        with torch.set_grad_enabled(False):
            step_outputs = []
            for i, (used_sigmas_trans, used_sigmas_rot) in tqdm(
                enumerate(zip(sigmas_trans, sigmas_rot)),
                total=sigmas_trans.size(0),
                desc="Denoising rigid...",
            ):
                if step_schedule == "linear":
                    step_size_trans = step_lr * used_sigmas_trans
                    step_size_rot = rot_step_lr * used_sigmas_rot
                elif step_schedule == "squared":
                    step_size_trans = step_lr * (used_sigmas_trans / sigmas_trans.min()) ** 2
                    step_size_rot = rot_step_lr * (used_sigmas_rot / sigmas_rot.min()) ** 2
                print('current step size: translation = %g, rotation = %g' % (step_size_trans, step_size_rot))

                # perform `structure_module.n_blocks` steps of update using this step_size_trans
                n_steps_this = n_steps_each
                denoise_feats = {}
                denoise_feats["used_sigmas_trans"] = used_sigmas_trans
                denoise_feats["used_sigmas_rot"] = used_sigmas_rot
                denoise_feats["denoise_step_size"] = step_size_trans
                for step in range(n_steps_this):
                    if i > 0 or step > 0:
                        initial_rigids = outputs["sm"]["rigids"][-1]
                    else:
                        if init == "identity":
                            initial_rigids = Rigid.identity(
                                feats["aatype"].shape,
                                feats["target_feat"].dtype,
                                feats["target_feat"].device,
                                self.training,
                                fmt="quat",
                            ).to_tensor_7()
                        else:
                            raise ValueError

                    outputs, m_1_prev, z_prev, x_prev, seqs_prev = self.iteration(
                        feats,
                        None,
                        None,
                        None,
                        None,
                        initial_rigids=initial_rigids,
                        initial_seqs=None,
                        _recycle=(num_iters > 1),
                        denoise_feats=denoise_feats,
                    )
                    outputs.update(self.aux_heads(outputs))
                    step_outputs.append(outputs)

        # Run auxiliary heads
        outputs = copy.copy(outputs)
        # outputs.update(self.aux_heads(outputs)) done in the loop
        outputs["recycle_step"] = step_outputs

        return outputs
