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

import torch
import torch.nn as nn
from typing import Optional, Tuple
import ml_collections as mlc

from openfold.model.primitives import Linear, LayerNorm
from openfold.utils.tensor_utils import one_hot


class InputEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        relpos_k: int,
        mask_loop_type: float,
        residue_emb_cfg: mlc.ConfigDict,
        residue_attn_cfg: mlc.ConfigDict,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(InputEmbedder, self).__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m
        self.mask_loop_type = mask_loop_type
        self.residue_emb_cfg = residue_emb_cfg
        self.residue_attn_cfg = residue_attn_cfg

        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)

        # residue_emb enabled
        if self.residue_emb_cfg["enabled"]:
            if self.residue_emb_cfg["usage"] != "msa":
                emb_input_dim = self.residue_emb_cfg["c_emb"] * self.residue_emb_cfg["num_emb_feats"]
                self.linear_emb_m = Linear(emb_input_dim, c_m)
                self.linear_emb_z_i = Linear(emb_input_dim, c_z)
                self.linear_emb_z_j = Linear(emb_input_dim, c_z)
            else: # msa case
                emb_input_dim = self.residue_emb_cfg["c_emb"]
                self.linear_emb_m = nn.ModuleList(
                    [Linear(emb_input_dim, c_m) for i in range(self.residue_emb_cfg["num_emb_feats"])]
                )
        # residue_attn enabled
        if self.residue_attn_cfg["enabled"]:
            attn_input_dim = self.residue_attn_cfg["c_emb"]
            self.linear_attn = Linear(attn_input_dim, c_z)
            # self.layer_norm_attn = LayerNorm(c_z)
            # self.relu_attn = nn.ReLU()

        # RPE stuff
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def relpos(self, ri: torch.Tensor):
        """
        Computes relative positional encodings

        Implements Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N]
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        )
        oh = one_hot(d, boundaries).type(ri.dtype)
        return self.linear_relpos(oh)

    def forward(
        self,
        tf: torch.Tensor,
        ri: torch.Tensor,
        msa: torch.Tensor,
        emb: torch.Tensor,
        loop_mask: torch.Tensor,
        attn: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf:
                "target_feat" features of shape [*, N_res, tf_dim]
            ri:
                "residue_index" features of shape [*, N_res]
            msa:
                "msa_feat" features of shape [*, N_clust, N_res, msa_dim]
            emb:
                "residue_emb" features of shape [*, N_model, N_res, emb_dim] from pre-trained language models.
            attn:
                "residue_attn" features of shape [*, N_res, N_res, attn_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding
            residue_emb:
                [*, N_res, C_m] or [*, N_model, N_res, c_m], updated residue embedding

        """
        # mask loop type for loop design
        if self.mask_loop_type:
            tf_unk = torch.zeros_like(tf)
            tf_unk[..., -1] = 1.0
            loop_mask_expand = loop_mask[..., None].expand_as(tf_unk)
            tf = loop_mask_expand * tf_unk + (1 - loop_mask_expand) * tf

        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, c_m]
        tf_m = self.linear_tf_m(tf)
        
        residue_emb_m = None
        
        # residue_emb enabled        
        if self.residue_emb_cfg["enabled"]:
            # initialize functions to digest pre-trained residue embeddings
            fn_cat = lambda i, j, emb_z_i, emb_z_j, m, emb_m: (i + emb_z_i, j + emb_z_j, m + emb_m)
            fn_replace = lambda i, j, emb_z_i, emb_z_j, m, emb_m: (emb_z_i, emb_z_j, emb_m)
            fn_msa = lambda i, j, emb_z_i, emb_z_j, m, emb_m: (i, j, m)
            fn_dict = {
                "cat": fn_cat,
                "replace": fn_replace,
                "msa": fn_msa
            }

            emb = torch.unbind(emb, dim=-3) # [[*, N_res, emb_dim]...]
            if len(emb) != self.residue_emb_cfg["num_emb_feats"]:
                raise ValueError(
                    f"""
                     The number of residue embedding {len(emb)} 
                     does not match the config {self.residue_emb_cfg["num_emb_feats"]}
                     """
                )

            if self.residue_emb_cfg["usage"] != "msa":          
                emb = torch.cat(emb, dim=-1) # [*, N_res, emb_dim * N_model]
                
                residue_emb_z_i = self.linear_emb_z_i(emb) # [*, N_res, c_z]
                residue_emb_z_j = self.linear_emb_z_j(emb) # [*, N_res, c_z]
                residue_emb_m = self.linear_emb_m(emb) # [*, N_res, c_m]
            
            else:
                residue_emb_z_i = None
                residue_emb_z_j = None
                
                # [[*, N_res, emb_dim]...]
                # Warning: Note that this operation is order-sensitve!
                per_model_residue_emb = []
                for i, layer in enumerate(self.linear_emb_m):
                    per_model_residue_emb.append(layer(emb[i]))
                    
                # [*, N_model, N_res, c_m]
                residue_emb_m = torch.stack(per_model_residue_emb, dim=-3)
              
            tf_emb_i, tf_emb_j, tf_m = fn_dict[self.residue_emb_cfg["usage"]](
                tf_emb_i, tf_emb_j, 
                residue_emb_z_i, residue_emb_z_j,
                tf_m, residue_emb_m
            )

        # [*, N_res, N_res, c_z]
        pair_emb = tf_emb_i[..., None, :] + tf_emb_j[..., None, :, :]
        pair_emb = pair_emb + self.relpos(ri.type(pair_emb.dtype))
        
        # residue_attn enabled
        if self.residue_attn_cfg["enabled"]:
            assert attn is not None
            attn_feat = self.linear_attn(attn)
            pair_emb = pair_emb + attn_feat

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = (
            tf_m
            .unsqueeze(-3)
            .expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))
        )
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb, residue_emb_m


class Ca_Aware_Embedder(nn.Module):
    """
    Embeds the output structure of a structure block to bias the attention map.

    Adapted from Algorithm 32.
    """

    def __init__(
        self,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(Ca_Aware_Embedder, self).__init__()

        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf

        self.linear = Linear(self.no_bins, self.c_z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:
                [*, N_res, 3] predicted C_alpha coordinates
        Returns:
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )

        # This squared method might become problematic in FP16 mode.
        # I'm using it because my homegrown method had a stubborn discrepancy I
        # couldn't find in time.
        squared_bins = bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        
        # [*, N_res, N_res, 1]
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )

        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)

        # [*, N, N, C_z]
        d = self.linear(d)
        z_update = d

        return z_update


class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(RecyclingEmbedder, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf

        self.linear = Linear(self.no_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )

        # [*, N, C_m]
        m_update = self.layer_norm_m(m)

        # This squared method might become problematic in FP16 mode.
        # I'm using it because my homegrown method had a stubborn discrepancy I
        # couldn't find in time.
        squared_bins = bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )

        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)

        # [*, N, N, C_z]
        d = self.linear(d)
        z_update = d + self.layer_norm_z(z)

        return m_update, z_update


class TemplateAngleEmbedder(nn.Module):
    """
    Embeds the "template_angle_feat" feature.

    Implements Algorithm 2, line 7.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
        """
        super(TemplateAngleEmbedder, self).__init__()

        self.c_out = c_out
        self.c_in = c_in

        self.linear_1 = Linear(self.c_in, self.c_out, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.c_out, self.c_out, init="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N_templ, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_templ, N_res, C_out] embedding
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x


class TemplatePairEmbedder(nn.Module):
    """
    Embeds "template_pair_feat" features.

    Implements Algorithm 2, line 9.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:

            c_out:
                Output channel dimension
        """
        super(TemplatePairEmbedder, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        # Despite there being no relu nearby, the source uses that initializer
        self.linear = Linear(self.c_in, self.c_out, init="relu")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, C_in] input tensor
        Returns:
            [*, C_out] output tensor
        """
        x = self.linear(x)

        return x


class ExtraMSAEmbedder(nn.Module):
    """
    Embeds unclustered MSA sequences.

    Implements Algorithm 2, line 15
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_out:
                Output channel dimension
        """
        super(ExtraMSAEmbedder, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.linear = Linear(self.c_in, self.c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_extra_seq, N_res, C_in] "extra_msa_feat" features
        Returns:
            [*, N_extra_seq, N_res, C_out] embedding
        """
        x = self.linear(x)

        return x
