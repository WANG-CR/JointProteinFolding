import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging
from openfold.model.primitives import Linear, LayerNorm
from openfold.utils.tensor_utils import one_hot
import ml_collections as mlc

class InputEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(
        self,
        tf_dim: int,
        c_z: int,
        c_m: int,
        relpos_k: int,
        residue_emb_cfg: mlc.ConfigDict,
        residue_attn_cfg: mlc.ConfigDict,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            c_z:
                Pair embedding dimension
            c_m:
                sequence embedding dimension
            relpos_k:
                Window size used in relative positional encoding
            mask_loop_type:
                mask whole loop region/certain loop region/none mask
        """
        super(InputEmbedder, self).__init__()

        self.tf_dim = tf_dim
        logging.info(f"tf_dim is {tf_dim}")
        self.c_z = c_z
        self.c_m = c_m
        if not residue_emb_cfg["enabled"]:
            self.linear_tf_z_i = Linear(tf_dim, c_z)
            self.linear_tf_z_j = Linear(tf_dim, c_z)
            self.linear_tf_m = Linear(tf_dim, c_m)

        # RPE stuff
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

        self.esm_model, _ = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        emb_input_dim = 1280
        self.linear_emb_m = Linear(emb_input_dim, c_m)
        self.linear_emb_z_i = Linear(emb_input_dim, c_z)
        self.linear_emb_z_j = Linear(emb_input_dim, c_z)
        self.tf_squeeze_flag = False

        self.residue_attn_cfg = residue_attn_cfg
        self.residue_emb_cfg = residue_emb_cfg
        # if self.residue_attn_cfg["enabled"]:
        #     attn_input_dim = self.residue_attn_cfg["c_emb"]
        #     self.linear_attn = Linear(attn_input_dim, c_z)
        if self.residue_emb_cfg["enabled"]:
            logging.warning(f"residue emb is set as: {self.residue_emb_cfg['usage']}")
        if not self.residue_emb_cfg["enabled"]:   
            logging.warning('residue emb cfg not enabled')  
        # self.is_lm_finetune = False
        self.is_lm_finetune = True

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

    def _mapping(self, seq):
        # mapping_esm_x=[1, 5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7, 3]
        mapping_esm_x=[5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7, 3]
        count = 0
        for i in range(seq.size(1)):
            seq[0,i] = mapping_esm_x[seq[0,i].to(torch.int)]
            if seq[0,i]==1 and count == 0:
                count += 1 
                seq[0,i]=2
        return count

    def _convert_to_esm_index(self, tf: torch.Tensor):
        tf_esm = torch.argmax(tf, dim=-1)
        count = self._mapping(tf_esm)

        # add bos and eos to the sequence
        cls = torch.tensor([[0]]).to(tf_esm.device)
        tf_esm = torch.cat((cls,tf_esm),1)
        eos = torch.tensor([[2-count]]).to(tf_esm.device)
        tf_esm = torch.cat((tf_esm,eos),1)
        return tf_esm


    def residue_encoding(self, tf: torch.Tensor):
        if not self.is_lm_finetune:
            for params in self.esm_model.parameters():
                params.requires_grad = False
            # logging.info(f"freezing language model parameters")
            
        # logging.info("feeding tf esm into esm encoder")
        self.tf_squeeze_flag = False
        if tf.ndim == 2:
            tf = tf.clone().unsqueeze(0)
            self.tf_squeeze_flag = True

        # calculating Heavy Chain embedding
        tf_H = self._convert_to_esm_index(tf)
        if not self.is_lm_finetune:
            with torch.no_grad():
                results = self.esm_model(tf_H, repr_layers=[33], need_head_weights=True, return_contacts=True)
        elif self.is_lm_finetune:
            results = self.esm_model(tf_H, repr_layers=[33], need_head_weights=True, return_contacts=True)

        tf_H = results["representations"][33]
        tf_H = tf_H[:, 1:-1 ,:]

        # shape [batchSize, nLayer, nHead, nRes, nRes]
        att_H = results["attentions"][... , 1:-1 ,1:-1]
        att_H = att_H.mean(1).mean(1)

        # shape [batchsize, Nres, Nres]
        contact = results["contacts"]

        # shape [batchsize, Nres+2, 33]
        logits = results["logits"][:, 1:-1 ,:].mean(-1)

        if self.tf_squeeze_flag:
            tf_H.squeeze_(0)

        return {
            "tf": tf_H,
            "attention": att_H,
            "contact": contact, 
            "logits": logits,
        }


    def forward(
        self,
        tf: torch.Tensor,
        ri: torch.Tensor,
        initial_seqs: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf:
                "target_feat" features of shape [*, N_res, 21]
            ri:
                "residue_index" features of shape [*, N_res]

        Returns:
            tf_m:
                "sequence embedding" features of [*, N_res, C_m] 
            pair_emb:
                "pair embedding" features [*, N_res, N_res, C_z] 
        """

        tf_emb_i = None
        tf_emb_j = None
        tf_m = None

        if not self.residue_emb_cfg["enabled"]:
            # [*, N_res, c_z]
            tf_emb_i = self.linear_tf_z_i(tf)
            tf_emb_j = self.linear_tf_z_j(tf)
            # [*, N_res, c_m]
            tf_m = self.linear_tf_m(tf)
        residue_emb_m = None

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
            
            representation = self.residue_encoding(tf)
            # tf_esm, attn = self.residue_encoding(tf, chain_index_copy)
            tf_esm = representation["tf"]
            if self.residue_emb_cfg["usage"] != "msa":                      
                residue_emb_z_i = self.linear_emb_z_i(tf_esm) # [*, N_res, c_z]
                residue_emb_z_j = self.linear_emb_z_j(tf_esm) # [*, N_res, c_z]
                residue_emb_m = self.linear_emb_m(tf_esm) # [*, N_res, c_m]
            
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
            attn = representation["attention"].unsqueeze(-1)
            contact = representation["contact"].unsqueeze(-1)
            logits = representation["logits"].unsqueeze(-1).unsqueeze(-1)
            assert attn is not None
            # attn_feat = self.linear_attn(attn)
            pair_emb = pair_emb + attn * 0 + contact * 0 + logits * 0.0
            # attn_feat = self.linear_attn(attn)
            # pair_emb = pair_emb + attn_feat

        return tf_m, pair_emb


from openfold.model.gvp.gvp_encoder import GVPEncoder

# following gvp paper settings
gvp_gnn_args = {
    # GVPGraphEmbedding
    'node_hidden_dim_scalar': 100,
    'node_hidden_dim_vector': 16,
    'edge_hidden_dim_scalar': 32,
    'edge_hidden_dim_vector': 1,
    'top_k_neighbors': 30, 
    # GVPConv
    'dropout': 0.,
    'num_encoder_layers': 3,    
}

class GVPEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(
        self,
        tf_dim: int,
        c_z: int,
        c_m: int,
        relpos_k: int,
        i_z: int = 400, 
        max_len: int = 500,
        use_gvp: bool = True,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            c_z:
                Pair embedding dimension
            c_m:
                sequence embedding dimension
            relpos_k:
                Window size used in relative positional encoding
            mask_loop_type:
                mask whole loop region/certain loop region/none mask
        """
        super(GVPEmbedder, self).__init__()

        self.use_gvp = use_gvp

        self.tf_dim = tf_dim
        self.c_z = c_z
        self.c_m = c_m
        self.i_z = i_z

        if self.use_gvp:
            print("will use GVP to featurize node embedding")
            self.gvp_embedding = GVPEncoder(**gvp_gnn_args)
            self.post_gvp = Linear(gvp_gnn_args["node_hidden_dim_scalar"], self.c_m)

        # RPE stuff
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

        self.linear_contact = Linear(i_z, c_z) 

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
        pair_rbf: torch.Tensor,
        coords: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf:
                "target_feat" features of shape [*, N_res, 21]
            ri:
                "residue_index" features of shape [*, N_res]
            pair_rbf:
                "pair_rbf" features of shape [*, N_res, N_res, i_z]

        Returns:
            tf_m:
                "sequence embedding" features of [*, N_res, C_m] 
            pair_emb:
                "pair embedding" features [*, N_res, N_res, C_z] 
        """
        # _dtype = torch.get_default_dtype()
        _dtype = tf.dtype
        # oh = oh.type(torch.float32)
        pair_emb = self.relpos(ri.type(_dtype))
        mask = mask.bool()

        # [*, N_res, c_m]
        if self.use_gvp:
            assert coords is not None and mask is not None
            node_s, node_v = self.gvp_embedding(coords.type(_dtype), mask, ~mask)    # s: B x T x C
            # node_s, node_v = self.gvp_embedding(coords.type(torch.float32), mask, ~mask)    # s: B x T x C
            tf_m = self.post_gvp(node_s).type(pair_emb.dtype) + node_v.mean().type(pair_emb.dtype) * 0
        else:
            tf_m = self.idx_embedding(ri).type(pair_emb.dtype)

        assert pair_rbf is not None
        pair_rbf = pair_rbf.type(pair_emb.dtype)
        
        contact_emb = self.linear_contact(pair_rbf).type(pair_emb.dtype)
        pair_emb = contact_emb + pair_emb

        return tf_m, pair_emb


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
                Seq channel dimension
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

        # seqs_prev
        self.linear_seqs = Linear(21, self.c_m)
        self.layer_norm_seqs = LayerNorm(self.c_m)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        seqs: torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                Sequence embedding. [*, N_res, C_m]
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
        m_update = self.layer_norm_m(m) + self.layer_norm_seqs(self.linear_seqs(seqs))

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
