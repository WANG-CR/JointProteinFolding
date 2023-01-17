# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as T
import logging
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from openfold.data.data_transforms import make_atom14_masks
from openfold.utils.feats import (
    pseudo_beta_fn,
    atom14_to_atom37,
)
from openfold.np import residue_constants
from openfold.utils.loss import compute_predicted_aligned_error, compute_tm, categorical_lddt
from openfold.utils.tensor_utils import tensor_tree_map
from torch import nn
from torch.nn import LayerNorm

# import esm
# from esm import Alphabet
from openfold.model.esm.trunk import FoldingTrunk, FoldingTrunkConfig, StructureModuleConfig
from openfold.model.esm.misc import (
    batch_encode_sequences,
    collate_dense_tensors,
    output_to_pdb,
)


@dataclass
class ESMFoldConfig:
    trunk: T.Any = FoldingTrunkConfig()
    lddt_head_hid_dim: int = 128
    lm: str = "esm2_t33_650M_UR50D"

def constructConfigFromYAML(config):
    assert config.model.structure_module.no_blocks == config.model.evoformer_stack.no_blocks
    structure_module = StructureModuleConfig(
        # c_m = config.globals.c_m_structure,
        c_s = config.globals.c_m_structure,
        c_z = config.globals.c_z_structure,
    )
    trunk = FoldingTrunkConfig(
        num_blocks=config.model.structure_module.no_blocks,
        sequence_state_dim = config.globals.c_m,
        pairwise_state_dim = config.globals.c_z,
        sequence_head_width = config.model.evoformer_stack.c_hidden_seq_att,
        pairwise_head_width = config.model.evoformer_stack.c_hidden_pair_att,
        position_bins = config.model.input_embedder.relpos_k,
        dropout = config.model.evoformer_stack.seq_dropout,
        structure_module = structure_module,
        )
    lm = config.globals.lm_name
    print(trunk)
    return ESMFoldConfig(trunk = trunk, lm=lm)

class ESMFold(nn.Module):
    def __init__(self, esmfold_config=None, using_fair=False, track_seq_states=False, **kwargs):
        super().__init__()

        self.cfg = esmfold_config if esmfold_config else ESMFoldConfig(**kwargs)
        cfg = self.cfg
        self.using_fair = using_fair
        self.track_seq_states = track_seq_states
        self.distogram_bins = 64
        print(f"using fair 1")
        if using_fair:
            self.esm, self.esm_dict = torch.hub.load("facebookresearch/esm:main", "esm2_t36_3B_UR50D")
            # self.esm, self.esm_dict = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
            print(f"using fair 2")
        else:
            self.esm, self.esm_dict = torch.hub.load("facebookresearch/esm:main", cfg.lm)
            logging.info(f"using Language model: {cfg.lm}")
        
        # require_grad = False
        self.esm.requires_grad_(False)
        self.esm.half()

        self.esm_feats = self.esm.embed_dim
        self.esm_attns = self.esm.num_layers * self.esm.attention_heads
        self.register_buffer("af2_to_esm", ESMFold._af2_to_esm(self.esm_dict))
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm.num_layers + 1))

        c_s = cfg.trunk.sequence_state_dim
        c_z = cfg.trunk.pairwise_state_dim
        
        if using_fair:
            c_s_sturcture = cfg.trunk.structure_module.c_s
        else:
            # c_s_sturcture = cfg.trunk.structure_module.c_m
            c_s_sturcture = cfg.trunk.structure_module.c_s

        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

        # 0 is padding, N is unknown residues, N + 1 is mask.
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        if (isinstance(self.cfg.trunk, dict)):
            self.trunk = FoldingTrunk(**cfg.trunk)
        else:
            self.trunk = FoldingTrunk(**asdict(cfg.trunk))

        self.distogram_head = nn.Linear(c_z, self.distogram_bins)
        self.ptm_head = nn.Linear(c_z, self.distogram_bins)
        
        if using_fair:
            self.lm_head = nn.Linear(c_s, self.n_tokens_embed)
        else:
            self.lm_head = nn.Linear(c_s, self.n_tokens_embed-2)

        self.lddt_bins = 50
        self.lddt_head = nn.Sequential(
            nn.LayerNorm(c_s_sturcture),
            nn.Linear(c_s_sturcture, cfg.lddt_head_hid_dim),
            nn.Linear(cfg.lddt_head_hid_dim, cfg.lddt_head_hid_dim),
            nn.Linear(cfg.lddt_head_hid_dim, 37 * self.lddt_bins),
        )

    @staticmethod
    # def _af2_to_esm(d: Alphabet):
    def _af2_to_esm(d):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [d.get_idx(v) for v in residue_constants.restypes_with_x]
        return torch.tensor(esm_reorder)

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        # print(f"aa masked is {aa}")
        return self.af2_to_esm[aa]

    def _compute_language_model_representations(self, esmaa: torch.Tensor) -> torch.Tensor:
        """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
        batch_size = esmaa.size(0)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = esmaa.new_full((batch_size, 1), bosi)
        eos = esmaa.new_full((batch_size, 1), self.esm_dict.padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

        res = self.esm(
            esmaa,
            repr_layers=range(self.esm.num_layers + 1),
            need_head_weights=False,
        )
        esm_s = torch.stack([v for _, v in sorted(res["representations"].items())], dim=2)
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        return esm_s

    def _mask_inputs_to_esm(self, esmaa, pattern):
        new_esmaa = esmaa.clone()
        new_esmaa[pattern == 1] = self.esm_dict.mask_idx
        return new_esmaa

    # def forward(
    #     self,
    #     aa: torch.Tensor,
    #     mask: T.Optional[torch.Tensor] = None,
    #     residx: T.Optional[torch.Tensor] = None,
    #     masking_pattern: T.Optional[torch.Tensor] = None,
    #     num_recycles: T.Optional[int] = None,
    # ):
    def forward(
        self,
        batch,
    ):
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            aa (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """
        ## how to input
        num_recycles = batch["aatype"].shape[-1]
        fetch_cur_batch = lambda t: t[..., 0]
        feats = tensor_tree_map(fetch_cur_batch, batch)
        aa = feats["aatype"]
        mask = feats["seq_mask"]
        residx = feats["residue_index"]
        masking_pattern = None
        # the no_recycle is sampled from a uniform distribution
        # print(f"aatype shape is { batch['aatype'].shape}")
        # print(f"aa is {aa}")
        # print(f"mask is {mask}")
        # print(f"residx is {residx}")
        if "masking_pattern" in feats:
            masking_pattern = feats["masking_pattern"]

        if mask is None:
            mask = torch.ones_like(aa)

        B = aa.shape[0]
        L = aa.shape[1]
        device = aa.device

        if residx is None:
            residx = torch.arange(L, device=device).expand_as(aa)

        # === ESM ===
        esmaa = self._af2_idx_to_esm_idx(aa, mask)

        if masking_pattern is not None:
            esmaa = self._mask_inputs_to_esm(esmaa, masking_pattern)

        esm_s = self._compute_language_model_representations(esmaa)

        # Convert esm_s to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.
        esm_s = esm_s.to(self.esm_s_combine.dtype)

        esm_s = esm_s.detach()

        # === preprocessing ===
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)

        s_s_0 = self.esm_s_mlp(esm_s)
        s_z_0 = s_s_0.new_zeros(B, L, L, self.cfg.trunk.pairwise_state_dim)

        s_s_0 += self.embedding(aa)

        structure: dict = self.trunk(s_s_0, s_z_0, aa, residx, mask, no_recycles=num_recycles)
        # Documenting what we expect:
        structure = {
            k: v
            for k, v in structure.items()
            if k
            in [
                "s_z",
                "s_s",
                "frames",
                "sidechain_frames",
                "unnormalized_angles",
                "angles",
                "positions",
                "frames",
                "sidechain_frames",
                # "states",
                "seqs_logits",
                "singles",
            ]
        }
        # question: we need to change s_z, s_s keys
        
        ############# No.1 fape loss

        ############# No.2 distogram loss
        ## question: do they need only taking last dimension of structure? [-1]
        disto_logits = self.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        ############# No.3 lm loss
        lm_logits = self.lm_head(structure["s_s"])
        if self.using_fair:
            seqResNet_logits = structure["seqs_logits"]
        # print(f"lm_logits shape is : {lm_logits.shape}")
        # print(f"seqResNet_logits shape is : {seqResNet_logits.shape}")
            structure["lm_logits"] = (lm_logits + seqResNet_logits)[-1]
        else:
            structure["lm_logits"] = lm_logits[-1]

        ############# No.4 plddt loss
        # need final_atom_positions
        structure["aatype"] = aa
        make_atom14_masks(structure)
        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= mask.unsqueeze(-1)
        structure["residue_index"] = residx 
        structure["final_atom_positions"] = atom14_to_atom37(structure["positions"][-1], structure)
        # for k, v in structure.items():
        #     print(f"structure contains key: {k}")
        lddt_head = self.lddt_head(structure["singles"]).reshape(
            structure["singles"].shape[0], B, L, -1, self.lddt_bins
        )
        # print(f"lddt_head shape is {lddt_head.shape}")
        # print(f"lddt_head last dimension shape is {lddt_head[-1].shape}")
        structure["lddt_head"] = lddt_head
        plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
        structure["plddt"] = (
            100 * plddt
        )  # we predict plDDT between 0 and 1, scale to be between 0 and 100.
        structure["mean_plddt"] = (structure["plddt"] * structure["atom37_atom_exists"]).sum(
            dim=(1, 2)
        ) / structure["atom37_atom_exists"].sum(dim=(1, 2))
        # shape is [..., 37]
        # print(f"plddt shape is {plddt.shape}")
        ca_pos = residue_constants.atom_order["CA"]
        structure["lddt_logits"] = (100 * plddt[..., : , ca_pos])
        # print(f"lddt logits shape is {structure['lddt_logits'].shape}")
        # print(f"lddt logits content is {structure['lddt_logits']}")
        ############# No.4 ptm loss
        ptm_logits = self.ptm_head(structure["s_z"])
        # shape (batch_size, seq_len, seq_len, self.distogram_bins)

        seqlen = mask.type(torch.int64).sum(1)
        structure["ptm_logits"] = ptm_logits
        structure["final_affine_tensor"] = structure["frames"][-1]
        structure["ptm"] = torch.stack(
            [
                compute_tm(
                    batch_ptm_logits[None, :sl, :sl], max_bins=31, no_bins=self.distogram_bins
                )
                for batch_ptm_logits, sl in zip(ptm_logits, seqlen)
            ]
        )
        # print(f"ptm shape is {structure['ptm'].shape}")
        structure.update(
            compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.distogram_bins)
        )

        return structure

    def forward4infer(
        self,
        aatype,
        mask,
        residx=None,
        masking_pattern=None,
        num_recycles=None,
    ):
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            aa (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """
        ## how to input
        aa = aatype

        # the no_recycle is sampled from a uniform distribution
        # print(f"aatype shape is { batch['aatype'].shape}")
        # print(f"aa is {aa}")
        # print(f"mask is {mask}")
        # print(f"residx is {residx}")

        if mask is None:
            mask = torch.ones_like(aa)

        B = aa.shape[0]
        L = aa.shape[1]
        device = aa.device

        if residx is None:
            residx = torch.arange(L, device=device).expand_as(aa)

        # === ESM ===
        esmaa = self._af2_idx_to_esm_idx(aa, mask)

        if masking_pattern is not None:
            esmaa = self._mask_inputs_to_esm(esmaa, masking_pattern)

        esm_s = self._compute_language_model_representations(esmaa)

        # Convert esm_s to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.
        esm_s = esm_s.to(self.esm_s_combine.dtype)

        esm_s = esm_s.detach()

        # === preprocessing ===
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)

        s_s_0 = self.esm_s_mlp(esm_s)
        s_z_0 = s_s_0.new_zeros(B, L, L, self.cfg.trunk.pairwise_state_dim)

        s_s_0 += self.embedding(aa)

        structure: dict = self.trunk(s_s_0, s_z_0, aa, residx, mask, no_recycles=num_recycles, track_seq_states = self.track_seq_states)
        # Documenting what we expect:
        structure = {
            k: v
            for k, v in structure.items()
            if k
            in [
                "s_z",
                "s_s",
                "frames",
                "sidechain_frames",
                "unnormalized_angles",
                "angles",
                "positions",
                "frames",
                "sidechain_frames",
                # "states",
                "seqs_logits",
                "singles",
            ]
        }
        # question: we need to change s_z, s_s keys
        
        ############# No.1 fape loss

        ############# No.2 distogram loss
        ## question: do they need only taking last dimension of structure? [-1]
        disto_logits = self.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        ############# No.3 lm loss
        lm_logits = self.lm_head(structure["s_s"])
        if self.track_seq_states:
            seqResNet_logits = structure["seqs_logits"]
        # print(f"lm_logits shape is : {lm_logits.shape}")
        # print(f"seqResNet_logits shape is : {seqResNet_logits.shape}")
            structure["lm_logits"] = (lm_logits + seqResNet_logits)[-1]
        else:
            structure["lm_logits"] = lm_logits[-1]

        ############# No.4 plddt loss
        # need final_atom_positions
        structure["aatype"] = aa
        make_atom14_masks(structure)
        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= mask.unsqueeze(-1)
        structure["residue_index"] = residx 
        structure["final_atom_positions"] = atom14_to_atom37(structure["positions"][-1], structure)
        # for k, v in structure.items():
        #     print(f"structure contains key: {k}")
        lddt_head = self.lddt_head(structure["singles"]).reshape(
            structure["singles"].shape[0], B, L, -1, self.lddt_bins
        )
        # print(f"lddt_head shape is {lddt_head.shape}")
        # print(f"lddt_head last dimension shape is {lddt_head[-1].shape}")
        structure["lddt_head"] = lddt_head
        plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
        structure["plddt"] = (
            100 * plddt
        )  # we predict plDDT between 0 and 1, scale to be between 0 and 100.
        structure["mean_plddt"] = (structure["plddt"] * structure["atom37_atom_exists"]).sum(
            dim=(1, 2)
        ) / structure["atom37_atom_exists"].sum(dim=(1, 2))
        # shape is [..., 37]
        # print(f"plddt shape is {plddt.shape}")
        ca_pos = residue_constants.atom_order["CA"]
        structure["lddt_logits"] = (100 * plddt[..., : , ca_pos])
        # print(f"lddt logits shape is {structure['lddt_logits'].shape}")
        # print(f"lddt logits content is {structure['lddt_logits']}")
        ############# No.4 ptm loss
        ptm_logits = self.ptm_head(structure["s_z"])
        # shape (batch_size, seq_len, seq_len, self.distogram_bins)

        seqlen = mask.type(torch.int64).sum(1)
        structure["ptm_logits"] = ptm_logits
        structure["final_affine_tensor"] = structure["frames"][-1]
        structure["ptm"] = torch.stack(
            [
                compute_tm(
                    batch_ptm_logits[None, :sl, :sl], max_bins=31, no_bins=self.distogram_bins
                )
                for batch_ptm_logits, sl in zip(ptm_logits, seqlen)
            ]
        )
        # print(f"ptm shape is {structure['ptm'].shape}")
        structure.update(
            compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.distogram_bins)
        )

        return structure

    @torch.no_grad()
    def infer(
        self,
        sequences: T.Union[str, T.List[str]],
        residx=None,
        masking_pattern: T.Optional[torch.Tensor] = None,
        num_recycles: T.Optional[int] = None,
        residue_index_offset: T.Optional[int] = 512,
        chain_linker: T.Optional[str] = "G" * 25,
        cpu_only=False,
    ):
        """Runs a forward pass given input sequences.

        Args:
            sequences (Union[str, List[str]]): A list of sequences to make predictions for. Multimers can also be passed in,
                each chain should be separated by a ':' token (e.g. "<chain1>:<chain2>:<chain3>").
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles (cfg.trunk.max_recycles), which is 4.
            residue_index_offset (int): Residue index separation between chains if predicting a multimer. Has no effect on
                single chain predictions. Default: 512.
            chain_linker (str): Linker to use between chains if predicting a multimer. Has no effect on single chain
                predictions. Default: length-25 poly-G ("G" * 25).
        """
        if cpu_only:
            self.esm.float()
        
        if isinstance(sequences, str):
            sequences = [sequences]

        aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, residue_index_offset, chain_linker
        )

        if residx is None:
            residx = _residx
        elif not isinstance(residx, torch.Tensor):
            residx = collate_dense_tensors(residx)

        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
        )

        output = self.forward4infer(
            aatype,
            mask=mask,
            residx=residx,
            masking_pattern=masking_pattern,
            num_recycles=num_recycles,
        )

        output["atom37_atom_exists"] = output["atom37_atom_exists"] * linker_mask.unsqueeze(2)

        output["mean_plddt"] = (output["plddt"] * output["atom37_atom_exists"]).sum(
            dim=(1, 2)
        ) / output["atom37_atom_exists"].sum(dim=(1, 2))
        output["chain_index"] = chain_index

        return output

    def output_to_pdb(self, output: T.Dict) -> T.List[str]:
        """Returns the pbd (file) string from the model given the model output."""
        return output_to_pdb(output)

    def infer_pdbs(self, seqs: T.List[str], *args, **kwargs) -> T.List[str]:
        """Returns list of pdb (files) strings from the model given a list of input sequences."""
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)

    def infer_pdb(self, sequence: str, *args, **kwargs) -> str:
        """Returns the pdb (file) string from the model given an input sequence."""
        return self.infer_pdbs([sequence], *args, **kwargs)[0]

    def set_chunk_size(self, chunk_size: T.Optional[int]):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        # Setting the value to None will return to default behavior, disable chunking.
        self.trunk.set_chunk_size(chunk_size)

    @property
    def device(self):
        return self.esm_s_combine.device
