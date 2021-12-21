import copy
import gzip

import os

import pickle

import numpy
import torch
import unittest

from data.data_transforms import make_seq_mask, add_distillation_flag, make_all_atom_aatype, fix_templates_aatype
from openfold.config import model_config


class TestDataTransforms(unittest.TestCase):
    def test_make_seq_mask(self):
        seq = torch.tensor([range(20)], dtype=torch.int64).transpose(0,1)
        seq_one_hot = torch.FloatTensor(seq.shape[0], 20).zero_()
        seq_one_hot.scatter_(1, seq, 1)
        protein_aatype = torch.tensor(seq_one_hot)
        protein = {'aatype': protein_aatype}
        protein = make_seq_mask(protein)
        print(protein)
        assert 'seq_mask' in protein
        assert protein['seq_mask'].shape == torch.Size((seq.shape[0], 20))

    def test_add_distillation_flag(self):
        protein = {}
        protein = add_distillation_flag.__wrapped__(protein, True)
        print(protein)
        assert 'is_distillation' in protein
        assert protein['is_distillation'] is True

    def test_make_all_atom_aatype(self):
        seq = torch.tensor([range(20)], dtype=torch.int64).transpose(0, 1)
        seq_one_hot = torch.FloatTensor(seq.shape[0], 20).zero_()
        seq_one_hot.scatter_(1, seq, 1)
        protein_aatype = torch.tensor(seq_one_hot)
        protein = {'aatype': protein_aatype}
        protein = make_all_atom_aatype(protein)
        print(protein)
        assert 'all_atom_aatype' in protein
        assert protein['all_atom_aatype'].shape == protein['aatype'].shape

    def test_fix_templates_aatype(self):
        template_seq = torch.tensor(list(range(20))*2, dtype=torch.int64)
        template_seq = template_seq.unsqueeze(0).transpose(0, 1)
        template_seq_one_hot = torch.FloatTensor(template_seq.shape[0], 20).zero_()
        template_seq_one_hot.scatter_(1, template_seq, 1)
        template_aatype = torch.tensor(template_seq_one_hot).unsqueeze(0)
        protein = {'template_aatype': template_aatype}
        protein = fix_templates_aatype(protein)
        print(protein)
        template_seq_ours = torch.tensor([[0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18]*2])
        assert torch.all(torch.eq(protein['template_aatype'], template_seq_ours))


if __name__ == '__main__':
    unittest.main()

