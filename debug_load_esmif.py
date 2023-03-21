
import torch
import numpy as np
import os
from collections import defaultdict
from openfold.np import residue_constants
from openfold.data import cath_dataset
from tqdm import tqdm
from openfold.model.esmif.utils import CoordBatchConverter
from statistics import mean


def featurize_esmif(batch):
    # for i, b in enumerate(batch):
    b = batch[0]
    x = np.stack([b[c] for c in ['N', 'CA', 'C']], 1) # [#atom, 3, 3]
    sequence = b['seq']
    sequence = sequence
    return x, sequence
    
class DataLoader_GTrans(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=None, **kwargs):
        super(DataLoader_GTrans, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,**kwargs)
        self.featurizer = collate_fn

def getSequence(resnames):
    """Returns polypeptide sequence as from list of *resnames* (residue
    name abbreviations)."""
    AAMAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'SEC': 'U', 'PYL': 'O', 'XLE': 'J', '': '-'
    }
    get = AAMAP.get
    return ''.join([get(rn, 'X') for rn in resnames])

# data process
def parsePDB(pdb, chain=['A']):
    title, ext = os.path.splitext(os.path.split(pdb)[1])
    title, ext = os.path.splitext(title)
    pdb = open(pdb)
    lines = defaultdict(list)
    for loc, line in enumerate(pdb):
        # line = line.decode('ANSI_X3.4-1968')
        startswith = line[0:6]
        lines[startswith].append((loc, line))
    
    pdb.close()
    sequence = ''
    CA_coords, C_coords,N_coords = [], [], []
    
    # chain_id = []
    for idx, line in lines['ATOM  ']:
        # if line[21:22].strip() not in chain:
        #     continue
        if line[13:16].strip() == 'CA':
            CA_coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            CA_coords.append(CA_coord)
            sequence += ''.join(getSequence([line[17:20]]))
        elif line[13:16].strip() == 'C':
            C_coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            C_coords.append(C_coord)
        elif line[13:16].strip() == 'N':
            N_coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            N_coords.append(N_coord)
    
    CA_coords = np.array(CA_coords)
    C_coords = np.array(C_coords)
    N_coords = np.array(N_coords)

    coordinates = np.stack([N_coords, CA_coords, C_coords], 1)
    return {'title': title,
            'seq': sequence,
            "coordinates": coordinates,
            'CA': CA_coords,
            'C': C_coords,
            'N': N_coords}

## load pdb data
# data = parsePDB('toydata/benchmark/T1025.pdb', ['A'])
# coordinates = data["coordinates"].tolist()
# sequence = data["seq"]
# generated_sequence = model.sample(coordinates, device=device)
# print(f"generate sequnece: {generated_sequence}")
# print(f"native sequence: {sequence}")

## load json data: 
## create cath dataset and dataloader
cath43_dir = "/home/chuanrui/scratch/database/structure_datasets/cath43"
cath_set = cath_dataset.CATH(cath43_dir, mode='train')
test_set = cath_set
test_set.change_mode('test')
collate_fn = featurize_esmif
test_loader = DataLoader_GTrans(test_set, shuffle=False, num_workers=4, collate_fn=collate_fn)


## load model
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm_if1_gvp4_t16_142M_UR50")
device="cuda:0"
model.to(device)

batch_converter = CoordBatchConverter(alphabet)
aars = []
ppl_fullseqs = []
ppl_withcoords = []
nan_count = 0
num_sample = 0

## predict
with torch.no_grad():
    for batch in tqdm(test_loader):
        coordinates, gt_seqs = batch
        generated_sequence = model.sample(coordinates.tolist(), temperature=1e-6, device=device)
        aar = np.mean([(a==b) for a, b in zip(gt_seqs, generated_sequence)])
        aars.append(aar)
        
        batch = [(coordinates.tolist(), None, gt_seqs)]
        coords, confidence, strs, tokens, padding_mask = batch_converter(
            batch, device=device)
        prev_output_tokens = tokens[:, :-1].to(device)
        target = tokens[:, 1:]
        target_padding_mask = (target == alphabet.padding_idx)
        logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
        loss = torch.nn.functional.cross_entropy(logits, target, reduction='none')
        loss = loss[0].cpu().detach().numpy()
        target_padding_mask = target_padding_mask[0].cpu().numpy()
        ll_fullseq = -np.sum(loss * ~target_padding_mask) / np.sum(~target_padding_mask)
        # Also calculate average when excluding masked portions
        coord_mask = np.all(np.isfinite(coordinates.tolist()), axis=(-1, -2))
        ll_withcoord = -np.sum(loss * coord_mask) / np.sum(coord_mask)
        ppl_fullseq = np.exp(-ll_fullseq)
        ppl_withcoord = np.exp(-ll_withcoord)
        if ppl_fullseq != ppl_withcoord:
            nan_count += 1
        ppl_fullseqs.append(ppl_fullseq)
        ppl_withcoords.append(ppl_withcoord)
        # print(f"loss shape is {loss.shape}")
        # print(f"target_padding_mask is {target_padding_mask.shape}")
        # print(f"ppl_fullseq is {ppl_fullseq}")
        # print(f"ppl_withcoord is {ppl_withcoord}")
        num_sample += 1
        if num_sample % 100 == 0:
            print(f">>>>>>>>>>>>>>>>>>>>>>>")
            print(f"{num_sample} samples")
            print(f"nan_countis: {nan_count}")
            print(f"mean aar is :{mean(aars)}")
            print(f"mean ppl_fullseqs is :{mean(ppl_fullseqs)}")
            print(f"mean ppl_withcoords is :{mean(ppl_withcoords)}")
            
print(f"mean aar is :{mean(aars)}")
print(f"mean ppl_fullseqs is :{mean(ppl_fullseqs)}")
print(f"mean ppl_withcoords is :{mean(ppl_withcoords)}")