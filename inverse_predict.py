
import torch
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
from openfold.np import residue_constants, protein
from openfold.utils.seed import seed_everything
from statistics import mean
import logging
import argparse
import re
logging.basicConfig(level=logging.INFO)

def gather_job(pdb_dir, protein_list=None):
    pdb_paths = []
    for f_path in os.listdir(pdb_dir):
        if f_path.endswith('.pdb'):
            if protein_list:
                if f_path[0] in protein_list:
                    pdb_path = os.path.join(pdb_dir, f_path)
                    pdb_paths.append(pdb_path) 
            elif not protein_list:
                pdb_path = os.path.join(pdb_dir, f_path)
                pdb_paths.append(pdb_path)
    return pdb_paths


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

def convert_to_37(atom_positions):
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]
    # 37 atom types
    # 0,1,2,4 is N, CA, C, O
    length = atom_positions.shape[0]
    atom_positions1 = atom_positions[:, 0:3, :]
    atom_positions2 = np.zeros((length,1,3))
    atom_positions3 = atom_positions[:, None, 3, :]
    atom_positions4 = np.zeros((length,32,3))
    atom_positions = np.concatenate((atom_positions1,atom_positions2,atom_positions3,atom_positions4), axis=1)
    # print(atom_positions.shape)

    atom_mask1 = np.ones((length,3))
    atom_mask2 = np.zeros((length,1))
    atom_mask3 = np.ones((length,1))
    atom_mask4 = np.zeros((length,32))
    atom_mask = np.concatenate((atom_mask1, atom_mask2, atom_mask3, atom_mask4), axis=1)

    return atom_positions, atom_mask

# data process
def parsePDB(pdb):
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
    CA_coords, C_coords, N_coords, O_coords = [], [], [], []
    
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
        elif line[13:16].strip() == 'O':
            O_coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            O_coords.append(O_coord)
    
    CA_coords = np.array(CA_coords)
    C_coords = np.array(C_coords)
    N_coords = np.array(N_coords)
    O_coords = np.array(O_coords)

    CA_len = CA_coords.shape[0]
    C_len = C_coords.shape[0]
    N_len = N_coords.shape[0]
    O_len = O_coords.shape[0]

    if not ((N_len==CA_len) and (N_len==C_len) and (N_len==O_len)):
        return False
    coordinates = np.stack([N_coords, CA_coords, C_coords], 1)
    all_positions = np.stack([N_coords, CA_coords, C_coords, O_coords], 1)
    return {'title': title,
            'seq': sequence,
            "coordinates": coordinates,
            "all_positions": all_positions,}

## load pdb data
# data = parsePDB('toydata/benchmark/T1025.pdb', ['A'])
# coordinates = data["coordinates"].tolist()
# sequence = data["seq"]
# generated_sequence = model.sample(coordinates, device=device)
# print(f"generate sequnece: {generated_sequence}")
# print(f"native sequence: {sequence}")

def main(args):
    if args.seed is not None:
        seed_everything(args.seed)
    temperature = args.temperature
    ## load model
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm_if1_gvp4_t16_142M_UR50")
    device=args.model_device
    model.to(device)


    directory = args.output_dir
    if not os.path.exists(directory):
        os.makedirs(directory)

    protein_list = None
    if args.protein_name is not None:
        protein_list = [item for item in args.protein_name.split(',')]

    jobs = gather_job(args.input_dir, protein_list)
    logging.info(f"get {len(jobs)} jobs")
    # print(f"get {len(jobs)} jobs")
    ## predict
    average_aar = []
    false_count = []
    logging.info(f"line 151")
    # prevent_list = ["<null_0>", '<null_1>', "<pad>", "<eos>", "<unk>", "<mask>", "<cath>", "<af2>"]
    # text = "This is a <X> formatted text with <AB> and <C>."
    # pattern = r"<\w{3,4}>"
    pattern = re.compile(r"<null_[01]>|<pad>|<eos>|<unk>|<mask>|<cath>|<af2>")

    with torch.no_grad():
        for job in tqdm(jobs):
            logging.info(f"line 154")
            f_path = os.path.basename(job)
            name = f_path.split(".")[0]
            # print(f"treat {name}")
            chain = name[-1]
            data = parsePDB(job)
            if not data:
                false_count.append(name)
                logging.info(f"false_count is {false_count}")
                continue
            else:
                coordinates = data["coordinates"].tolist()
                sequence = data["seq"]
            sample_sequences = []
            aars = []

            for i in range(3):
                generated_sequence = model.sample(coordinates, temperature=temperature, device=device)
                # generated_sequence.to("cpu")
                # print(f"generated_sequence: {generated_sequence}")
                generated_sequence = re.sub(pattern, "X", generated_sequence)
                sample_sequences.append(generated_sequence)
                aar = 100*np.mean([(a==b) for a, b in zip(sequence, generated_sequence)])
                # print(aar)
                aars.append(aar)
                average_aar.append(aar)

            atom_positions, atom_mask = convert_to_37(data["all_positions"])
            residue_index = np.arange(len(sequence), dtype=int)

            for i in range(3):
                sample_protein = protein.from_esmif(sample_sequences[i], atom_positions, atom_mask,  residue_index, aars[i], chain)
                unrelaxed_output_path = os.path.join(directory, f"{name}_sample{i}.pdb")
                with open(unrelaxed_output_path, 'w') as f:
                    f.write(protein.to_pdb(sample_protein))

            logging.info(f"generate {int(len(average_aar)/3)} samples. mean aar is {mean(average_aar)}")  
            # we dump the aar as metric into r-value
            # we can also choose to dump "perplexity"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Path to model parameters."
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="Name of the directory in which to output the prediction",
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        '--temperature', type=float, default=1e-6,
        help="Random seed"
    )
    parser.add_argument(
        '--model_device', type=str, default="cuda:0",
        help="Random seed"
    )
    parser.add_argument(
        '--protein_name', type=str, default=None,
        help="Random seed"
    )
    args = parser.parse_args()
    main(args)