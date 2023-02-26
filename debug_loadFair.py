
from openfold.model.esm.esmfold import ESMFold, ESMFoldConfig, constructConfigFromYAML
from openfold.model.esm.trunk import StructureModuleConfig, FoldingTrunkConfig
import torch
import numpy as np
import os
from openfold.config import model_config
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.np import residue_constants, protein

def save_esm_protein(output, output_dir, name, postfix):
    b_factors = np.zeros_like(output["final_atom_mask"])
    unrelaxed_protein = protein.from_esm_prediction(
        result=output,
        b_factors=b_factors
    )        
    unrelaxed_output_path = os.path.join(
        output_dir,
        f"{name}_{postfix}.pdb"
    )
    with open(unrelaxed_output_path, 'w') as f:
        f.write(protein.to_pdb(unrelaxed_protein))


model_data = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt", progress=False, map_location="cpu")
# cfg = model_data["cfg"]["model"]
config = model_config(
    name='3B+48L',
    yaml_config_preset='/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/yaml_config/joint_finetune/esm3B+8inv_samlldim.yml',
    train=False,
    low_prec=False,
)
cfg = constructConfigFromYAML(config)

# cfg = ESMFoldConfig()
model_state = model_data["model"]
model = ESMFold(esmfold_config=cfg, using_fair=True)
model.load_state_dict(model_state, strict=False)
print(f"load succesfully")

expected_keys = set(model.state_dict().keys())
found_keys = set(model_state.keys())

for print_key in expected_keys:
    if not print_key.startswith("esm."):
        print(print_key)


# abundant_keys = expected_keys - found_keys
# abundant_essential_keys = []
# for missing_key in abundant_keys:
#     if not missing_key.startswith("esm."):
#         abundant_essential_keys.append(missing_key)

# abundant_essential_keys is:  
# structure_module's: seq_resnet, seq_emb_nn,


# missing_keys = found_keys - expected_keys
# missing_keys is:  {
# 'trunk.structure_module.group_idx', 
# 'trunk.structure_module.default_frames', 
# 'trunk.structure_module.atom_mask', 
# 'positional_encoding._float_tensor', 
# 'trunk.structure_module.lit_positions'}


# print(f"abundant_essential_keys is:  {abundant_essential_keys}")
# print(f"missing_keys is:  {missing_keys}")

# sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
sequence = "RPYACPVESCDRRFSRSADLTRHIRIHTGQ"
with torch.no_grad():
    output = model.infer_pdb(sequence, num_recycles=3, cpu_only=True)

with open("/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/toydata/generated/1i1A_generated_48layer_2.pdb", "w") as f:
    f.write(output)

with torch.no_grad():
    output2 = model.infer_bb(sequence, num_recycles=3, cpu_only=True)

output2 = tensor_tree_map(lambda x: np.array(x[0, ...]), output2)
output_dir = "/home/chuanrui/scratch/research/ProteinFolding/JointProteinFolding/toydata/generated/"
save_esm_protein(output2, output_dir, "1i1A", "generated_48layer_inferbb")
