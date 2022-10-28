# Joint Protein Folding
We are developing an performant and easy-to-use codebase for protein folding and inverse folding co-training, which have no dependency on MSA pre-computing.

## Installation* (Linux-Like)

All Python dependencies are specified in `environment.yml`. Some download scripts require `aria2c`.

For convenience, we provide a script that creates a `conda` virtual environment, installs all Python dependencies, and downloads useful resources (including DeepMind's pretrained parameters).
We provide scripts for mac and mist cluster. Run:

```bash
bash scripts_local/install_third_party_dependencies.sh
bash scripts_cath/install_third_party_dependencies.sh
```

To activate the environment, run:

```bash
source activate pf3    
```


## Prepare Datasets and Benchmarks*


## Usage
### Training

To train the model, you will not need to precompute protein alignments.

You have three options:
1. train protein folding model individually
2. train protein inverse folding model individually
3. train these models together, by adding a joint training loss

You can follow the same procedure in scripts provided in 
```
scripts_cath/train_mist/folding
scripts_cath/train_mist/inverse_folding
scripts_cath/train_mist/joint
```

### Configuration setup




## Copyright notice

## Citing this work