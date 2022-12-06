# Joint Protein Folding
We are developing an performant and easy-to-use codebase for protein folding and inverse folding co-training, which have no dependency on MSA pre-computing.

## Installation* (Linux-Like)

All Python dependencies are specified in `environment.yml`. Some download scripts require `aria2c`.

For convenience, we provide a script that creates a `conda` virtual environment, installs all Python dependencies, and downloads useful resources (including DeepMind's pretrained parameters).
We provide scripts for mist cluster. Run:

```bash
bash scripts/install_third_party_dependencies.sh
```

To activate the environment, run:

```bash
source activate pf3    
```


## Prepare Datasets and Benchmarks*
1. Link of CATH general protein dataset: 
```https://drive.google.com/file/d/1bU2wd5bnLkuvb0yFScW0dgpuf3ZXr8dh/view?usp=sharing```

    The CATH folder is structured as follows
    ```
    structure_datasets
    |
    |___cath
        |
        |___processed/  # the whole dataset after preprocessing
            |   
            |___top_split_512_2023_0.01_0.04_train/  # train split
            |   
            |___top_split_512_2023_0.01_0.04_valid/ # valid split
            |
            |___top_split_512_2023_0.01_0.04_test/ # test split
    ```

2. Link of miniprotein dataset: 
```https://drive.google.com/file/d/1hrNHHgE8DhrpPrKMDfysAfF3fKtcNzuq/view?usp=sharing```

    The miniprotein folder is structured as follows
    ```
    miniprotein
    |___filtered/  # the whole dataset after preprocessing
    |   
    |___train/ # train split
    |   
    |___valid/ # valid split
    |
    |___test/ # test split
    ```

Note that CATH dataset and miniprotein dataset are both PDB type data, which contains atom-level structures of Protein. Please download and unzip these data into a `database` folder, and we will use these data to train the Folding model and InverseFolding model soon.

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