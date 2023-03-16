# Joint Protein Folding
We are developing an performant and easy-to-use codebase for protein folding and inverse folding co-training, which have no dependency on MSA pre-computing.

## Installation* (Linux-Like)

All Python dependencies are specified in `environment.yml`. Some download scripts require `aria2c`.

For convenience, we provide a script that creates a `conda` virtual environment, installs all Python dependencies, and downloads useful resources (including DeepMind's pretrained parameters).
We provide scripts for mist cluster. Run:

```bash
bash scripts/install_third_party_dependencies.sh

# alternatively, you can install in another way with script:
bash scripts/install_third_party_dependencies_concret.sh

# in case of cpu environment, please use:
bash scripts/install_third_party_dependencies_cpu.sh
```


To activate the environment, run:

```bash
source activate <env name>    
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

    You can follow the same procedure in the following scripts to start the training (with `bash xxx.sh`). Note that you may need to modify a few variables.
    ```
    scripts/example/train_foldingModel_cath.sh
    scripts/example/train_inverseFoldingModel_cath.sh
    scripts/example/train_jointModel_miniprotein.sh
    ```
    

    We also make training ESMFold possible with our framework:
    ```
    scripts/example/train_esmFold_cath.sh
    ```

    Note: if you are using slurm job system, you can submit the job with `sbatch xxx.sh`.

### Configuration setup
    You can simply modify model configuration by writing different yaml files.
    We simply look at an example `replace_esmFold_16+16.yml`, which implies a foldingModel's configuration of 16 evoformer layers and 16 structure modules. It also configures 16 evoformers layer in the inverse folding model, which will be useful during joint training setup:

    ```
    model:
        evoformer_stack:    # define the number of layers in folding model
            no_blocks: 16
        structure_module:
            no_blocks: 16
        inverse_evoformer_stack:    # define the number of layers in folding model
            no_blocks: 16
    
    globals:
        c_m: 1024   # define the hidden size
        c_z: 128
        c_m_structure: 384
        c_z_structure: 128
        
    data:
        data_module:
            data_loaders:
                num_workers: 10

    optimizer:
        lr: 0.0005  # define the learning rate

    scheduler:  # define the LRscheduler
        warmup_no_steps: 5000
        start_decay_after_n_steps: 50000
        decay_every_n_steps: 5000
    ```


### Logging with W&B
    The codebase currently use Weight&Bias tool to help save the logs, training curves and validation curves.

    If you have not initialize W&B(wandb), please kindly follow the [instruction](https://wandb.ai/quickstart/pytorch) here.


## evaluating ESMFold model
We follow the original evaluation process written in ESMFold paper. The evaluation process can be reproduced as follows:
```
1. download CASP14 dataset from link: https://zenodo.org/record/7713779#.ZA0uKOyZMUF
2. extract the zip, and pick out the 51 proteins used for ESMFold evaluation
3. run script: folding_benchmark.sh 

#note
Esmfold_v0 checkpoint gives the same tm-score in paper, which is 0.68 
Esmfold_v1 checkpoint gives a higher tm-score = 0.70
``` 

P.S. we also need to install additional packages via this cmd:
```
pip install biotite
pip install git+https://github.com/jvkersch/tmtools.git#egg=tmtools
```

## ESMFold training set curation

- step 1: download a list of protein names from PDB


- step 2: 
`bash scripts/data_process/esmfold/batch_download.sh -f scripts/data_process/esmfold/filtered.txt -o <output_dir_for_pdb_gz> -p`


- step 3: pack the .pdb.gz files into a zip, and move this file into a repository associated with the compute node.
`bash scripts/data_process/esmfold/continue_zip.sh`

- step 4: do the following unzip operations on the compute node.
`bash scripts/data_process/esmfold/unzip_pdb.sh`


- step 5: filter the pdb with certain criteria (filter out pdb with 20% of the sequence being the same residue type); seperate pdbs into single chain pdbs; save fasta too
`bash process_pdb.sh`


- step 6: do mmseqs2 easy-cluster with 40% sequence identity

## Copyright notice

## Citing this work
