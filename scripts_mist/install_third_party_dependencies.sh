#!/bin/bash

ENV_NAME=biofold
# You may need to run the following two lines first.
# module load gcc
# module load cuda/11.2

# Grab conda-only packages
conda update -qy conda
conda create -p ~/scratch/envs/$ENV_NAME -y python==3.7
source activate ~/scratch/envs/$ENV_NAME

conda install -y -c https://ftp.osuosl.org/pub/open-ce/1.5.1/ pytorch=1.10.1 cudatoolkit=11.2
conda install -y -c conda-forge easydict ipython scikit-learn
conda install -y llvmdev=10.0.0 matplotlib
pip install cmake
pip install biopython==1.79 deepspeed==0.5.3 ml-collections==0.1.0 PyYAML==5.4.1 requests==2.26.0 \
        tqdm==4.62.2 typing-extensions==3.10.0.2 pytorch_lightning==1.5.0 \
        beautifulsoup4 fair-esm wandb \
        git+https://github.com/NVIDIA/dllogger.git nvidia-pyindex \
        git+https://github.com/deepmind/tree.git

# These lines fail on mist
# conda install -y -c conda-forge openmm=7.5.1 pdbfixer
# conda install -y -c bioconda aria2
# conda install -y -c bioconda hmmer==3.3.2 hhsuite==3.3.0 kalign2==2.04

# patch the software
setrpaths.sh --path ~/scratch/envs/$ENV_NAME/bin

# Download folding resources
wget -q -P openfold/resources --no-check-certificate \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Certain tests need access to this file
mkdir -p tests/test_data/alphafold/common
ln -rs openfold/resources/stereo_chemical_props.txt tests/test_data/alphafold/common

# Download pretrained openfold weights
# scripts/download_alphafold_params.sh openfold/resources

# Decompress test data
gunzip tests/test_data/sample_feats.pickle.gz