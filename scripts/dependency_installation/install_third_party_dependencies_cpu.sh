#!/bin/bash

ENV_NAME=pf_cpu
# You may need to run the following two lines first.
# module load gcc
# module load cuda/11.2

# Grab conda-only packages
conda update -qy conda
conda create -p ~/scratch/envs/$ENV_NAME -y python==3.7
source activate ~/scratch/envs/$ENV_NAME

# using cudatoolkit 11.2 because the system load module cuda/11.2
# conda install -y -c https://ftp.osuosl.org/pub/open-ce/1.5.1/ pytorch=1.10.1 cudatoolkit=11.2
# conda install -y pytorch=1.10.1 cpuonly -c pytorch
conda install -y pytorch=1.10.0 cpuonly pyg -c pytorch -c pyg -c conda-forge
conda install -y -c conda-forge easydict ipython scikit-learn
# if pyg not compatible, please use
# pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.1+cu112.html
# pip install torch-geometric
conda install -y llvmdev=10.0.0 matplotlib
pip install cmake
pip install biopython==1.79 deepspeed==0.5.3 ml-collections==0.1.0 PyYAML==5.4.1 requests==2.26.0 \
        tqdm==4.62.2 typing-extensions==3.10.0.2 pytorch_lightning==1.5.0 setuptools==59.5.0 \
        beautifulsoup4 wandb dm-tree \
        git+https://github.com/NVIDIA/dllogger.git nvidia-pyindex

# patch the software
# setrpaths.sh --path ~/scratch/envs/$ENV_NAME/bin

# Download folding resources
#wget -q -P openfold/resources --no-check-certificate \
#    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Certain tests need access to this file
#mkdir -p tests/test_data/alphafold/common
#ln -rs openfold/resources/stereo_chemical_props.txt tests/test_data/alphafold/common

# Download pretrained openfold weights
# scripts/download_alphafold_params.sh openfold/resources

# Decompress test data
#gunzip tests/test_data/sample_feats.pickle.gz
