#!/bin/bash

ENV_NAME=pf2

# Grab conda-only packages
conda update -qy conda
conda env create --name=$ENV_NAME -f scripts_cath/environment.yml
source activate $ENV_NAME

# Install DeepMind's OpenMM patch
work_path=$(pwd)
python_path=$(which python)
cd $(dirname $(dirname $python_path))/lib/python3.7/site-packages
patch -p0 < $work_path/lib/openmm.patch
cd $work_path

# patch the software
setrpaths.sh --path ~/anaconda3/envs/$ENV_NAME/bin

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
