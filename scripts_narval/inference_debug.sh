#! /bin/bash

# 1. Activate environement
ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
echo env done
WORK_DIR=$SCRATCH/biofold
TRAIN_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_train_merged
VALID_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/20220319_99_True_All__4_valid_merged
ROSETTA_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/rosetta_merged
THERAPEUTICS_EMBED_DIR=$WORK_DIR/pretrained_embeddings/esm1b/therapeutics_merged

# 2. Openfold
# 2.1 With pre-alignments
python run_pretrained_openfold.py \
    $WORK_DIR/example_data/inference/7n4l_cat30x.fasta \
    --use_precomputed_alignments $WORK_DIR/example_data/alignments \
    --output_dir $WORK_DIR/example_data/inference \
    --model_device cuda:0 \
    --param_path openfold/resources/params/params_model_1.npz \
    --no_recycling_iters 3 \
    --relax true \

# 2.2 Without pre-alignments
# ...

# 3. Biofold
# 3.1 Single fasta inference
YAML_CONFIG_PRESET=wm5000_long_esm
VERSION=narval_v1
CKPT_NAME=epoch55-step26655-val_loss=0.91.ckpt
python run_pretrained_biofold.py \
    $WORK_DIR/example_data/inference/7n4l.fasta \
    $WORK_DIR/output/wandb_biofold/${YAML_CONFIG_PRESET}-${VERSION}/checkpoints/${CKPT_NAME} \
    --pdb_path $WORK_DIR/database/pdb/20220319_99_True_All__4_sabdab \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --output_dir $WORK_DIR/example_data/inference \
    --residue_embedding_dir $VALID_EMBED_DIR/ \
    --model_device cuda:0 \
    --no_recycling_iters 3 \
    --relax false \

# 3.2 batch inference
# See scripts_narvel/inference/