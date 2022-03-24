ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
WORK_DIR=$SCRATCH/biofold

YAML_CONFIG_PRESET=wm10000
VERSION=narval_v1
python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-renum

python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/database/pdb/rosetta \
    $WORK_DIR/database/pdb/rosetta-renum

python scripts_evaluate/get_metric_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-renum \
    $WORK_DIR/database/pdb/rosetta-renum