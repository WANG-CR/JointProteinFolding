ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
WORK_DIR=$SCRATCH/biofold

YAML_CONFIG_PRESET=msa
VERSION=narval_v1
python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-renum

python scripts_evaluate/get_metric_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-renum \
    $WORK_DIR/database/pdb/rosetta-renum

python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/database/pdb/rosetta \
    $WORK_DIR/database/pdb/rosetta-renum



YAML_CONFIG_PRESET1=decay
YAML_CONFIG_PRESET2=cat
YAML_CONFIG_PRESET3=replace
YAML_CONFIG_PRESET4=msa
VERSION=narval_v1
python scripts_evaluate/get_metric_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET1}-${VERSION}-renum \
    $WORK_DIR/database/pdb/rosetta-renum \
    --pred_dir2 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET2}-${VERSION}-renum \
    --pred_dir3 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET3}-${VERSION}-renum \
    --pred_dir4 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET4}-${VERSION}-renum \

python scripts_evaluate/ensemble_dir.py \
    --pred_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET1}-${VERSION} \
    --target_dir $WORK_DIR/output/rosetta_benchmark/decay_cat \
    --pred_dir2 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET2}-${VERSION} \

python scripts_evaluate/ensemble_dir.py \
    --pred_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET1}-${VERSION} \
    --target_dir $WORK_DIR/output/rosetta_benchmark/decay_cat_replace \
    --pred_dir2 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET2}-${VERSION} \
    --pred_dir3 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET3}-${VERSION} \

python scripts_evaluate/ensemble_dir.py \
    --pred_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET1}-${VERSION} \
    --target_dir $WORK_DIR/output/rosetta_benchmark/decay_cat_replace_msa \
    --pred_dir2 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET2}-${VERSION} \
    --pred_dir3 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET3}-${VERSION} \
    --pred_dir4 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET4}-${VERSION} \

python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/decay_cat \
    $WORK_DIR/output/rosetta_benchmark/decay_cat-renum

python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/decay_cat \
    $WORK_DIR/output/rosetta_benchmark/decay_cat-renum


python scripts_evaluate/get_metric_dir.py \
    $WORK_DIR/output/rosetta_benchmark/decay_cat-renum \
    $WORK_DIR/database/pdb/rosetta-renum