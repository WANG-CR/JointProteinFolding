ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
WORK_DIR=$SCRATCH/biofold

python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/database/pdb/rosetta \
    $WORK_DIR/database/pdb/rosetta-renum

#######################
YAML_CONFIG_PRESET=cat_refine
VERSION=v2_finetune
python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-renum
VERSION=finetune_v1
python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-renum
VERSION=finetune_v3
python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-renum
VERSION=finetune_v4
python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-renum

python scripts_evaluate/get_metric_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-renum \
    $WORK_DIR/database/pdb/rosetta-renum

# ensemble upper bound
python scripts_evaluate/get_metric_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-v2_finetune-renum \
    $WORK_DIR/database/pdb/rosetta-renum \
    --pred_dir2 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-finetune_v1-renum \
    --pred_dir3 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-finetune_v3-renum \
    --pred_dir4 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-finetune_v4-renum \
#######################

#######################
YAML_CONFIG_PRESET=cat
VERSION=oas_v1
python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-renum
VERSION=narval_v1
python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-renum
VERSION=v2
python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-renum
VERSION=v4
python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION} \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-${VERSION}-renum
# ensemble upper bound
python scripts_evaluate/get_metric_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-narval_v1-renum \
    $WORK_DIR/database/pdb/rosetta-renum \
    --pred_dir2 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-v2-renum \
    --pred_dir3 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-v4-renum \
# ensemble dir based on CDR H3 plddt mean
python scripts_evaluate/ensemble_dir.py \
    --pred_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-narval_v1-H3mean \
    --target_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-H3mean_ensemble \
    --pred_dir2 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-v2-H3mean \
    --pred_dir3 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-v4-H3mean \
# ensemble dir based on mean distance
python scripts_evaluate/ensemble_by_distance.py \
    --pdb_dir1 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-narval_v1-renum \
    --target_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-distance_ensemble \
    --pdb_dir2 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-v2-renum \
    --pdb_dir3 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-v4-renum \
python scripts_evaluate/get_metric_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET}-distance_ensemble \
    $WORK_DIR/database/pdb/rosetta-renum
#######################

YAML_CONFIG_PRESET1=decay
YAML_CONFIG_PRESET2=cat
YAML_CONFIG_PRESET3=replace
YAML_CONFIG_PRESET4=cat_refine
YAML_CONFIG_PRESET5=cat_refine
VERSION=narval_v1
python scripts_evaluate/get_metric_dir.py \
    $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET1}-${VERSION}-renum \
    $WORK_DIR/database/pdb/rosetta-renum \
    --pred_dir2 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET2}-${VERSION}-renum \
    --pred_dir3 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET3}-${VERSION}-renum \
    --pred_dir4 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET4}-${VERSION}-renum \

python scripts_evaluate/ensemble_dir.py \
    --pred_dir $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET2}-${VERSION} \
    --target_dir $WORK_DIR/output/rosetta_benchmark/cat_cat_refine \
    --pred_dir2 $WORK_DIR/output/rosetta_benchmark/${YAML_CONFIG_PRESET5}-${VERSION} \

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
    $WORK_DIR/output/rosetta_benchmark/cat-narval_v1-relaxed \
    $WORK_DIR/output/rosetta_benchmark/cat-narval_v1-relaxed-renum

python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/decay_cat_replace \
    $WORK_DIR/output/rosetta_benchmark/decay_cat_replace-renum


python scripts_evaluate/get_metric_dir.py \
    $WORK_DIR/output/rosetta_benchmark/decay_cat_replace-renum \
    $WORK_DIR/database/pdb/rosetta-renum

python scripts_evaluate/run_relaxation.py \
    $WORK_DIR/output/rosetta_benchmark/cat-narval_v1 \
    $WORK_DIR/output/rosetta_benchmark/cat-narval_v1-relaxed \

python scripts_evaluate/run_relaxation_pyrosetta.py \
    $WORK_DIR/output/rosetta_benchmark/cat_refine-finetune_v4 \
    $WORK_DIR/output/rosetta_benchmark/cat_refine-finetune_v4_pyrosetta \
python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/cat_refine-finetune_v4_pyrosetta \
    $WORK_DIR/output/rosetta_benchmark/cat_refine-finetune_v4_pyrosetta-renum
python scripts_evaluate/renumber_dir.py \
    $WORK_DIR/output/rosetta_benchmark/replace-narval_v1 \
    $WORK_DIR/output/rosetta_benchmark/replace-narval_v1-renum
python scripts_evaluate/get_metric_dir.py \
    $WORK_DIR/output/rosetta_benchmark/cat_refine-finetune_v4_pyrosetta-renum \
    $WORK_DIR/database/pdb/rosetta-renum

# ensemble dir based on mean distance
python scripts_evaluate/ensemble_by_distance.py \
    --target_dir $WORK_DIR/output/rosetta_benchmark/distance_ensemble \
    --pdb_dir1 $WORK_DIR/output/rosetta_benchmark/cat_refine-finetune_v4-renum \
    --pdb_dir2 $WORK_DIR/output/rosetta_benchmark/replace-narval_v1-renum \
    --pdb_dir3 $WORK_DIR/output/rosetta_benchmark/decay-narval_v1-renum \
    --pdb_dir4 $WORK_DIR/output/rosetta_benchmark/cat-narval_v1-renum \
    --pdb_dir5 $WORK_DIR/output/rosetta_benchmark/cat-v2-renum \
    --pdb_dir6 $WORK_DIR/output/rosetta_benchmark/cat-v4-renum \
python scripts_evaluate/get_metric_dir.py \
    $WORK_DIR/output/rosetta_benchmark/distance_ensemble \
    $WORK_DIR/database/pdb/rosetta-renum
python scripts_evaluate/get_metric_dir.py \
    $WORK_DIR/output/rosetta_benchmark/cat-oas_v1-renum \
    $WORK_DIR/database/pdb/rosetta-renum