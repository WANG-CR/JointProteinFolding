export DATA_DIR=$SCRATCH/SAbDab_database
export WORK_DIR=$SCRATCH/torchfold

python batchrun_pretrained_torchfold.py \
    $DATA_DIR/fasta_from_pdb/rosetta_HL.fasta \
    0 \
    $WORK_DIR/model_evo8/warmup1000_new/checkpoints/2022-02-09-10-30-52_ng30907/last.ckpt \
    --output_dir $WORK_DIR/output/model_evo8/warmup1000_new/rosetta_132 \
    --model_device cuda:0 \
    --cpus 48 \
    --model_name model_evo8 \
    --no_recycling_iters 3 \
    --relax False \
    --ckpt_tag global_step45200

cd $WORK_DIR/output/model_evo8/warmup1000_new/

python $WORK_DIR/scripts_mac/renumber_dir.py rosetta_132/ rosetta_132_renum/
python $WORK_DIR/scripts_mac/get_metric_dir.py rosetta_132_renum/ $DATA_DIR/rosetta_antibody_benchmark_renum