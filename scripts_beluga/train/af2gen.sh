#! /bin/bash
#SBATCH --account=ctb-lcharlin
#SBATCH --cpus-per-task=10
#SBATCH --mem=186G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=/home/shichenc/scratch/af2gen/output/slurm_log/gen_abag_v2.out
#SBATCH --error=/home/shichenc/scratch/af2gen/output/slurm_log/gen_abag_v2.err
#SBATCH --qos=unkillable

ENV_NAME=biofold
module load cuda/11.4
source activate $ENV_NAME
WORK_DIR=$SCRATCH/af2gen
TRAIN_DIR=$WORK_DIR/sabdab/pdb/20220618_Fv_4_All_cdrh3_train
VALID_DIR=$WORK_DIR/sabdab/pdb/20220618_Fv_4_All_cdrh3_valid
OUTPUT_DIR=$WORK_DIR/output

YAML_CONFIG_PRESET=gen_abag
srun python train_biofold.py $TRAIN_DIR/ $TRAIN_DIR/ $TRAIN_DIR/ \
    $OUTPUT_DIR/ \
    2021-12-31 \
    --seed 2022 \
    --yaml_config_preset yaml_config/${YAML_CONFIG_PRESET}.yml \
    --val_data_dir $VALID_DIR/ \
    --val_alignment_dir $VALID_DIR/ \
    --precision 16 --gpus 4 --log_every_n_steps 50 \
    --wandb true \
    --wandb_entity chenceshi \
    --wandb_version abag_v2 \
    --wandb_project wandb_af2gen \
    --deepspeed_config_path deepspeed_config_scc.json \
    --sabdab_summary_file $WORK_DIR/sabdab/info/20220618_Fv_4_All_sabdab_summary.tsv \