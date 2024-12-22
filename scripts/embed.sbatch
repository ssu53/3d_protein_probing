#!/bin/bash
#
#SBATCH --job-name=embed_for_retrieval
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:RTX_2080Ti|GPU_SKU:TITAN_V|GPU_SKU:TITAN_Xp
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-0%1
#SBATCH --partition=jamesz
#SBATCH --output=/home/groups/jamesz/shiye/3d_protein_probing/slurm_output/embed/%x.%j.%a.out
#SBATCH --error=/home/groups/jamesz/shiye/3d_protein_probing/slurm_error/embed/%x.%j.%a.err
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shiye@stanford.edu

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory="$SLURM_SUBMIT_DIR

source /home/users/shiye/.bashrc

conda activate 3d_protein_probing

cd /home/groups/jamesz/shiye/3d_protein_probing

PROJECT_NAME=embed_for_retrieval_1

experiments=()


for LR in 1e-4 # 1e-4 1e-3 1e-2
do
      
    no_interaction="python scripts/embed_for_retrieval.py \
        --project_name $PROJECT_NAME \
        --save_dir results/embed_for_retrieval_1 \
        --embeddings_path /home/groups/jamesz/shiye/3d_protein_probing/data/embed_for_retrieval/encodings/encodings_foldseek_sinusoid.pt \
        --proteins_path data/scope40_foldseek_compatible/proteins.pt \
        --pairfile_train_path data/embed_for_retrieval/train_data/tmaln_data_train_3.csv \
        --preencoder_type egnn \
        --preencoder_num_layers 3 \
        --preencoder_hidden_dim 64 \
        --preencoder_max_neighbors 24 \
        --preencoder_noise_std 0.0 \
        --embedding_dim 128 \
        --num_layers 4 \
        --num_heads 4 \
        --dropout 0.1 \
        --learning_rate $LR \
        --batch_size 16 \
        --max_epochs 50 \
        --loss_func mseweighted \
        --val_check_interval 2500 \
        --run_name_suffix 19 \
        --run_id_number $SLURM_ARRAY_TASK_ID \
        --entity ssu53"
    interaction="${no_interaction} --interaction_model transformer"
    experiments+=("$no_interaction" "$interaction")
    
done

command=${experiments[SLURM_ARRAY_TASK_ID]}
echo "Task ID = $SLURM_ARRAY_TASK_ID"
echo "Number of experiments = ${#experiments[@]}"
echo "$command"
eval "$command"

echo "Done"
exit 0
