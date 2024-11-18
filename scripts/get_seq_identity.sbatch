#!/bin/bash
#
#SBATCH --job-name=foo
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=jamesz
#SBATCH --output=/home/groups/jamesz/shiye/3d_protein_probing/slurm_output/other/%x.%j.%a.out
#SBATCH --error=/home/groups/jamesz/shiye/3d_protein_probing/slurm_error/other/%x.%j.%a.err
#SBATCH --time=2:00:00
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



python scripts/get_seq_identity.py  



exit 0
