#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=60GB
##SBATCH --gres=gpu:2
##SBATCH --partition=gpu
#SBATCH --job-name=kerasMNIST
#SBATCH --mail-type=END
#SBATCH --mail-user=skp401@nyu.edu
#SBATCH --array=0-249
#SBATCH --output=slurm_%a.out
module purge
module load tensorflow/python2.7/20170707
module load python/intel/2.7.12
# module load keras/2.0.2
# module load h5py/intel/2.7.0rc2
# module load opencv/intel/3.2
# module load pillow/intel/4.0.0
# module load scikit-image/intel/0.12.3
cd /scratch/skp401/git_code/models/slim

echo 'SLURM_ARRAY_JOB_ID':${SLURM_ARRAY_JOB_ID}
python hess_mul_batch.py ${SLURM_ARRAY_TASK_ID}


