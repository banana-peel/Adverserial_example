#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=0:40:00
#SBATCH --mem=20GB
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu
#SBATCH --job-name=kerasMNIST
#SBATCH --mail-type=END
#SBATCH --mail-user=skp401@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load tensorflow/python2.7/20170218 
module load python/intel/2.7.12
module load keras/2.0.2 
module load h5py/intel/2.7.0rc2
module load opencv/intel/3.2  
module load pillow/intel/4.0.0
module load scikit-image/intel/0.12.3
module load scikit-learn/intel/0.18.1

cd /home/skp401/yale_face/
cat test_yf_hess.py | srun python


