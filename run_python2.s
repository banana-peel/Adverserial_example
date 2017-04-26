#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:15:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
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

cd /home/skp401/
##cat keras_cifar3.py | srun python
cat inception_train.py | srun python


