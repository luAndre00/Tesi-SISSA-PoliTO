#!/bin/bash
#SBATCH --partition=short_cpu
#SBATCH --job-name=test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s316680@studenti.polito.it
#SBATCH --time=23:59:59
#SBATCH --mem=5G

export PYTHONPATH=$PYTHONPATH:/home/atataranni/PINA

module load singularity
singularity exec /home/atataranni/python-3.9.9.sif python3 '/home/atataranni/PINA/Prove_elliptic_optimal_problem/alpha/run_parametric_elliptic_optimal.py'

