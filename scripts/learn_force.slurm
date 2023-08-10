#!/bin/bash -l

#SBATCH --time=02:00:00
#SBATCH -N 1 -c 20
#SBATCH --constraint=cstd01
#SBATCH --partition=short

#SBATCH --mem=32G

#SBATCH --mail-user=felix.finkeldey@tu-dortmund.de
#SBATCH --mail-type=ALL
#SBATCH -o /work/smfefink/vibration-velocity-force/out.txt
cd /work/smfefink/vibration-velocity-force
export PYTHONPATH=$PYTHONPATH:/work/smfefink/pylib

module purge
module load python/3.9.10-wip
echo "sbatch: START SLURM_JOB_ID $SLURM_JOB_ID (SLURM_TASK_PID $SLURM_TASK_PID) on $SLURMD_NODENAME"
echo "sbatch: SLURM_JOB_NODELIST $SLURM_JOB_NODELIST"
echo "sbatch: SLURM_JOB_ACCOUNT $SLURM_JOB_ACCOUNT"
pip3 install -r requirements.txt
python3 scripts/learn_force.py

