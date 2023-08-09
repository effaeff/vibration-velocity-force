#!/bin/bash -l

#SBATCH --time=8:00:00
#SBATCH -N 1 -c 20 -C cstd01
#SBATCH --partition=short

#SBATCH --mem=60G

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
pip install -r requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
python3 scripts/learn_force.slurm