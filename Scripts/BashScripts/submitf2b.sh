#!/bin/bash
#SBATCH --array=0-167
#SBATCH --ntasks=168
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH -o /home/ERR/arraytest-%A_%a.out
#SBATCH -e /home/ERR/arraytest-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --job-name=Unfolding
#SBATCH --mail-type=end
#SBATCH --mail-user=
#SBATCH --no-requeue

umask 022
ulimit -c 0
export OMP_NUM_THREADS=1
export PATH="/home/local/fold2Bloch-VASP-master":$PATH

echo "Hi, I am step $SLURM_ARRAY_TASK_ID in this array job $SLURM_ARRAY_JOB_ID"

FILESNAME=(Sb90.7 Sb92.6 Sb94.4 Sb94.9 Sb95.4 Sb96.3 Sb97.2 Sb98.1)
FILESNAMESTRAIN=(S0 S-0.5 S-1 S-1.5 S-2 S-2.5 S-3 S-3.5 S-4 S-4.5 S-5 S+5 S+4.5 S+4 S+3.5 S+3 S+2.5 S+2 S+1.5 S+1 S+0.5)

CURRENTPATH='/lustre/scratch2/ws/1/GaSbP'
fnamearraylen=${#FILESNAMESTRAIN[@]}
II=$((SLURM_ARRAY_TASK_ID/fnamearraylen))
JJ=$((SLURM_ARRAY_TASK_ID%fnamearraylen))

cd ${CURRENTPATH}/${FILESNAME[II]}/${FILESNAMESTRAIN[JJ]}/conf01
echo $(pwd)
fold2Bloch WAVECAR 6:6:6 -ncl > output.txt
cd $CURRENTPATH

exit
