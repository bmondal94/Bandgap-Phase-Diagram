#!/bin/bash

#SBATCH --ntasks=24
#SBATCH --job-name=Biaxial_mpi_db
#SBATCH --exclusive
#SBATCH --time=00-00:25:00
##SBATCH --wait-all-nodes=1

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --no-requeue
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=
#SBATCH --error=/home/err/BiaxialMpi4Py.stderr_%j
#SBATCH --output=/home/err/BiaxialMpi4Py.stdout_%j
#SBATCH --partition=haswell64

#export OMP_NUM_THREADS=24
module load SciPy-bundle/2020.03-intel-2020a-Python-3.8.2

#scriptpathserial="${HOME}/script/BPDPythonScripts/BPDserial.py"
scriptpath="${HOME}/script/BPDPythonScripts/BPDmpi.py"
DIRPATH='/home/GaSbP'
NELECT=3888
NSPECIES='Sb'
NN=216
BW=20
M='max'
srun -n 24 python $scriptpath -d $DIRPATH -N $NELECT -SP $NSPECIES -NN $NN -ncol -BWcutoff $BW > $DIRPATH/Output_DB_mpi_$(date '+%Y-%m-%d-%H:%M:%S')_BW${BW}_${M}.txt
#srun -n 24 python $scriptpath -d $DIRPATH -N $NELECT -SP $NSPECIES -NN $NN -ncol -BWcutoff $BW -CompareMean > $DIRPATH/Output_DB_mpi_$(date '+%Y-%m-%d-%H:%M:%S')_BW${BW}_${M}.txt
#python $scriptpathserial -d $DIRPATH -N $NELECT -SP $NSPECIES -NN $NN -ncol -BWcutoff $BW -CompareMean > $DIRPATH/Output_DB_serial_$(date '+%Y-%m-%d-%H:%M:%S')_BW${BW}_${M}.txt

wait
exit
