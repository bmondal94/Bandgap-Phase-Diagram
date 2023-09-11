#!/bin/bash
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2540
#SBATCH --no-requeue
#SBATCH --job-name=Bi20S-5
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=
#SBATCH --error=/home/err/conf.stderr_%j
#SBATCH --output=/home/err/conf.stdout_%j
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive
#SBATCH --partition=haswell64
#SBATCH --time=00-26:00:00

umask 022
ulimit -c 0
export OMP_NUM_THREADS=1
module purge
module load VASP/5.4.4-intel-2019b

########## Job path #######################
CDDIR=/scratch/ws/1/GaPBi/Bi20/S-5/conf01/

########## Run Job in loop ###############
cd ${CDDIR}
printf "$(pwd)\n"
printf "*------------------------*\n"
cp ~/INCARfiles/INCAR_Volm_Opt INCAR
cp ~/KPOINTSfiles/KPOINTS KPOINTS 
if [ -s CONTCAR ];then
       echo 'Moving CONTCAR'
       mv CONTCAR POSCAR
fi
CleanVasp
PotVaspV2 -M 'Ga_GW P_GW Bi_GW'
printf "Volm|"
lastloop='False'
while true;do
        mpiexec -n 96 bash -c "ulimit -Ss unlimited && /home/local/VASP5.4.4/vasp_gam" > conf.out
        wait
        number=$(grep "E0" OSZICAR | tail -n 1 | awk '{print $1}')
        if [[ "$number" -gt 1 ]] || ([[ $number -eq 1 ]] && [[ "$lastloop" == 'False' ]]);then
                if [ -s CONTCAR ];then
                        printf 'Moving CONTCAR|'
                        mv CONTCAR POSCAR
		else
			printf "Warning: No CONTCAR|"
                fi
                isi=$(awk '/ISIF/ {print $3}' OUTCAR)
                if [[ "$isi" -eq 7 ]];then
                        cp ~/INCARfiles/INCAR_Pos_Opt INCAR
                        printf "Pos|"
                else
                        cp ~/INCARfiles/INCAR_Volm_Opt INCAR
                        printf "Volm|"
                fi
		if [[ $number -eq 1 ]]; then
			lastloop='True'
			printf "\n*Running final loop."
		fi
        else
                printf "\n---------------------------------------"
                break
        fi
done

############# Bandgap run ##############################
printf "\n* Running bandgap \n"
[ -s CONTCAR ] && mv CONTCAR POSCAR
CleanVasp
PotVaspV2 -M 'Ga_d_GW P_GW Bi_GW'
cp ~/INCARfiles/INCAR_band_gap INCAR
mpiexec -n 96 bash -c "ulimit -Ss unlimited && /sw/installed/VASP/5.4.4-intel-2019b/bin/vasp_ncl" > conf.out
wait
############### Job Finished ###########################
exit
