#!/bin/bash
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2450
#SBATCH --no-requeue
#SBATCH --job-name=r2SET101/In9Ga207P167As29Sb20
#SBATCH --mail-type=fail
#SBATCH --mail-user=
#SBATCH --error=/home/err/Inconf.stderr_%j
#SBATCH --output=/home/err/Inconf.stdout_%j
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive
#SBATCH --partition=haswell64
#SBATCH --time=40:00:00

umask 022
ulimit -c 0
export OMP_NUM_THREADS=1

module purge
module load vasp/6.3.2-intel-hdf5_iimpi-2021b

########## Vasp binary paths ##############
VaspRun='/projects/VASP/BINARIES/6.3.2-intel-hdf5_iimpi-2021b/bm/bin'
VaspRunConstr='/projects/VASP/BINARIES/6.3.2-intel-hdf5_iimpi-2021b/bm/bin_constr_xy'

########## Job path #######################
CDDIR=/scratch/InGaPAsSb/Simulations/SESSION1/SET101/In9Ga207P167As29Sb20/EQM/conf01/

########## Run eqm optimization pos-volm opt ###############
cd ${CDDIR}
	echo "Folder = $(pwd)"
	echo '* Running equilibrium optimization'
	echo '*------------------------*'
	cp ~/Sample_INCAR/INCAR_PosVolm_Opt_low_level INCAR
	cp ~/Sample_KPOINTS/KPOINTS KPOINTS 
	[ -s CONTCAR ] && mv CONTCAR POSCAR
	CleanVasp
	PotVasp
	echo '----- Low level Position-Volume optimization -----'
	echo "Start = $(date)"
	start_d=$(date +%s)
	srun -n $SLURM_NTASKS bash -c "ulimit -Ss unlimited && ${VaspRun}/vasp_gam_opt" > conf.out
	wait
	echo "End = $(date)"
	end_d=$(date +%s)
	echo "Simulation time = " $(echo $((end_d - start_d)) | awk '{printf "%02d:%02d:%02d", $1/3600, ($1/60)%60, $1%60}')
	
	sleep 20
	TotalIonicSteps=$(grep "E0" OSZICAR | tail -n 1 | awk '{print $1}')
	echo "Total ionic steps = ${TotalIonicSteps}"
	
	echo '----- High level Position-Volume optimization -----'
	cp ~/Sample_INCAR/INCAR_PosVolm_Opt INCAR
	[ -s CONTCAR ] && mv CONTCAR POSCAR
	echo "Start = $(date)"
	start_d=$(date +%s)
	srun -n $SLURM_NTASKS bash -c "ulimit -Ss unlimited && ${VaspRun}/vasp_gam_opt" > conf.out
	wait
	echo "End = $(date)"
	end_d=$(date +%s)
	echo "Simulation time = " $(echo $((end_d - start_d)) | awk '{printf "%02d:%02d:%02d", $1/3600, ($1/60)%60, $1%60}')
	
	sleep 20
	TotalIonicSteps=$(grep "E0" OSZICAR | tail -n 1 | awk '{print $1}')
	echo "Total ionic steps = ${TotalIonicSteps}"
	
	[ -s CONTCAR ] && mv CONTCAR POSCAR
	CleanVasp
	rm INCAR KPOINTS

cd ../../

########## Create strain folders ###############
echo " "
echo '* Creating strained structures'
CurrentFolder=$(pwd)
WhichFolder=${CurrentFolder##*/}
WhichFolder=${WhichFolder:-/}
X=$(find ../../../../AllRandomFolders/SESSION1/${WhichFolder}/S* -maxdepth 0 -type d -printf '%f\n')
echo 'The strain folders are: '
for K in $X;do 
	cp -r EQM $K
	echo "Strain folder = $K"
	XX=${K#*S}
	lp=$(echo "scale=6;$XX /100 + 1"|bc -l) 
	XXX=$(echo "scale=6;1 - $XX /50 "|bc -l )
	echo "  POSCAR replace 2c = $lp   $lp   $XXX"
	sed -i "2c $lp  $lp  $XXX " $K/conf01/POSCAR
done

########## Run strained jobs ###############
echo ' '
echo '* Running strained optimization'

for SF in S* ; do
	cd ${SF}/conf01
	echo "Folder = $(pwd)"
	echo '*------------------------*'
	echo '----- Low level Position-Volume optimization -----'
	cp ~/Sample_INCAR/INCAR_PosVolm_Opt_low_level INCAR
	cp ~/Sample_KPOINTS/KPOINTS KPOINTS 
	[ -s CONTCAR ] && mv CONTCAR POSCAR
	CleanVasp
	PotVasp
	echo "Start = $(date)"
	start_d=$(date +%s)
	srun -n $SLURM_NTASKS bash -c "ulimit -Ss unlimited && ${VaspRunConstr}/vasp_gam_opt" > conf.out
	wait
	echo "End = $(date)"
	end_d=$(date +%s)
	echo "Simulation time = " $(echo $((end_d - start_d)) | awk '{printf "%02d:%02d:%02d", $1/3600, ($1/60)%60, $1%60}')
	
	sleep 20
	TotalIonicSteps=$(grep "E0" OSZICAR | tail -n 1 | awk '{print $1}')
	echo "Total ionic steps = ${TotalIonicSteps}"
	
	echo '----- High level Position-Volume optimization -----'
	cp ~/Sample_INCAR/INCAR_PosVolm_Opt INCAR
	[ -s CONTCAR ] && mv CONTCAR POSCAR
	echo "Start = $(date)"
	start_d=$(date +%s)
	srun -n $SLURM_NTASKS bash -c "ulimit -Ss unlimited && ${VaspRunConstr}/vasp_gam_opt" > conf.out
	wait
	echo "End = $(date)"
	end_d=$(date +%s)
	echo "Simulation time = " $(echo $((end_d - start_d)) | awk '{printf "%02d:%02d:%02d", $1/3600, ($1/60)%60, $1%60}')
	
	sleep 20
	TotalIonicSteps=$(grep "E0" OSZICAR | tail -n 1 | awk '{print $1}')
	echo "Total ionic steps = ${TotalIonicSteps}"

	cd ../../
done

module purge
module load VASP/5.4.4-intel-2019b
echo ' '	
echo '* Running bandgap'
echo '*------------------------*'
	
for SF in S* ; do
	cd ${SF}/conf01
	echo "Folder = $(pwd)"
	echo '*------------------------*'
	############# Bandgap run ##############################
	[ -s CONTCAR ] && mv CONTCAR POSCAR
	CleanVasp
	PotVasp
	cp ~/Sample_INCAR/INCAR_MBJ_bandgap_tmp INCAR
	cp ~/Sample_KPOINTS/KPOINTS KPOINTS 
	echo "Start = $(date)"
	start_d=$(date +%s)
	srun -n $SLURM_NTASKS bash -c "ulimit -Ss unlimited && /sw/installed/VASP/5.4.4-intel-2019b/bin/vasp_ncl" > conf.out
	wait
	sleep 20
	echo "End = $(date)"
	end_d=$(date +%s)
	echo "Simulation time = " $(echo $((end_d - start_d)) | awk '{printf "%02d:%02d:%02d", $1/3600, ($1/60)%60, $1%60}')
	echo " "
	echo '-- Done bandgap'

	############## Convergence check and band unfolding #######################
	echo '* Bandgap Convergency check'
        if [ -s "OUTCAR" ] ; then
		if grep -q "METAGGA = MBJ" OUTCAR; then
                        nelmoutcar=$(grep 'NELM' OUTCAR | tail -n 1 | awk '{print $3}'| cut -d ';' -f1)
                        nelmoszicar=$(tail -n 2 OSZICAR | head -n 1 | awk '{print $2}')
                        nelmdiff=$(bc<<<"$nelmoutcar-$nelmoszicar")
                        if [[ $nelmdiff -ne 0 ]];then
                                if [ $( grep -L 'Total CPU time used' OUTCAR) ] ; then
                                        echo '-- Error: Simulation did not finish.'
				else
					echo '-- Perfect simulation'
					############# Bandunfolding ##############################
					#echo '* Bandgap unfolding'
					#fold2Bloch WAVECAR 6:6:6 -ncl > unfolding.txt
					#wait
					#echo '* Cleaning up'
					#rm CHG* POTCAR CONTCAR XDATCAR PCDAT IBZKPT REPORT INCAR KPOINTS *.out  DOSCAR  EIGENVAL  OSZICAR WAVECAR
                                fi
                        else
                                echo '-- Error: Simulation did not finish properly. All the NELM step is consumed.'
                        fi
                else
                        echo '-- Error: MBJ metagga is not used.'
	                i=$(grep 'ISIF' OUTCAR | tail -n 1 | awk '{print $3}')
                        if [[ $i -ne 0 ]]; then
                                echo '-- Error: ISIF in OUTCAR = $i (not 0).'
                        fi
                fi
        else
                echo 'Error: No or empty OUTCAR'
        fi
	cd ../../
done
wait
############### Job Finished ###########################
exit
