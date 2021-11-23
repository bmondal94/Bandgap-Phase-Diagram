strain=(0 -0.5 -1 -1.5 -2 -2.5 -3 -3.5 -4 -4.5 -5 +5 +4.5 +4 +3.5 +3 +2.5 +2 +1.5 +1 +0.5)
strain=( "${strain[@]/#/S}" )
dirnamee='/scratch/ws/1/GaPBi'
dirnameee="${dirnamee}"
submitscript=${dirnamee}/scripts/submit.sh
n=0

cd ${dirnameee}
for J in Bi*
do
	cd ${dirnameee}/${J}
	for I in ${strain[@]}
	do
		sed -i -e "s/--job-name[^ ]*/--job-name=${J}${I}/" -e "s|CDDIR=[^ ]*|CDDIR=${dirnameee}/${J}/${I}/conf01/|" ${submitscript}
		((n++))
		sbatch ${submitscript}
		sleep 2
	done
done

echo "Total number of job has been submitted = "$n

