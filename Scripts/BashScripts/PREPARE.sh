#!bin/bash
com=(0.995 0.99 0.985 0.98 0.975 0.97 0.965 0.96 0.955 0.95)
ten=(1.005 1.01 1.015 1.02 1.025 1.03 1.035 1.04 1.045 1.05)
strain=(0.5 1 1.5 2 2.5 3 3.5 4 4.5 5)


for I in Sb*; do
	for J in ${!com[@]}; do
		nfile="$I/S-${strain[J]}"
		cp -r $I/S0 $nfile
		sed -i "2c ${com[J]}  ${com[J]}  1.0" ${nfile}/conf01/POSCAR
	done
	for J in ${!ten[@]}; do
		nfile="$I/S+${strain[J]}"
		cp -r $I/S0 $nfile
		sed -i "2c ${ten[J]}  ${ten[J]}  1.0" ${nfile}/conf01/POSCAR
	done
done
