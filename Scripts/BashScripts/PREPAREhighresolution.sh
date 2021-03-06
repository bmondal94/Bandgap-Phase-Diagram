#!/bin/bash
Conc=(80 75 70 65 60 55 50 45 40 35 30 30 35 40 45 50 55 60 65)
target=(+4.5 +4 +4 +3.5 +3.5 +3 +2.5 +3 +2 +1.5 +1 -1.5 -1.5 -2 -3 -4 -4 -4.5 -4.5)
ToDo=('+4.1;+4.2;+4.3;+4.4;+4.6;+4.7;+4.8' '+3.3;+3.4;+3.6;+3.7;+3.8;+3.9;+4.1;+4.2;+4.3;+4.4' '+3.8;+3.9;+4.1;+4.2;+4.3;+4.4' '+3.3;+3.4;+3.6;+3.7;+3.8;+3.9' '+3.3;+3.4;+3.6;+3.7;+3.8;+4.2' '+2.7;+2.8;+2.9;+3.1;+3.2;+3.3;+3.4;+3.6;+3.7;+3.8' '+2.4;+2.6;+2.8;+2.9;+3.1' '+2.6;+2.7;+2.8;+2.9;+3.1;+3.2' '+1.2;+1.3;+1.3;+1.4;+1.6;+1.7;+1.8;+1.9;+2.2;+2.4;+2.6' '+0.8;+0.9;+1.1;+1.2;+1.3;+1.4;+1.6;+1.7;+1.8;+2.2;+2.3' '+0.6;+0.7;+0.8;+0.9;+1.1;+1.2;+1.3;+1.4;+1.6;+1.8;+2.2;+2.3' '-0.8;-0.9;-1.1;-1.2;-1.3;-1.4;-1.6;-1.8;-2.2' '-1.1;-1.2;-1.3;-1.4;-1.6;-2.2' '-1.6;-1.8;-2.2;-2.4;-2.6' '-3.2;-3.4;-3.6' '-3.8;-4.2;-4.4;-4.6;-4.8' '-3.6;-3.8;-4.2;-4.4;-4.3' '-4.6;-4.8' '-4.6;-4.8')

# Assertion ---------------------------------------------------------------------------------------------
len1=${#Conc[@]}
len2=${#target[@]}
len3=${#ToDo[@]}
if [[ ${len1} > ${len2} ]]; then 
	echo 
	echo "Error: Concentration array length: ${len1}, does not match with target array length: ${len2}."
	echo
	exit 
fi

if [[ ${len1} > ${len3} ]]; then
	echo 
	echo "Error: Concentration array length: ${len1}, does not match with ToDo array length: ${len3}"
	echo
	exit
fi
# --------------------------------------------------------------------------------------------------------

CurPath=$(pwd)

for I in ${!Conc[@]}; do
	II=${Conc[I]}
	K=${target[I]}
	KK=${K#*+}
	IFS=";" read -r -a nRange <<< "${ToDo[I]}"
	for J in ${nRange[@]};do
		fname=${CurPath}/HighResolution/part2/Sb${II}/S${J}/conf01
		mkdir -p $fname
		cp ${CurPath}/Sb${II}/S${K}/conf01/POSCAR ${fname}
		JJ=${J#*+}
		factor=$( echo "scale=16; (100+$JJ) / (100+$KK) " | bc )
		sed -i "2c ${factor} ${factor} 1.0" ${fname}/POSCAR
		#echo "${fname} $factor ${CurPath}/Sb${II}/S${K}/conf01/POSCAR"
	done
done

