#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:00:34 2021

@author: bmondal
"""

#%%-------------------- Importing modules -------------------------------------
import glob, sys
import numpy as np
import json
import time
import BPDFunctions as mf
from datetime import datetime

#%%----------------- main() ---------------------------------------------------
if __name__ == '__main__':
    try:
        args = mf.ParserOptions()
        ## +++++++++++++ Define the parser variables ++++++++++++++++++++++++++
        fuldir = args.d
        NKPOINTS = args.NKP
        NELECT = args.N
        NionNumber = args.NN
        Nspecies = args.SP
        ISPIN = args.ispin
        NCOL = args.ncol
        BW_cutoff = args.BWcutoff
        CompareMean = args.CompareMean
        SuprecellDimension = np.asarray(args.SF)
    except:
    	sys.exit(0)

    #%% ++++++++++++++++ Collect the necessary file paths +++++++++++++++++++++
    
    files = glob.glob(fuldir + '/*/**/WAVECAR_spinor1.f2b', recursive=True)
    if ISPIN==2: NCOL=True
    CB_index = NELECT  if NCOL else NELECT//2
    
    if NKPOINTS > 1:
        print("This scripts is only valid for single Gamma point calculation.")
        print("Multiple kpoints is not implemented yet.")
        #if NKPOINTS>1: FullData = np.split(FullData, NKPOINTS)[0] 
        sys.exit(1)

    #%% +++++++++++++ Create the data dictionary ++++++++++++++++++++++++++++++
    data = mf.NestedDefaultDict()
    print("Preparing data:")
    start_time = time.time()
    for file in files:
        print(f"File: {file}")
        fullfilepath = file.split("/")
        keyvalues = fullfilepath[-4:-1]
        # val1=N%, val2=strain%, val3=configuration
        val1, val2, val3= mf.CollectDigit(keyvalues[0]), mf.CollectDigit(keyvalues[1]), mf.CollectDigit(keyvalues[2])
        CB_VB = mf.CreateData(file, SuprecellDimension, CB_index, CompareMean=CompareMean, BW_cutoff=BW_cutoff)
        data[val1][val2][val3] = CB_VB
    
        if float(val2)==0: # Equilibrium lattice parameter needed for substrate strain calculation.
            poscar_filepath = '/'.join(fullfilepath[:-1])+'/POSCAR'
            llp = mf.getEqmLP(poscar_filepath, SuprecellDimension)
            data[val1][val2][val3]['LP'] = llp
            if float(val3)==1:
                newConcKey = mf.getNC(poscar_filepath, Nspecies, NionNumber)
                data[val1][val2][val3]['NK'] = newConcKey
        print("\t* Processing successful.")
        
    finish_time = time.time()

    #%% +++++++++++++++++ Final data storing ++++++++++++++++++++++++++++++++++
    print("\n************* Data Collection finished *****************\n")
    print("Program:"+' '*17+"BPD_serial.py")
    Total_time = finish_time - start_time
    n = len(files)
    Time_per_file = Total_time/n 
    print(f"\nDirectory path: {' '*9}{fuldir}\nTotal number of files: {' '*2}{n}")
    print(f"Total time: {' '*13}{Total_time:.3f} s\n")
    print("\n************* Creating Database *****************\n")
    BW_mean='_mean' if CompareMean else '_max'
    dbname = fuldir+'/BPD_DataBase_serial_'+datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+'_BW'+str(BW_cutoff)+BW_mean+'.json'
    with open(dbname,'w') as f:
        json.dump(data, f)
    print(f'+ Database creation successfull: {dbname}')
    print("\n************* Congratulation: All done *****************\n")
#------------------------------------------------------------------------------
