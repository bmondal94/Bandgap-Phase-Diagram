#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:20:18 2021

@author: bmondal
"""


#%%-------------------- Importing modules -------------------------------------
import numpy as np      
import glob
import BPDFunctions as mf
import json
from mpi4py import MPI
from datetime import datetime

#%%----------------- main() ---------------------------------------------------
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()
    
    if rank == 0:
        try:
            args = mf.ParserOptions()
            ## +++++++++ Define the parser variables ++++++++++++++++++++++++++
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
            
            ## +++++++++++ Define other necessary variables +++++++++++++++++++
            dirnamesavefig=fuldir+'/figs'
            files = glob.glob(fuldir + '/*/**/WAVECAR_spinor1.f2b', recursive=True)
            
            if ISPIN == 2 : NCOL = True
            if NKPOINTS > 1:
                print("This scripts is only valid for single Gamma point calculation.")
                print("Multiple kpoints is not implemented yet.")
                exit(1)       

            CB_index = NELECT  if NCOL else NELECT//2
            
            pos_list = [CB_index, SuprecellDimension, Nspecies,NionNumber, BW_cutoff, CompareMean]
            #pos_list = [NCOL, SuprecellDimension, Nspecies,NionNumber, BW_cutoff, CompareMean]
                
        except:
            files = None

    else:
        files = None
        pos_list = None

    files = comm.bcast(files, root=0)
    if files is None:
        exit(0)

    pos_list = comm.bcast(pos_list, root=0)
    
    #%% +++++++++ Collect the data ++++++++++++++++++++++++++++++++++++++++++++ 
    data = mf.NestedDefaultDict_type2()  
    n = len(files)
    ineachNode = n//size
    num_larger_procs = n - size*ineachNode
    if rank<num_larger_procs:
        ineachNode += 1
        ini = rank*ineachNode
    else:
        ini = rank*ineachNode+num_larger_procs

    start_time = MPI.Wtime()
    for file in files[ini:ini+ineachNode]:
        print(f"File: {file} in Process:{rank}")
        fullfilepath = file.split("/")
        keyvalues = fullfilepath[-4:-1]
        val1, val2, val3= mf.CollectDigit(keyvalues[0]), mf.CollectDigit(keyvalues[1]), mf.CollectDigit(keyvalues[2])       
               
        CB_index, SuprecellDimension = pos_list[0], pos_list[1]
        # NCOL, SuprecellDimension = pos_list[0], pos_list[1]
        # NELECT = 
        #CB_index = NELECT  if NCOL else NELECT//2
        ionspecies, NionNumber, BW_cutoff, CompareMean = pos_list[2], pos_list[3], pos_list[4], pos_list[5]
        
        CB_VB = mf.CreateData(file, SuprecellDimension, CB_index, CompareMean=CompareMean, BW_cutoff=BW_cutoff)
        
        data[val1][val2][val3] = CB_VB

        if float(val2)==0: # Equilibrium lattice parameter needed for substrate strain calculation.
            poscar_filepath = '/'.join(fullfilepath[:-1])+'/POSCAR'
            llp = mf.getEqmLP(poscar_filepath, SuprecellDimension)
            data[val1][val2][val3]['LP'] = llp
            if float(val3)==1:
                newConcKey = mf.getNC(poscar_filepath, ionspecies, NionNumber)
                data[val1][val2][val3]['NK'] = newConcKey
        
        print(f"\t* Processing successful in Process:{rank}")
            
    finish_time = MPI.Wtime() 
    
    #%% ++++++++++++++ Final data gather and storing ++++++++++++++++++++++++++
    if rank == 0:
        for i in range(1, size, 1): 
            rdata = comm.recv(source=i, tag=11, status=status)
            data = mf.Merge2Dict(data,rdata)
            
        print("\n***************** Data Collection finished *******************\n")
        print("Program:"+' '*21 +'BPD_mpi.py')
        Total_time = finish_time - start_time
        Time_per_file = Total_time/n 
        print(f"\nDirectory path: {' '*13}{fuldir}\nTotal number of files: {' '*6}{n}\nThe number of mpi process: {' '*2}{size}")
        print(f"Total time: {' '*17}{Total_time:.3f} s\n")
        print("\n******************** Creating Database ***********************\n")
        BW_mean='_mean' if CompareMean else '_max'
        dbname = fuldir+'/BPD_DataBase_mpi_'+datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+'_BW'+str(BW_cutoff)+BW_mean+'.json'
        with open(dbname,'w') as f:
            json.dump(data, f) #,indent=2)
        print(f'+ Database creation successfull: {dbname}')
        print("\n*************** Congratulation: All done *********************\n")

    else:
        request = comm.send(data, dest=0, tag=11)
    
#------------------------------------------------------------------------------             
