#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 22:15:19 2021

@author: bmondal
"""

import numpy as np
from collections import defaultdict
from itertools import combinations
import sys
from math import isclose
import re
import argparse
from datetime import datetime
import HeaderTxt
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata #, make_lsq_spline, BSpline, make_interp_spline 

#%%----------------------------------------------------------------------------
def BPD_header(mytxt):
    """
    This function is printing out the supplied text in terminal as a big text.

    Parameters
    ----------
    mytxt : String
        The text that you want to print in terminal.

    Returns
    -------
    None.

    """
    split_list = []; result_list = []
    for I in mytxt:
        split_list.append(HeaderTxt.standard_dic[I].split("\n"))
    
    for i in range(len(split_list[0])):
        temp=""
        for j, item in enumerate(split_list):
            if j>0 and (i==1 or i==len(split_list[0])-1):
                temp += ""
            temp += item[i]
        result_list.append(temp)       
    
    result = ('\n').join(result_list)
    print(result)
    return

def star(func):
    """
    Decorator for Header function, HeaderDecorator(). 
    """
    def inner(*args, **kwargs):
        print('\n'+"*" * 151)
        func(*args, **kwargs)
        print("*" * 151+'\n')
    return inner

def percent(func):
    '''
    Decorator for Header function, HeaderDecorator().
    '''
    def inner(*args, **kwargs):
        print("%" * 151)
        func(*args, **kwargs)
        print("%" * 151)
    return inner

@star
@percent
def HeaderDecorator():
    """
    The header function to print the decorated text on top of BPD_mpi/serial.py output.
    """
    now = datetime.now()
    BPD_header('BANDGAP     PHASE    DIAGRAM')
    print(f"{' '*64} Date: {now.strftime('%Y-%m-%d  %H:%M:%S')}\n")
    return 


def ParserOptions():
    """
    This finction defines the parsers.

    Returns
    -------
    Parser arguments

    """
    HeaderDecorator()
    sys.stdout.flush()
    parser = argparse.ArgumentParser(prog='BPD_serial/mpi.py', description='This script creates the database for Bandgap phase diagram from VASP WAVECAR and POSCAR file.', epilog='Have fun!')
    parser.add_argument('-d', metavar='DIRECTORYNAME', default=".", help='The file path where the tree of strain folders are (default: current directory). e.g. /home/mondal/VASP/test/')
    parser.add_argument('-N', type=int, help='Total number of electrons.')
    parser.add_argument('-SP', type=str, help='Ion species w.r.t which concentration will be calculated. e.g. P')
    parser.add_argument('-NN', type=int, help='Total number of ion w.r.t which concentration will be calculated.')
    parser.add_argument('-NKP', type=int, default=1, help='Total number of KPOINTS. (default: 1)')
    parser.add_argument('-SF', nargs='+', type=int, default=[6,6,6], help='Supercell dimensions (must be int) in a, b and c lattice vector directions respectively. (default: [6,6,6])')
    parser.add_argument('-ispin', type=int, default=1, help='If ISPIN=1 or 2  (default: 1)')
    parser.add_argument('-ncol', action='store_true', default=False, help='Noncolliner calculation = True or False (default: False).')
    parser.add_argument('-BWcutoff', type=float, default=20, help='Cutoff Bloch weight  (default: 20)')
    parser.add_argument('-CompareMean', action='store_true', default=False, help='Whether the BW at the symmetry equivalent point willl be calculated as average. (default: False, uses maximum).')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    
    return parser.parse_args()
        
#------------------------------------------------------------------------------
#  https://stackoverflow.com/a/56338725
'''
## https://stackoverflow.com/a/19189356
def rec_dd():
    return defaultdict(rec_dd)
#or ------------------------------
tree = lambda: defaultdict(tree)
'''
class NestedDefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))
    
def NestedDefaultDict_type2():
    return defaultdict(NestedDefaultDict_type2)

def CollectDigit(mystring):
    """
    This function returns the signed number in a string.

    Parameters
    ----------
    mystring : String
        The string from which you want to extract the numbers.

    Returns
    -------
    String
        All non-overlapping matches of pattern in string, as a list of strings or tuples.

    """
    return re.findall('-?\+?\d+\.?\d*', mystring)[0]
    
#****************************************************************************** 
#%%  
def SymmetryEquivKpoints_zincblende_Gfolding(N, dmprecesion):
    """
    This function calculates the symmetry equivalent k-points for zincblende
    structures with only Gamma point band folding.

    Parameters
    ----------
    N : Integer array
        The array containing 3 supercell dimensions.
    dmprecesion : Integer
        The decimal precision that will be used for rounding float.

    Returns
    -------
    Float nd-list
        The k-points coordinates in reciprocal space.

    """
    NN = np.array(N)/2

    kp = [np.arange(1,N[0],1),np.arange(1,N[1],1),np.arange(1,N[2],1)]
    KP = [set(np.around(abs(np.where(I>NN[i], I-N[i], I))/N[i], decimals=dmprecesion)) for i, I in enumerate(kp)]
    KP2intersection = [ i[0] & i[1] for i in combinations(KP,2) ]
    KP3intersection = KP[0] & KP[1] & KP[2]   
    
    KPP = defaultdict(list)
    KPP[frozenset([0.0])] = [[0,0,0]]
    for i, I in enumerate(KP):
        for J in I:
            kptemp = [0,0,0]
            kptemp[i]=J
            KPP[frozenset([J])].append(kptemp) 
            
    for I in KP3intersection:
        KPP[frozenset([I])].append([I]*3)
            
    for i, I in enumerate(KP2intersection):
        x = list(I)
        kptemp = np.vstack((x,x,x)).T
        kptemp[:,2-i]= 0
        for J in kptemp:
            KPP[frozenset(J)].append(J)
    
    return list(KPP.values())   

def FindSymmetryEquivalentKPpos(FileSpecificKP, FoldFactor, CompareMean, dmprecesion=6, structure='zincblende'):
    """
    This function find the symmetry equivalent positions in the current file k-point
    list.

    Parameters
    ----------
    FileSpecificKP : Float array/list
        The k-point coordinate list for the current file.
    FoldFactor : Integer list/array
        The supercell dimensions (X, Y, Z).
    CompareMean : Bool
        Whether the symmetry averaged or maximum BW (among symmetry equivalent 
        k-points) will be calculated.
    dmprecesion : Integer, optional
        The decimal precision for rounding. The default is 6.
    structure : String, optional
        The crystal structure. The default is 'zincblende'.

    Returns
    -------
    Integer nd-list
        The list containg the position index of the symmetry equivalent k-points.

    """
    if 'zincblende' in structure:
        KPlist = SymmetryEquivKpoints_zincblende_Gfolding(FoldFactor, dmprecesion=dmprecesion)
    else:
        sys.exit("Error: Only zincblende struture is implemented so far.")
      
    FileKpPos = defaultdict(list)
    for i, I in enumerate(FileSpecificKP):
        notsymmetric = True
        for j, J in enumerate(KPlist):
             for JJ in J:
                 if np.allclose(abs(I),JJ): 
                     FileKpPos[j].append(i)
                     notsymmetric = False
                     
        if notsymmetric: 
            FileKpPos[i].append(i) if CompareMean  else FileKpPos['other'].append(i)
        
    return list(FileKpPos.values())

def CollectBWdata(BW_DATA, BW_posgroup_array, CompareMean):
    """
    This function calculates the symmetry averaged/maximum of the BWs.

    Parameters
    ----------
    BW_DATA : Float array
        Array containing the Bloch weights for a band.
    BW_posgroup_array : Integer array
        Array containing the index group based on symmetry equivalence.
    CompareMean : Bool
        Whether the symmetry averaged or maximum BW (among symmetry equivalent 
        k-points) will be calculated.

    Returns
    -------
    Float 2d-array
        The array containg the mean/maximum BWs of the symmetry equivalent k-points.
        (Mean/Max BWs, position of the mean/Max BWs). For Mean BWs the 1st k-point
        position in the symmetry group will be used.

    """
    return np.array([[round(np.mean(BW_DATA[ff]), 4), ff[0]] if CompareMean \
            else [round(max(BW_DATA[ff]), 4), ff[np.argmax(BW_DATA[ff])]] for ff in BW_posgroup_array]\
                    ,dtype=object)

def CheckCutoffBWcondition(BW_data_group, BW_cutoff):
    """
    This function checks the maximum BW criteria for DIT error.

    Parameters
    ----------
    BW_data_group : Float array
        The BWs array/list.
    BW_cutoff : Float
        The cutoff Bloch weight.

    Returns
    -------
    Bool
        Whether the maximum of the BWs is greater than the cutoff BW or not!

    """
    return True if max(BW_data_group) > BW_cutoff else False
 
def RedefineBand(BW_DATA, BW_posgroup_array, BW_cutoff, CompareMean):
    """
    This function calculates the Bloch weights of redefined bands.

    Parameters
    ----------
    BW_DATA : Float array
        The BWs array/list.
    BW_posgroup_array : Integer nd-array
        Array containing the position of the grouped symmetry equivalent k-point positions.
    BW_cutoff :  Float
        The cutoff Bloch weight.
    CompareMean : Bool
        Whether the symmetry averaged or maximum BW (among symmetry equivalent 
        k-points) will be calculated.

    Returns
    -------
    Float nd-array or None
        The array containg the mean/maximum BWs of the symmetry equivalent k-points.
        If Bloch weight condition is not satisfied then return None. This criterion
        will be used for redefinition criterion.

    """
    BW_data_group = CollectBWdata(BW_DATA, BW_posgroup_array, CompareMean)
    BWcondition = CheckCutoffBWcondition(BW_data_group[:,0], BW_cutoff)
    return BW_data_group if BWcondition else None
       
def PureCBVBband(BW_DATA, BW_posgroup_array, BW_cutoff, CompareMean):
    """
    This function calculates the Bloch weights of pure CB or VB.

    Parameters
    ----------
    BW_DATA : Float array
        The BWs array/list.
    BW_posgroup_array : Integer nd-array
        Array containing the position of the grouped symmetry equivalent k-point positions.
    BW_cutoff : Float
        The cutoff Bloch weight.
    CompareMean : Bool
        Whether the symmetry averaged or maximum BW (among symmetry equivalent 
        k-points) will be calculated.

    Returns
    -------
    BW_data_group : Float nd-array
        The array containg the mean/maximum BWs of the symmetry equivalent k-points.
    BWcondition : Bool
        If Bloch weight cutoff condition is satisfied or not. This criterion
        will be used for the redefinition criterion.

    """
    BW_data_group = CollectBWdata(BW_DATA, BW_posgroup_array, CompareMean)
    BWcondition = CheckCutoffBWcondition(BW_data_group[:,0], BW_cutoff)
    return BW_data_group, BWcondition
    
def FindBandM(FullData, BW_group, BW_cutoff, Band_index, CompareMean, ConductionBand=True):
    """
    This function finds the CBM and VBM. If the BWs of the original bands doesn't 
    satisfy the minimum Bloch weight criteria given by BW_cutoff then new CB and
    /or VB is defined recursively.

    Parameters
    ----------
    FullData : Float array
        The array containing the band energy and Bloch weights of the band.
    BW_group : Integer nd-array
        Array containing the position of the grouped symmetry equivalent k-point positions.
    BW_cutoff : Float
        The cutoff Bloch weight.
    Band_index : Integer
        The index of the band you want to examine.
    CompareMean : Bool
        Whether the symmetry averaged or maximum BW (among symmetry equivalent 
        k-points) will be calculated.
    ConductionBand : Bool, optional
        Whether the band is conduction band or not(valence band). The default is True.

    Returns
    -------
    Tuple
        (Band energy, Bloch weights after averaging/maximizing, All Bloch weights),
        (Band energy, Bloch weights after averaging/maximizing, All Bloch weights for redefined band.
         If redefinition is not needed then return None).

    """
    BW_data = FullData[Band_index][:, 1] * 100 # BW Percentage conversion
    Energy = FullData[Band_index][0,0]
    band_bw_data, notredefine = PureCBVBband(BW_data, BW_group, BW_cutoff, CompareMean)
    if not notredefine:
        bw_data = FullData[Band_index+1:] if ConductionBand else FullData[Band_index-1::-1]
        for _, datatmp in enumerate(bw_data):
            bww_data = datatmp[:, 1] * 100
            redefine_BW = RedefineBand(bww_data, BW_group, BW_cutoff, CompareMean) 
            if redefine_BW is not None: break
        redefine_energy = datatmp[0,0]
        
    return (Energy, band_bw_data, np.around(BW_data, decimals=4)), \
        None if notredefine else (redefine_energy, redefine_BW, np.around(bww_data, decimals=4))

def BandgapProperty(CBdata, VBdata, FileSpecificKP):
    """
    This function calculates the properties of Conduction band and Valence band.
    D:  Direct bandgap at the Gamma point.
    I:  Indirect bandgap with VBM at the Gamma point.
    i:  Indirect bandgap with VBM at k-point other than Gamma point.

    Parameters
    ----------
    CBdata : 
        (Band energy, Bloch weights after averaging/maximizing, All Bloch weights) for CB.
    VBdata : 
        (Band energy, Bloch weights after averaging/maximizing, All Bloch weights) for VB.
    FileSpecificKP : Float array/list
        The k-point coordinate list for the current file.

    Returns
    -------
    Farray : [bandgap, bandgap nature, BWs at CB-VB]
        BW list: CBBW@VBpos, VBBW@VBpos, CBBW@CBpos, VBBW@CBpos. If direct only 1 pair
        is returned.

    """
    Farray = []
    for I in CBdata:
        if I is None:
            Farray.append(None)
            Farray.append(None)
        else:
            for J in VBdata:
                if J is None:
                    Farray.append(None)
                else:
                    BW1_pos = I[1][np.argmax(I[1][:,0]), 1]
                    BW2_pos = J[1][np.argmax(J[1][:,0]), 1]
                    BandGap = round(I[0] - J[0], 4)
                    CheckGpoint = np.allclose(FileSpecificKP[BW2_pos], [0, 0, 0])
                    if BW1_pos==BW2_pos:
                        EgNature = 'D' if CheckGpoint else 'd'
                        BWARRAY = [I[2][BW2_pos], J[2][BW2_pos]]
                    else:
                        EgNature = 'I' if CheckGpoint else 'i' 
                        # CBBW@VBpos, VBBW@VBpos, CBBW@CBpos, VBBW@CBpos
                        BWARRAY = [I[2][BW2_pos], J[2][BW2_pos], I[2][BW1_pos], J[2][BW1_pos]]
                    Farray.append([BandGap, EgNature, BWARRAY])                    
    return Farray  
        
def CB_VB_data(FullData, BW_posgroup_array, FileSpecificKP, CB_index, BW_cutoff, CompareMean):    
    """
    This function calculates the properties of (redefined)Conduction band and (redefined)Valence band.

    Parameters
    ----------
    FullData : Float 2d-array
        Array containes the band energy and Bloch weights.
    BW_posgroup_array : Integer nd-array
        Array containing the position of the grouped symmetry equivalent k-point positions.
    FileSpecificKP : Float array/list
        The k-point coordinate list for the current file.
    CB_index : Integer
        The index of the conduction band == No. of electron(/2) for Noncolliner 
        /collinear (non spin polarized).
    BW_cutoff : Float
        The cutoff Bloch weight. This will be used to determine the DIT horizontal
        error.
    CompareMean : Bool
        Whether the symmetry averaged or maximum BW (among symmetry equivalent 
        k-points) will be calculated.

    Returns
    -------
    [bandgap, bandgap nature, BWs at CB-VB]
        BW list: CBBW@VBpos, VBBW@VBpos, CBBW@CBpos, VBBW@CBpos. If direct only 1 pair
        is returned.

    """
    CBdata = FindBandM(FullData, BW_posgroup_array, BW_cutoff, CB_index, CompareMean)
    VBdata = FindBandM(FullData, BW_posgroup_array, BW_cutoff, CB_index-1, CompareMean, ConductionBand=False) 
    return BandgapProperty(CBdata, VBdata, FileSpecificKP)
           
def CreateData(fname, FoldFactor, CB_index, CompareMean=True, BW_cutoff=20):
    """
    This function calculates the bandstructure properties
    (band gap, bandgap nature, BWs at the VBM-CBM).

    Parameters
    ----------
    fname : String
        The filename containing the unfolded k-points (e.g. WAVECAR_spinor.f2b).
    FoldFactor : Integer list
        Supercell dimensions (e.g. [6, 6, 6]).
    CB_index : Integer
        The index of the conduction band == No. of electron(/2) for Noncolliner 
        /collinear (non spin polarized).
    CompareMean : Bool, optional
        Whether the symmetry averaged or maximum BW (among symmetry equivalent 
        k-points) will be calculated. The default is True.
    BW_cutoff : Float, optional
        The cutoff Bloch weight. This will be used to determine the DIT horizontal
        error. The default is 20.

    Returns
    -------
    dict
        [bandgap, bandgap nature, BWs at CBM-VBM], [bandgap, bandgap nature, BWs at CBM-rVBM],
        [bandgap, bandgap nature, BWs at rCBM-VBM], [bandgap, bandgap nature, BWs at rCBM-rVBM].
        None is returned if redefinition doesn't exist.
        BW list: CBBW@VBpos, VBBW@VBpos, CBBW@CBpos, VBBW@CBpos. If direct only 1 pair
        is returned.

    """
    FullData = np.genfromtxt(fname)
    NUNFOLDED = FoldFactor[0]*FoldFactor[1]*FoldFactor[2]
    FileSpecificKP = FullData[:NUNFOLDED,:3]
    BW_posgroup_array = FindSymmetryEquivalentKPpos(FileSpecificKP, FoldFactor, CompareMean)
    FullData = np.split(FullData[:,3:], len(FullData)//NUNFOLDED)
    FinalData = CB_VB_data(FullData, BW_posgroup_array, FileSpecificKP, CB_index, BW_cutoff, CompareMean)   
    return {'BP': FinalData}

#------------------------------------------------------------------------------
"""
For streched zincblende:
x, y, z == Lattice parameter of streched zincblende structure.
xy, yz, and xz == Diagonals of streched zincblende or 2 times lattice vactors
Note1: Angle b/w x, y, z are always 90 degree in (streched)zincblende structure
        Angle b/w xy, xz and yz might be differ from 60 in streched case
Note2: 6x6 primitive supercell is equivalent to 3x3 unit cell for zincblende

(2*xy)**2 = x**2 + y**2; (2*yz)**2 = y**2 + z**2; (2*xz)**2 = x**2 + z**2

Solving above 3 equations we get;
        y**2 = xy**2 + yz**2 - xz**2
        x**2 = xy**2 + xz**2 - yz**2
        z**2 = yz**2 + xz**2 - xy**2
"""
def vec(a):
    """
    Calculates the lattice vector magnitude.

    Parameters
    ----------
    a : Float
        [x, y, and z] coordinate of the lattice vector.

    Returns
    -------
    Float
        Lattice vector magnitude.

    """
    return (np.sqrt(np.sum(a**2, axis=1)))

def CalculateLatticeParameter(vector,crys_str):
    """
    This function calculates the lattice parameter for a given crystal structure. So far only
    cubic and zincblende structure is implemented.

    Parameters
    ----------
    vector : Float
        3 lattice vector magnitudes.
    crys_str : String
        The crystal structure.

    Returns
    -------
    lp : Float
        3 Lattice parameters.

    """
    if(crys_str == 'cubic'):
        lp = vector
    elif(crys_str == 'zincblende'): 
        nv = vector**2
        lp =  np.array([np.sqrt(nv[0] + nv[2] - nv[1]),
                        np.sqrt(nv[0] + nv[1] - nv[2]),
                        np.sqrt(nv[1] + nv[2] - nv[0])])

        lp *= np.sqrt(2)
    else:
        print("Only supports cubic,zincblende. The other symmetry are not implemented yet.")
        sys.exit()
    
    return lp

def getLP(file, SupercellDimension, ClusterType):
    """
    This function reads the lattice vector file (e.g. POSCAR) and calculates lattice
    parameters using vec() and CalculateLatticeParameter().

    Parameters
    ----------
    file : String filename
        Filename (e.g. POSCAR).
    SupercellDimension : Integer
        Dimensions of supercell in array.
    ClusterType : String
        Common name of the crystal system. So far only cubic and zincblende structure is implemented.

    Returns
    -------
    lp : Float
        3 Lattice parameters.

    """
    fac = np.genfromtxt(file, max_rows=1, skip_header=1)
    lvec = np.genfromtxt(file, max_rows=3, skip_header=2) * fac
    plvec = np.divide(vec(lvec), SupercellDimension)
    lp = CalculateLatticeParameter(plvec, ClusterType)
    return lp

def lpEquality(lp):
    """
    This function checks if the 3 lattice parameters are same or not?

    Parameters
    ----------
    lp : Float
        3 lattice parameters.

    Returns
    -------
    Bool
        All the lattice parameters are same or not?

    """
    pre = 1e-3
    lpdiff = [isclose(lp[0], lp[1], abs_tol=pre), isclose(lp[1], lp[2], abs_tol=pre)]
    return True if np.all(lpdiff) else False

def getEqmLP(file, SupercellDimension, ClusterType='zincblende'):
    """
    This calculates the lattice parameters using getLP() and then checks if the lattice 
    parameters are same or not using lpEquality() and returns one of the lattice parameter.

    Parameters
    ----------
    file : String filename
        Filename (e.g. POSCAR).
    SupercellDimension : Integer
        Dimensions of supercell in array.
    ClusterType : String
        Common name of the crystal system. So far only cubic and zincblende structure is implemented.

    Returns
    -------
    Float
        One of the lattice parameter.

    """
    latticeparam = getLP(file, SupercellDimension, ClusterType)
    if ClusterType=='zincblende': assert lpEquality(latticeparam), \
        f"The 3 equilibrium lattice parameters seem to be different for {file}."
    return round(latticeparam[0],4)

#------------------------------------------------------------------------------
#********** Post processing functions *****************************************
# https://stackoverflow.com/a/19871956
def lookupkey(mydict, mykey):
    """
    This function checks if a keyword exists in a 'nested' dictionary.

    Parameters
    ----------
    mydict : dict
        Nested dictionary.
    mykey : String
        Key you are looking for.

    Yields
    ------
    Dictionary value
        Value corrsponding to the key supplied. Returns none if the key doesn't exists..

    """
    if isinstance(mydict, list):
        for K in mydict:
            for x in lookupkey(K, mykey):
                yield x
    elif isinstance(mydict, dict):
        if mykey in mydict:
            yield mydict[mykey]
        for nestdict in mydict.values():
            for x in lookupkey(nestdict, mykey):
                yield x
                
def Float2String_BW(X):
    """
    This function can be used to convert the float Bloch weights to integer strings.

    Parameters
    ----------
    X : Float
        Arry to be converted.

    Returns
    -------
    String array with rounded integer conversion.

    """
    return np.round(X).astype(int)
                
def getNC(file, species, Nspecies): # Which POSCAR file and which element concentration?
    elements = np.genfromtxt(file, max_rows=2, skip_header=5, dtype=str)
    Number = elements[1].astype(float)
    pos = np.where(elements[0] == species)
    conc = sum(Number[pos])/Nspecies * 100.
    return "{:.2f}".format(conc)

def Merge2Dict(mydict1, mydict2):
    """
    This function merges two nested dictionary.

    Parameters
    ----------
    mydict1 : Dictionary 1
    mydict2 : Dictionary 1

    Returns
    -------
    mydict1 : Merged dictionary

    """
    for key1 in mydict2: # Concentration loop
        for key2 in mydict2[key1]: # Strain loop
            for key3 in mydict2[key1][key2]: # Configuration loop
                mydict1[key1][key2][key3] = mydict2[key1][key2][key3]
    return mydict1

#------------------------------------------------------------------------------
def ReturnBandGapProperty(mylist, cutoff):
    """
    This function returns the nature of bandgap(e.g. directness) determined by the cutoff.

    Parameters
    ----------
    mylist : Float list
        The list containing the original nature of bandgap.
    cutoff : Float
        Cut-off criteria in BW that will be used for DIT error determination.

    Returns
    -------
    Eg : Float
        The bandgap value.
    BN : String
        The nature of bandgap. If the original bandgap is of indirect in nature then
        the a value of directness is calculated based on cutoff BW which will then
        be used for DIT horizontal error region construction.

    """
    Eg = mylist[0];   BN = mylist[1].upper()       
    if mylist[1].upper() == 'I':
        # CBBW@VBpos, VBBW@VBpos, CBBW@CBpos, VBBW@CBpos
        if (mylist[2][0]>cutoff and mylist[2][1]>cutoff) \
            or (mylist[2][2]>cutoff and mylist[2][3]>cutoff): BN = 'ID'
        
    return Eg, BN

def BandgapProperty_post(AA, cutoff):
    """
    This function collects the bandgap and nature of bandgaps.

    Parameters
    ----------
    AA : Dict
        The dictionary containing the bandgap and Bloch weights.
    cutoff : Float
        Cut-off criteria in BW that will be used for DIT error determination.

    Returns
    -------
    Bandgap : Float list
        The bandgaps for original CB-VB and redefined CB-VB. For the redefined CB-CB
        if both CB and VB are redefined then the bandgap will be returnted from both
        redefined bands (rCB-rVB) else if either CB or VB is redefined then rCB-VB or CB-rVB
        will be used. If no redefinition exists then a copy of CB-VB is returned for
        the 2nd set.
    BandgapNature : String list
        Type of bandgap. Same as 'Bandgap'.

    """
    Eg, BN = ReturnBandGapProperty(AA['BP'][0], cutoff)
    Bandgap, BandgapNature = [Eg], [BN]
    for i in AA['BP'][1:]:
        if i is not None:
            Eg, BN = ReturnBandGapProperty(i, cutoff)
            
    Bandgap.append(Eg)
    BandgapNature.append(BN)
    return (Bandgap, BandgapNature)

def CheckDIT(tmptest, oldNE, JJ, oldStrain, dit, diter):
    """
    This function calculates the DIT points based on nature of bandgap.
    D:  Direct bandgap at the Gamma point.
    I:  Indirect bandgap with VBM at the Gamma point.
    i:  Indirect bandgap with VBM at k-point other than Gamma point.
    ID: Indirect bandgap with enough BW(given by cutoff) for the direct transition.

    Parameters
    ----------
    tmptest : String
        The nature of current bandgap.
    oldNE : String
        The nature of bandgap for the previous strain point.
    JJ : Float
        Current strain point.
    oldStrain : Float
        The strain for the previous point.
    dit : Float list
        The strain amount for the direct-indirect transition point. The first 'D'
        strain point when ID/I-D transition happens, will be used for the DIT.
    diter : Float list
        The error in DIT determination calculated from cutoff. 
        The first 'ID' strain point when ID-I or first 'D' strain point when I-D 
        transition happens, will be used for the DIT error.

    Returns
    -------
    oldNE : String
        The updated nature of previous bandgap.
    oldStrain : Float
        The updated strain for the previous point.
    dit : Float list
        The updated DIT error list. The strain amount for the direct-indirect 
        transition point. The first 'D' strain point when ID/I-D transition 
        happens, will be used for the DIT.
    diter : Float list
        The updated DIT error list. The error in DIT determination calculated from cutoff. 
        The first 'ID' strain point when ID-I or first 'D' strain point when I-D 
        transition happens, will be used for the DIT error.


    """
    # ...I...ID...D...ID...I...ID...D...ID...I...
    if tmptest != oldNE and oldNE is not None: # The oldNE1 condition is to avoid 1st loop.
        if tmptest=='D':
            dit.append(JJ)
            if oldNE=='I': diter.append(JJ)
        elif tmptest=='ID':
            dit.append(oldStrain) if oldNE=='D' else diter.append(JJ)
        elif tmptest=='I' and oldNE=='D':
            dit.append(oldStrain)
            diter.append(oldStrain)
        else:
            diter.append(oldStrain)

    oldNE = tmptest
    oldStrain = JJ
    return oldNE, oldStrain, dit, diter

def CalculateDITs(data, cutoff=20):
    """
    This functions collects/calculates the different DIT parameters.

    Parameters
    ----------
    data : Dictionary
        Data dictionary from json file read.
    cutoff : Float
        Minimum amount of Bloch weight that will be used as the creterion in determining
        bandgap directness.

    Returns
    -------
    BPD : Dictionary
        The bandgap phase diagram parameters.
    MixedConfig : Bool
        Whether multiple configurations exist even in one of the concetration-strain dictionary.

    """
    BPD = NestedDefaultDict() # Bandgap Phase Diagram==BPD
    MixedConfig = False
    for J,_ in sorted(data.items(), key=lambda t: float(t[0])): # N concentration loop
        newConcKey = data[J]['0']['01']['NK']
        dit1, dit2, dit1er, dit2er = [], [], [], []
        oldStrain1 = oldStrain2 = oldNE2 = oldNE1 = None
        for JJ, ndict in sorted(data[J].items(),key=lambda t: float(t[0])): # strain loop
            llist = np.array([BandgapProperty_post(ndict[JJJ], cutoff) for JJJ in ndict], dtype=object)
            
            singleConfig = True if len(ndict.keys()) else False
            if singleConfig==False: MixedConfig = True 
            
            NatureEg1, NatureEg2 = set(llist[:,1,0]), set(llist[:,1,1]) 
            # This condition make sures that you have only single type of bandgap
            # for all the SQSs for a particular Concentration/Strain/
            if len(NatureEg1) >1 :
                print(f"Warning: For {J}, {JJ} strain configuraions have multiple bandgap nature of bangap type-1.")
                sys.exit("Error: Not implemented yet.")
            if len(NatureEg2) >1 :
                print(f"Warning: For {J}, {JJ} strain configuraions have multiple bandgap nature of bangap type-2.")
                sys.exit("Error: Not implemented yet.")
            if len(NatureEg1) == 1 and len(NatureEg2) == 1:
                NE1, NE2 = next(iter(NatureEg1)), next(iter(NatureEg2))
                if singleConfig:
                    BPD[newConcKey]['BS'][JJ] = [float(llist[:,0,0]), float(llist[:,0,1]), 0, 0] #, NE1, NE2] 
                else:
                    BPD[newConcKey]['BS'][JJ] = [np.mean(llist[:,0,0]), np.mean(llist[:,0,1]), 
                                        np.std(llist[:,0,0]), np.std(llist[:,0,1])] #, NE1, NE2] 
                # Calculate DIT 
                #**************************************************************
                # This is for 1st type of Bandgap
                oldNE1, oldStrain1, dit1, dit1er = CheckDIT(NE1.upper(), oldNE1, JJ, oldStrain1, dit1, dit1er)

                # This is for 2nd type of Bandgap : Redefined
                oldNE2, oldStrain2, dit2, dit2er = CheckDIT(NE2.upper(), oldNE2, JJ, oldStrain2, dit2, dit2er)
                #**************************************************************
                
                
        LParray = list(lookupkey(data[J]['0'], 'LP'))
        BPD[newConcKey]['DIT2'] = dit2
        BPD[newConcKey]['DIT1'] = dit1
        BPD[newConcKey]['DIT1err'] = dit1er
        BPD[newConcKey]['DIT2err'] = dit2er
        #BPD[newConcKey]['DIT1verr'] = dit1ver
        #BPD[newConcKey]['DIT2verr'] = dit2ver
        # Equilibrium lattice parameter needed for substrate strain calculation.
        BPD[newConcKey]['eqmLP'] = np.mean(LParray) if len(LParray)>1 else LParray[0]
        
    return BPD, MixedConfig

def RearrangeDIT(NC, DIT):
    """
    This function rearranges the DIT points.

    Parameters
    ----------
    NC : String or float list or array
        Concentration array.
    DIT : String or float list or array
        List of DIT points.

    Returns
    -------
    conc1array : Float array
        Rearranged concentration array.
    ditarray : Float array
        Rearranged DIT points.

    """
    concarray = [ [NC[I]]*len(DIT[I]) for I in range(len(NC)) if DIT[I] ]      
    conc1array  = np.array([val for sublist in concarray for val in sublist], dtype=float)
    ditarray = np.array([val for sublist in DIT for val in sublist], dtype=float)
    
    assert len(conc1array) == len(ditarray), 'After rearrange conc. and DIT array length does not match'
               
    return conc1array, ditarray  

def PrintDITs(NC, DIT1, DIT2, PrintDIT = False):
    """
    This function can be used to print the DITs.

    Parameters
    ----------
    NC : String or Float array
        The concentration array.
    DIT1 : String or float array
        The DIT points.
    DIT2 : String or float array
        The DIT points.
    PrintDIT : Bool, optional
        DO you want to print the DITs? The default is False.

    Returns
    -------
    None.

    """
    if PrintDIT:
        print("\nConcentration\tDIT\t\tDIT(redefined)")
        for I, J, K in zip(NC,DIT1,DIT2):
            print(f"{I} ==> {J} {' '*4} {K}")
    return 

def GetStrainBandgapStd(BPD):
    """
    

    Parameters
    ----------
    BPD : Dictionary
        Dictionary contains the necessary data in the format of return value from CalculateDITs() function.

    Returns
    -------
    STRAIN : Float array
        The Strain array.
    BANDGAP1 : Float array
        The bandgap array calculated from actual VB and CB.
    BANDGAP2 : Float array
        The bandgap array calculated from 'redefined' VB and CB.
    STDERROR1 : Float array
        The standered error in bandgap for multiple configuration, calculated from actual VB and CB.
    STDERROR2 : Float array
        The standered error in bandgap for multiple configuration, calculated from 'redefined' VB and CB.
    NC : String list
        The list of strings of concentrations.

    """
    STRAIN, BANDGAP1, BANDGAP2, STDERROR1, STDERROR2 = [], [], [], [], []   
    for J in BPD:
        strain = list(BPD[J]['BS'].keys())
        bandgap_all = np.array([BPD[J]['BS'][JJ] for JJ in strain])
        
        STRAIN.append(np.array(strain, dtype=float))
        BANDGAP1.append(bandgap_all[:,0])
        BANDGAP2.append(bandgap_all[:,1]) # From redefinition bands
        STDERROR1.append(bandgap_all[:,2])
        STDERROR2.append(bandgap_all[:,3])
    NC = list(BPD.keys())   
    return NC, STRAIN, BANDGAP1, BANDGAP2, STDERROR1, STDERROR2

def CreateHeatmapData(NC, STRAIN, BANDGAP, heatmap_bound=False, heatmap_strain_bound=None, \
                 Conc_bound=(0,100), G_filtering=True, Sigma=2):
    """
    This function calculates the necessary data for final heatmap in BPD.

    Parameters
    ----------
    NC : String or float array or list
        The concentration array.
    STRAIN : Float array
        Strain array.
    BANDGAP : Float array
        Bandgap arryay
    heatmap_bound : Bool, optional
       Whether the heatmap will be bound in strain or concentration? default is False.
    heatmap_strain_bound : Float, optional
        Defines the cutoff strain for heatmap. The default is None.
    Conc_bound : Float tuple (lower limit, upper limit), optional
        Defines the limits in concentration for bandgap. The default is (0,100).
    G_filtering : Bool, optional
        Gaussian filtering in the data. The default is True.
    Sigma : Scalar, optional
        Defines the sigma value in gaussian filtering. The default is 2.
        
    Returns
    -------
    filtered_arr : Float array
        Final heat map array.
    list [x_left, x_right, y_bottom, y_top]
        Extent of x and y axis.

    """
    Xarray=[]
    for ii in range(0,len(NC)):
        xxaxis = [NC[ii]]*len(STRAIN[ii])
        Xarray+= xxaxis
    
    Xarray = np.array(Xarray, dtype=float)
    strainarray = np.concatenate(STRAIN, axis=0)
    Bandgaparray = np.concatenate(BANDGAP, axis=0) 
    
    if heatmap_bound:
        if heatmap_strain_bound:
            strainarray_cutoff_ind = np.argwhere(abs(strainarray)>heatmap_strain_bound)
    
        if Conc_bound:
            NC_index = np.argwhere(((Xarray>=Conc_bound[0]) & (Xarray<=Conc_bound[1])))  
            
        strainarray_cutoff_ind = np.concatenate((strainarray_cutoff_ind,NC_index))
        
        strainarray = np.delete(strainarray, strainarray_cutoff_ind)
        Bandgaparray = np.delete(Bandgaparray, strainarray_cutoff_ind)
        Xarray = np.delete(Xarray, strainarray_cutoff_ind)

    extenty_b = np.amin(strainarray)
    extenty_t = np.amax(strainarray) 
    grid_x, grid_y = np.mgrid[0:100:100j, extenty_b:extenty_t:40j]

    points = np.stack((Xarray, strainarray),axis=-1)
    grid_z0 = griddata(points,Bandgaparray, (grid_x, grid_y), method='nearest')
    filtered_arr=gaussian_filter(grid_z0.T, sigma=Sigma) if G_filtering else grid_z0.T
    return filtered_arr, [0,100,extenty_b,extenty_t]

def DITpolyfit(XX, YY, order=3, xmin=None, xmax=None):
    """
    This function use polynomial fitting on the DIT.

    Parameters
    ----------
    XX : Float array
        X-numpy array.
    YY : Float array
        Y-numpy array.
    order : Integer, optional
        Polynomial order. The default is 3.
    xmin : Float, optional
        Minimum in X-data. The default is None.
    xmax : Float, optional
        Maximum in X-data. The default is None.

    Returns
    -------
    Float numpy array, Float numpy array
        The X and Y array after polynomial fitting.

    """
    z = np.polyfit(XX, YY, order)
    p = np.poly1d(z)
    if xmin is None: xmin=XX[0]
    if xmax is None: xmax=XX[-1]
    xx = np.linspace(xmin,xmax,100)
    return xx, p(xx)