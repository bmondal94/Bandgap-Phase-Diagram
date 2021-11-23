#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:20:00 2021

@author: bmondal
"""

#%%-------------------- Importing modules -------------------------------------
import numpy as np      
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import animation
import BPDFunctions as mf
import BPDPlotFunctions as mpf
import json

#%%------------------- Load data from database --------------------------------
dirnamee = '/home/bmondal/MyFolder/VASP/'
filename_pre = 'GaPSb/DATA/'
filename = filename_pre+'/BPD_DataBase_mpi_2021-11-23-18:05:04_BW10.0_max.json'
figfile = dirnamee+'GaPSb/Figs/'

Setup_text = ['BW10.0_mean', 'BW10.0_max', 'BW20.0_mean', 'BW20.0_max']
#****************************************
#titletext = "Biaxial strain (001)"
titletext = Setup_text[3]
#****************************************

filepath = dirnamee + filename
with open(filepath,'r') as f:
    data = json.load(f)
    
#%%------------------- Manual data analysis -----------------------------------
#sorted(data['75'].items(),key=lambda t: float(t[0]))

#%%---------- Create the dictionary for DIT, bandgap and strain ---------------  
BPD, MixedConfig = mf.CalculateDITs(data)       

#%% ------------------------- Data collection ---------------------------------
NC, STRAIN, BANDGAP_type1, BANDGAP_type2, STDERROR_type1, STDERROR_type2 = mf.GetStrainBandgapStd(BPD)
DIT2 = list(mf.lookupkey(BPD, 'DIT2')) # From redefinition bands.
DIT1 = list(mf.lookupkey(BPD, 'DIT1'))
EqmLP = np.array(list(mf.lookupkey(BPD, 'eqmLP')))
assert len(NC) == len(DIT1) == len(DIT2) == len(EqmLP), 'Mismatch among length of conc., DIT and Eqm. lattice parameter arrays.'

mf.PrintDITs(NC, DIT1, DIT2, PrintDIT = True)

rearrange_NC,rearrange_DIT  = mf.RearrangeDIT(NC, DIT1)
rearrange_NC_re,rearrange_DIT_re  = mf.RearrangeDIT(NC, DIT2)

pos_ = np.argsort(rearrange_DIT_re)
YY = rearrange_DIT_re[pos_]
XX = rearrange_NC_re[pos_]
yy, xx = mf.DITpolyfit(YY, XX)

#-----------
DIT2err = list(mf.lookupkey(BPD, 'DIT2err')) # From redefinition bands.
DIT1err = list(mf.lookupkey(BPD, 'DIT1err'))
assert len(NC) == len(DIT1err) == len(DIT2err) == len(EqmLP), 'Mismatch among length of conc., DITerr and Eqm. lattice parameter arrays.'
rearrange_NC_er,rearrange_DIT_er  = mf.RearrangeDIT(NC, DIT1err)
rearrange_NC_re_er,rearrange_DIT_re_er  = mf.RearrangeDIT(NC, DIT2err)

mf.PrintDITs(NC,DIT1err,DIT2err, PrintDIT = True)

pos_ = np.argsort(rearrange_DIT_re_er)
YYer = rearrange_DIT_re_er[pos_]
XXer = rearrange_NC_re_er[pos_]
yyer, xxer = mf.DITpolyfit(YYer, XXer)

#%%-------------- substrate strain calculations -------------------------------       
subname = ['GaAs','GaP','Si','Test']
sublattice = np.array([[5.689],[5.475],[5.42103],[5.612]]) # Substrate lattice parameters
subEg = [1.4662, 2.3645, 1.1915, 0]
myco = ['lightsteelblue','red','cornflowerblue','royalblue']
myline = ['dotted','dashed','dashdot','loosely dashed']
substrain = (sublattice[0:-1] - EqmLP)/EqmLP*100   
  
#%%------------- Plottings ----------------------------------------------------
#%%********************** Eg vs Strain ****************************************
xlabeltext = "In-plane strain (%)"
ylabeltext = "E$_{\mathrm{g}}$"
Both=True

if MixedConfig:
    fig, ax = mpf.plotEgStrainErrorBar(STRAIN, BANDGAP_type1, BANDGAP_type2, \
                                 NC,STDERROR_type1, STDERROR_type2, \
                                     ylabel=ylabeltext, xlabel=xlabeltext, Both=Both, title=titletext)
else:
    fig, ax = mpf.plotEgStrain(STRAIN, BANDGAP_type1, BANDGAP_type2, NC, \
                         ylabel=ylabeltext, xlabel=xlabeltext, Both=Both, title=titletext)


#%%*************** Heatmap for BPD ********************************************
filtered_arr, extent = mf.CreateHeatmapData(NC, STRAIN, BANDGAP_type1)
NC = np.array(NC, dtype=float)
_ = mpf.PlotBPD(NC, substrain, subname, subEg, myco, myline, XX, YY,
            filtered_arr, extent, Concfitdata=xx, DITfitdata=yy, DITfit=True, \
                Concfitdataerr=xxer, DITfitdataerr=yyer, DITfitErr=True, ErrLabel='20%',  \
                    DrawSub=True, drawheatmap=True, xlabel='Sb (%)',ylabel='Strain(%)', fignum=None, title=titletext)
    
# plt.savefig(dirnamesavefig+'/GaAsP-Eg-Phasediag-Biaxialstrain-sub_paper_color.eps', \
#     format='eps',bbox_inches = 'tight',dpi=300)    

#%%--------------- HTML file generation --------------------------------------- 
CreateHTML = True; substrate=True; strain0line=True
if CreateHTML:
    #p.toolbar.logo = None
    from bokeh.plotting import output_file, save
    
    tooltips = [("Sb", "$x %"), ("Strain", "$y %"), ("Bandgap", "@image eV")]
    inter_img = mpf.web_heatmap(filtered_arr, extent, tooltips, clabel="Eg") 
    
    DITs = mpf.web_dit_scatter(XX,YY,xlabel="Sb concentration (%)",ylabel="Strain (%)"\
                           ,title="GaPSb isotropic strain",fontsize=20)
    curve = mpf.web_dit_fit(xx, yy, label='Polyfit', fontsize=15) 
    curve2 = mpf.web_diterr_fit(xxer, yyer, label='20%', fontsize=15) 
    text1 = mpf.web_bpd_text(75,1, text='Indirect') 
    text2 = mpf.web_bpd_text(15,1, text='Direct')
    if strain0line: Eqmline = mpf.web_eqm_line(yeqm=0)
    
    overlay_list = [inter_img, DITs, curve, curve2, text1, text2, Eqmline]
    if strain0line: 
        Eqmline = mpf.web_eqm_line(yeqm=0)
        overlay_list.append(Eqmline)
    if substrate:
        suboverlay = mpf.web_plot_substrate(NC, substrain, subname, myco, myline, extent)
        overlay_list.append(suboverlay)
    
    fullfig = mpf.web_overlay_all(overlay_list)
    
    output_file(filename=figfile+"GaPSb.html", title="Bandgap phase diagram ")
    save(fullfig)
#------------------------------------------------------------------------------
