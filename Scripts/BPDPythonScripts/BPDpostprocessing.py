#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:20:00 2021

@author: bmondal
"""

#%%-------------------- Importing modules -------------------------------------
import numpy as np      
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib import animation
import BPDFunctions as mf
import BPDFittingFunctions as ff
import BPDPlotFunctions as mpf
import json
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as CM

#%%------------------- Load data from database --------------------------------
dirnamee = '/home/bmondal/MyFolder/VASP/'
filename_pre = 'GaPSb/DATA/'
filename = filename_pre+'/BPD_DataBase_mpi_2021-11-23-17:57:52_BW20.0_max.json'

figfile = dirnamee+'GaPSb/Figs/'

Setup_text = ['BW10.0_mean', 'BW10.0_max', 'BW20.0_mean', 'BW20.0_max']
#****************************************
#titletext = "Biaxial strain (001)"
titletext = ''# Setup_text[3]
#****************************************

filepath = dirnamee + filename
with open(filepath,'r') as f:
    data = json.load(f)

data.pop("94.9")
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
yy, xx = ff.Polyfit(YY, XX, xmin=-5, xmax=5)
#yy, xx = ff.SVRmodelFit(YY, XX, kernel='rbf', C=10, gamma="auto", epsilon=0.1)
#yy, xx = ff.SVRmodelFit(YY, XX)

#-----------
DIT2err = list(mf.lookupkey(BPD, 'DIT2err')) # From redefinition bands.
DIT1err = list(mf.lookupkey(BPD, 'DIT1err'))
assert len(NC) == len(DIT1err) == len(DIT2err) == len(EqmLP), 'Mismatch among length of conc., DITerr and Eqm. lattice parameter arrays.'
rearrange_NC_er,rearrange_DIT_er  = mf.RearrangeDIT(NC, DIT1err)
rearrange_NC_re_er,rearrange_DIT_re_er  = mf.RearrangeDIT(NC, DIT2err)

mf.PrintDITs(NC,DIT1err,DIT2err, PrintDIT = False)

pos_ = np.argsort(rearrange_DIT_re_er)
YYer = rearrange_DIT_re_er[pos_]
XXer = rearrange_NC_re_er[pos_]
yyer, xxer = ff.Polyfit(YYer, XXer, xmin=-5, xmax=5)

ErrorRegion = (yy,xx,xxer)

#%%-------------- substrate strain calculations -------------------------------       
subname = ['GaAs','GaP','Si','GaSb','InP']
subpos = ((20,2),(20,-1.8),(20,-2.9),(86,2.),(85,-1.2))
sublattice = np.array([[5.689],[5.475],[5.42103],[6.13383],[5.93926]]) # Substrate lattice parameters
subEg = [1.4662, 2.3645, 1.1915, 0.641,1.4341]
myco = ['lightsteelblue','red','cornflowerblue','royalblue','cyan']
myline = ['dotted','dashed','dashdot',':','-.']
 
#%%
compression_strain, compression_Eg = mf.ReadBinaryDatas("/home/bmondal/MyFolder/VASP/GaP/DATA/GaPinplanecompression.dat", 5.5)
expansion_strain, expansion_Eg = mf.ReadBinaryDatas("/home/bmondal/MyFolder/VASP/GaP/DATA/GaPinplaneexpansion.dat", 5.5)
GaP_biaxial_strain = np.concatenate((compression_strain,expansion_strain))
GaP_biaxial_Eg = np.concatenate((compression_Eg,expansion_Eg)) 
#%%
compression_strain, compression_Eg = mf.ReadBinaryDatas("/home/bmondal/MyFolder/VASP/GaSb/DATA/GaSbinplanecompression.dat", 5.5)
expansion_strain, expansion_Eg = mf.ReadBinaryDatas("/home/bmondal/MyFolder/VASP/GaSb/DATA/GaSbinplaneexpansion.dat", 5.5)
GaSb_biaxial_strain = np.concatenate((compression_strain,expansion_strain))
GaSb_biaxial_Eg = np.concatenate((compression_Eg,expansion_Eg)) 
#%%
xlabeltext = "Strain (%)"
ylabeltext = "E$_{\mathrm{g}}$ (eV)"
Both=False

NC_update = ['0.0']+list(NC)+['100']
Xarray = np.array(NC_update, dtype=float)
strainarray = [GaP_biaxial_strain]+STRAIN+[GaSb_biaxial_strain]
Bandgaparray = [GaP_biaxial_Eg]+BANDGAP_type2+[GaSb_biaxial_Eg]
markers = ["o","v","s","^","p","<","P",">","x","1","d","2","+","3","X","4","8","*","h","H","D"]*2
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'] * 4
fig, ax = mpf.plotEgStrain(strainarray,None, Bandgaparray, Xarray, marker=markers, color=colors,
                           ylabel=ylabeltext, xlabel=xlabeltext, Both=False, title=titletext)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(figfile+'/'+"figureS6g_1.png", bbox_inches="tight",dpi=300)
fig_leg = plt.figure(figsize=(8,2))
ax_leg = fig_leg.add_subplot(111)
ax_leg.legend(*ax.get_legend_handles_labels(), loc='center',ncol=8, handlelength=1)
ax_leg.axis('off')
plt.savefig(figfile+'/'+"figureS6_legends.png", bbox_inches="tight",dpi=300)

Xarray_=[]
for ii in range(0,len(Xarray)):
    xxaxis = [Xarray[ii]]*len(strainarray[ii])
    Xarray_+= xxaxis

Xarray = np.array(Xarray_, dtype=float)
strainarray = np.concatenate(strainarray, axis=0)
Bandgaparray = np.concatenate(Bandgaparray, axis=0)

#%%
NC_ = np.array(NC_update, dtype=float)
EqmLP_tmp = np.concatenate([[5.475], EqmLP, [6.13383]]) 
substrain = (sublattice[:-1] - EqmLP_tmp)/EqmLP_tmp*100
Xbound=(0,100);Ybound=(-6,6);gridextent=(0,100,-5,5)
#%%----------------- Image processing -----------------------------------------
#%% ---------- iterative interpolation ----------------------------------------
(X_b, Y_b, Z_b), (offset, scale), (fgridx, fgridy), \
    (points_iterative, points_iterative_Z_val), (AFTER_ITER_P, AFTER_ITER_P_Z, AFTER_ITER_P_Z_reshape),SHAPE = \
    ff.ImageProcessing(Xarray, strainarray, Bandgaparray,gridextent,Xbound=Xbound,Ybound=Ybound,gridresolutionX=10,gridresolutionY=10,
                        cutoff=None,method='griddata',submethod='cubic',smoothing=0.1,fillval=np.nan,FillwithNearestNeighbor=True,
                        IterativeInterpolation=0)
#%%% -------------------------------- Plotting --------------------------------
fig, ax = mpf.PlotScatterData(Xarray, strainarray, Bandgaparray,titletxt=None,xlabeltxt='Sb (%)',
                              ylabeltxt='Strain (%)',zlabeltxt='E$_{\mathrm{g}}$ (eV)',ZLIM=(0,2.5))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.tick_params(axis='both', which='major', labelsize=20)

plt.savefig(figfile+'/'+"figureS6g_2.png", bbox_inches="tight",dpi=300)

# points_data, points_grid = np.c_[X_b.ravel(), Y_b.ravel()] , np.c_[fgridx.ravel(), fgridy.ravel()] 
# fig_grid, ax_grid = mpf.PlotScatterData(points_grid[:,0],points_grid[:,1],'k',titletxt='iteration cover',xlabeltxt='Bi (%)',
#                                         ylabeltxt='Strain (%)',ShowColorBar=False,marker='.',ZLIM=(0,2.5))
# mpf.PlotScatterData(points_data[:,0],points_data[:,1],Z_,fig=fig_grid,ax=ax_grid,ShowColorBar=False,marker='.',ZLIM=(0,2.5))
# for POINTS in points_iterative:
#     mpf.PlotScatterData(POINTS[:,0],POINTS[:,1],None,fig=fig_grid,ax=ax_grid,ShowColorBar=False,marker='x')    
# prop_cycle = plt.rcParams['axes.prop_cycle']

# colorstmp = prop_cycle.by_key()['color']
# cmap = LinearSegmentedColormap.from_list('tmp_list', colorstmp, N=len(colorstmp))
# cbar = plt.colorbar(mappable=CM.ScalarMappable(cmap=cmap), ax=ax_grid, ticks=[])
# #### Movie for iterative covering
# outer_ani =  mpf.ShowbatchInteractive(points_data, points_grid, points_iterative,save_movie=False, cumulative=True, 
#                                       save_movie_path=None, xLimit=None, yLimit=None, xlabel='Bi (%)', ylabel='Strain (%)')

#%%
myco, fig2, ax2, im = mpf.PlotBPD(NC_, substrain, subname, subpos, subEg, myco, myline, XX, YY,
                                  AFTER_ITER_P_Z_reshape.T, gridextent, Concfitdata=xx, DITfitdata=yy, DITfit=True, \
                                  Concfitdataerr=xxer, DITfitdataerr=yyer, DITfitErr=1, ErrLabel='20%',  \
                                      DrawSub=True, drawheatmap=True, xlabel='Sb (%)',\
                                          ylabel='Strain(%)', fignum=None, title='',\
                                              DrawExp=False, ExpData=None, DrawFillBtw=True, filldata=ErrorRegion,\
                                                  Textpos=((55,0.25),(13.5,0.25)),mytext=['DIRECT','INDIRECT'],
                                                  vmin=0.0,vmax=2.5,label_rot=-50)

ax2.xaxis.set_major_locator(MultipleLocator(20))
ax2.yaxis.set_major_locator(MultipleLocator(2))
ax2.set_xlim(gridextent[:2])
# # plt.savefig(figfile+'/GaAsBi-BPD-Biaxialstrain.eps', format='eps',bbox_inches = 'tight',dpi=300)   
# plt.savefig(figfile+'/GaAsBi-BPD-Biaxialstrain.png', format='png',bbox_inches = 'tight',dpi=300) 
plt.savefig(figfile+'/figure6a.png', format='png',bbox_inches = 'tight',dpi=300)   
# plt.savefig('/home/bmondal/Dropbox/III-V_ternary/imgs/figure6a.png',bbox_inches = 'tight',dpi=300)

myco, fig2, ax2, im = mpf.PlotBPD(NC_, substrain, subname, subpos, subEg, myco, myline, XX, YY,
                                  (AFTER_ITER_P,AFTER_ITER_P_Z), gridextent, Concfitdata=xx, DITfitdata=yy, DITfit=False, \
                                  Concfitdataerr=xxer, DITfitdataerr=yyer, DITfitErr=0, ErrLabel='20%',  \
                                      DrawSub=True, drawheatmap=False,DrawScatterPlot=True, xlabel='Sb (%)',\
                                          ylabel='Strain(%)', fignum=None, title='scatter plot',\
                                              DrawExp=False, ExpData=None, DrawFillBtw=False, filldata=ErrorRegion,\
                                                  Textpos=((55,0.25),(13.5,0.25)),mytext=['DIRECT','INDIRECT'],
                                                  vmin=0.0,vmax=2.5,label_rot=-50)

ax2.xaxis.set_major_locator(MultipleLocator(10))
ax2.yaxis.set_major_locator(MultipleLocator(2))
ax2.set_xlim(gridextent[:2])

#%% ------------- Single shot interpolation -----------------------------------
# filtered_arr, extent = ff.CreateHeatmapData(Xarray, strainarray, Bandgaparray,preprocessing=0,smoothing=0.05,
#                                             heatmap_bound=1,G_filtering=False,
#                                             InterpolationMethod='griddata',
#                                             intermethod='linear',Sigma=1,kernel=None,
#                                             Conc_bound=(0,100),StrainBound=(-5,5),order=1)
# myco, fig2, ax2, im = mpf.PlotBPD(NC_, substrain, subname, subpos, subEg, myco, myline, XX, YY,
#                                   filtered_arr, extent, Concfitdata=xx, DITfitdata=yy, DITfit=1, \
#                                   Concfitdataerr=xxer, DITfitdataerr=yyer, DITfitErr=1, ErrLabel='20%',  \
#                                       DrawSub=True, drawheatmap=True, xlabel='Sb (%)',\
#                                           ylabel='Strain(%)', fignum=None, title=titletext,\
#                                               DrawExp=False, ExpData=None, DrawFillBtw=1, filldata=ErrorRegion,\
#                                                   Textpos=((55,0.25),(13.5,0.25)),mytext=['DIRECT','INDIRECT'],
#                                                   vmin=0.0,vmax=2.5,label_rot=-50)
    
# ax2.xaxis.set_major_locator(MultipleLocator(20))
# ax2.yaxis.set_major_locator(MultipleLocator(2))

# plt.tight_layout()
# plt.savefig(figfile+'/GaAsSb-BPD-Biaxialstrain.eps', format='eps',dpi=300) 
# plt.savefig(figfile+'/GaAsSb-BPD-Biaxialstrain.png', format='png',bbox_inches = 'tight',dpi=300)     
# plt.savefig('/home/bmondal/Dropbox/III-V_ternary/imgs/figure6a.png',bbox_inches = 'tight',dpi=300)

#%%--------------- HTML file generation --------------------------------------- 
CreateHTML = 1; substrate=True; strain0line=True
if CreateHTML:
    myline = ['dotted','dashed','dashdot','dashed','dotdash']
    #p.toolbar.logo = None
    from bokeh.plotting import output_file, save
    import BPDwebPlotFunctions as wpf
    filtered_arr = (AFTER_ITER_P_Z_reshape.T).copy()
    extent = gridextent
    tooltips = [("Sb", "$x %"), ("Strain", "$y %"), ("Bandgap", "@image eV")]
    filtered_arr = filtered_arr[::-1] # To swap the 'origin' of image. hv.Image has default 'origin'='up-left'
    inter_img = wpf.web_heatmap(filtered_arr, extent, tooltips, clabel="Eg") 
    
    DITs = wpf.web_dit_scatter(XX,YY,xlabel="Sb concentration (%)",ylabel="Strain (%)"\
                           ,title="GaPSb biaxial strain",fontsize=20)
    # exps = wpf.web_exp_scatter(ExpData[0],ExpData[1],label='Expriments(Appl.Phys.Lett. 52, 549 (1988))',fontsize=20)
    curve = wpf.web_dit_fit(xx, yy, label='Polyfit', fontsize=15) 
    curve2 = wpf.web_diterr_fit(xxer, yyer, label='20%', fontsize=15) 
    text1 = wpf.web_bpd_text(15,1, text='Indirect') 
    text2 = wpf.web_bpd_text(75,1, text='Direct')
    if strain0line: Eqmline = wpf.web_eqm_line(yeqm=0)

    overlay_list = [inter_img, DITs, curve, curve2, text1, text2, Eqmline]
    if strain0line: 
        Eqmline = wpf.web_eqm_line(yeqm=0)
        overlay_list.append(Eqmline)
    if substrate:
        suboverlay = wpf.web_plot_substrate(NC_, substrain, subname, myco, myline, extent)
        overlay_list.append(suboverlay)
    # overlay_list.append(exps)
    fullfig = wpf.web_overlay_all(overlay_list)
    
    output_file(filename=figfile+"GaPSb.html", title="Bandgap phase diagram ")
    save(fullfig)
#------------------------------------------------------------------------------