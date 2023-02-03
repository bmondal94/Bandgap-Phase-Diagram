#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 14:47:59 2021

@author: bmondal
"""
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import itertools
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from scipy.ndimage.filters import gaussian_filter

#------------------------------------------------------------------------------
params = {'legend.fontsize': 18,
          'figure.figsize': (8, 6),
         'axes.labelsize': 24,
         'axes.titlesize': 24,
         'xtick.labelsize':24,
         'ytick.labelsize': 24,
         'errorbar.capsize':2}
plt.rcParams.update(params)

#------------------------------------------------------------------------------
marker = itertools.cycle((',', '+', '.', 'o', '*','v','d','<','>','1','2','3','4',\
                          '8','s','p','P','h','H','X','x','D','^'))

#------------------------------------------------------------------------------
def DefineMyFigure(ylabel, xlabel, title, figsize=(8,6), fignum=None):
    """
    This function defines the skeliton for the Eg-Strain, Bandgap phase diagram etc. figure.

    Parameters
    ----------
    ylabel : String
        Y-axis label.
    xlabel : String
        X-axis label.
    title : String
        Title of the figure.
    figsize : Integer tuple
        Size of the figure (width, height).
    fignum : int or str or Figure, optional
        A unique identifier for the figure.
        If a figure with that identifier already exists, this figure is made active and returned. 
        An integer refers to the Figure.number attribute, a string refers to the figure label.
        If there is no figure with the identifier or num is not given, a new figure is created, 
        made active and returned. If num is an int, it will be used for the Figure.number attribute, 
        otherwise, an auto-generated integer value is used (starting at 1 and incremented for 
        each new figure). If num is a string, the figure label and the window title is set to this value.
    Returns
    -------
    fig : Pyplot Figure
    ax : Pyplot Figure axes

    """
    fig, ax = plt.subplots(figsize=figsize, num=fignum)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    #ax.xaxis.set_ticks(strain)
     
    
    return fig, ax

def plotEgStrain(strain, bandgap_type1, bandgap_type2, mylabels,Type1=False, \
                 Both=False, ylabel=None, xlabel=None, title=None, marker=None, color=None):
    """
    This function plots the strain vs bandgap with only single configuration for each 
    concentration and strain point.

    Parameters
    ----------
    strain : Signed float
        Strain array for each confuration.
    bandgap_type1 : Float
        Bandgap array considering usual CBM and VBM.
    bandgap_type2 : Float
        Bandgap array considering 'Redefined' CBM and VBM.
    mylabels : String
        Array of concentration will be used for labeling.
    Type1 : Bool, optional
        To plot bandgap from CBM-VBM. The default is False.
    Both : Bool, optional
        To plot bandgap from CBM-VBM (==Type1) and bandgap from redefined CBM-VBM (==Type2). 
        If Type1 and Both are both false then Type2=True.
        The default is False.
    ylabel : String, optional
        Y-axis label in figure. The default is None.
    xlabel : String, optional
        X-axis label in figure. The default is None.
    title : String, optional
        Title of the figure. The default is None.

    Returns
    -------
    fig : Pyplot Figure
    ax : Pyplot Figure axes

    """
    if marker is None: marker = ["o","v","s","^","p","<","P",">","x","1","d","2","+","3","X","4","8","*","h","H"]*10
    if color is None: color = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'] * 10
    
    Type2=True if not Type1 and not Both else False
    fig, ax = DefineMyFigure(ylabel,xlabel,title)
    ax.axvline(color='k', ls='--')
    YY = bandgap_type2 if Type2 else bandgap_type1
    for J in range(len(mylabels)):
        #print(strain[J], YY[J], mylabels[J])
        p = ax.plot(strain[J], YY[J],'-',marker = marker[J],color=color[J],label=f'{mylabels[J]:.1f}')
        if Both: 
            ax.plot(strain[J], bandgap_type2[J], color=p[0].get_color(), ls='--',lw=1)
    if Both: ax.plot([], [], color='k', ls='--',lw=1,label='Redefined')
    # ax.legend(loc=9, ncol=4, handlelength=1, bbox_to_anchor=(0.5, 1.3))
    # ax.set_ylim(ymax=2)
    # plt.tight_layout()
    return fig, ax
    
def plotEgStrainErrorBar(strain, bandgap1, bandgap2, std1, std2, mylabels,\
                         Type1=False, Both=False, ylabel=None, xlabel=None, title=None):
    """
    This function plots the strain vs bandgap with multiple configuration for each 
    concentration and strain point.

    Parameters
    ----------
    strain : TYPE
        DESCRIPTION.
    bandgap1 : Float
        Bandgap array considering usual CBM and VBM.
    bandgap2 : Float
        Bandgap array considering 'Redefined' CBM and VBM.
    std1 : Float
        Standared deviation in bandgap of multiple configuration, array considering usual CBM and VBM.
    std2 : Float
        Standared deviation in bandgap of multiple configuration, array considering 'Redefined' CBM and VBM.
    mylabels : String
        Array of concentration will be used for labeling.
    Type1 : Bool, optional
        To plot bandgap from CBM-VBM. The default is False.
    Both : Bool, optional
        To plot bandgap from CBM-VBM (==Type1) and bandgap from redefined CBM-VBM (==Type2). 
        If Type1 and Both are both false then Type2=True.
        The default is False.
    ylabel : String, optional
        Y-axis label in figure. The default is None.
    xlabel : String, optional
        X-axis label in figure. The default is None.
    title : String, optional
        Title of the figure. The default is None.

    Returns
    -------
    fig : Pyplot Figure
    ax : Pyplot Figure axes

    """
    Type2=True if not Type1 and not Both else False
    fig, ax = DefineMyFigure(ylabel,xlabel,title)
    YY = bandgap2 if Type2 else bandgap1
    for J in range(len(mylabels)):
        p = ax.errorbar(strain[J], YY[J], yerror=std2[J], label=mylabels[J])
        if Both:
            ax.errorbar(strain[J], bandgap2[J], yerror=std2[J], color=p[0].get_color(), ls='--',lw=1)
    if Both: ax.plot([], [], color='k', ls='--',lw=1,label='Redefined')
    ax.legend(ncol=3)      
    return fig, ax

def DefineFigureBPD(ylabel=None, xlabel=None, title=None, DITText=False, pos=((70,0),(10,0)),
                    mytext=['DIRECT','INDIRECT'],figsize=(8,6), fignum=None):
    """
    This function defines the skeliton of Bandgap phase diagram. It calles DefineMyFigure().

    Parameters
    ----------
    ylabel : String, optional
        Y-axis label in figure. The default is None.
    xlabel : String, optional
        X-axis label in figure. The default is None.
    title : String, optional
        Title of the figure. The default is None.
    figsize : Integer tuple
        Size of the figure (width, height).
        
    Returns
    -------
    fig : Pyplot Figure
    ax : Pyplot Figure axes

    """
    fig, ax = DefineMyFigure(ylabel,xlabel,title,figsize=figsize, fignum=fignum)
    ax.axhline(color='k', ls='--')  
    # ax1.axvline(x=38.5,ymax=0.55,color='k', ls='--') 
    # ax1.text(40,-4,'38.5% P',size=24,c='g')
    # ax1.text(15,-1.2,'DIRECT E$_{\mathrm{g}}$',rotation='vertical',size=24,c='b')
    # ax1.text(55,-1.6,'INDIRECT E$_{\mathrm{g}}$',rotation=270,size=24,c='b')
    if DITText:
        for i, I in enumerate(pos):
            ax.text(I[0],I[1],mytext[i],rotation='horizontal',rotation_mode='anchor',size=19,c='k',\
                    horizontalalignment='center', verticalalignment='center')
    #ax.text(70,-1.5,'INDIRECT',rotation=270,size=24,c='b')
    #ax1.axvline(x=30,ymin=0.2,ymax=0.7,color='k', ls='-.',lw=3) 
    #ax1.annotate("", xy=(25,-2), xytext=(25, -3.5),arrowprops=dict(arrowstyle="<->"))

    # ax.set_xticks([0,100])
    # ax.set_xticklabels(['As','P'])
    ax.set_ylim(-5,5)
    # ax.set_xlim(0,100)
    return fig, ax

def PlotDIT_points(ax, Conc, DIT, pointcolor='m'):
    """
    This function plots the direct-indirect transition points as the scatter plot.

    Parameters
    ----------
    ax : Figure axes
        Figure axes.
    Conc : Float
        Concentration array.
    DIT : Float
        Direct-indirect transition points array.
    pointcolor: string or point color object matplotlib pyplot.
        Color of the DIT points.

    Returns
    -------
    None.

    """
    ax.scatter(Conc, DIT, edgecolors=pointcolor,marker='o',label="DIT",facecolors=pointcolor,s=20)
    return ax

def PlotDIT_Fit(ax, Conc, DIT, linecolor='m'):
    """
    This function plots fitted the direct-indirect transition points.

    Parameters
    ----------
    ax : Figure axes
        Figure axes.
    Conc : Float
        Concentration array.
    DIT : Float
        Direct-indirect transition points array.
    linecolor: string or line color object matplotlib pyplot.
        Color of the DIT line.

    Returns
    -------
    None.

    """
    
    ax.plot(Conc, DIT, ls='-', c=linecolor, label='DIT-fit')
    return

def PlotDITerr_Fit(ax, Conc, DIT, label, linecolor='g', linesty='-'):
    """
    This function plots fitted the direct-indirect transition points.

    Parameters
    ----------
    ax : Figure axes
        Figure axes.
    Conc : Float
        Concentration array.
    DIT : Float
        Direct-indirect transition points array.
    label: String
        The Label for the DIT error fit.
    linecolor: string or line color object matplotlib pyplot.
        Color of the DIT error fit line.
    linesty: line styles from matplotlib pyplot.
        Line style of the DIT error fit line.

    Returns
    -------
    None.

    """
    
    ax.plot(Conc, DIT, c=linecolor, ls=linesty ,label=label)
    return

def DrawSubstrate(ax, NC, substrain, subname, subpos, mycolor, myline, mycolorIgnore=False,\
                  mylineIgnore=False, IgnoreLabel=False, label_rot=0):
    """
    This function plots the substrate map on Bandgap phase diagram.

    Parameters
    ----------
    ax : Figure axes
        Figure axes.
    NC : Float
        Concentration array.
    substrain : Float
        Strain array from substrate consideration. It is calculated based on equilibrium
        lattice parameter for above concentration array and selected substrate.
    subname : String
        Name array of the substrates.
    subpos : tuple
        Position array of the substrate name texts that will be used if IgnoreLabel=True.
    mycolor : Matplotlib pyplot color
        Defines the color for each substrate curve.
    myline : Matplotlib pyplot line style
        Defines the linestyle for each substrate curve.
    mycolorIgnore: Bool
        Whether colors will be used for different substrates.
        Default if False. If True all substrate line will be of black color.
    mylineIgnore: Bool
        Whether different line style will be used for different substrates.
        Default if False. If True all substrate line will be of solid line.
    IgnoreLabel: Bool
        Whether to label the substrate names on the plot. Default is False.
        Default is to put the substrate name as legend. 
    label_rot: Float
        Rotating the label texts.

    Returns
    -------
    None

    """
    if mycolorIgnore: mycolor = ['k']*len(mycolor)
    if mylineIgnore: myline = ['-']*len(myline)
    NC_tmp = NC #np.insert(NC, 0, 0) # To include x=0
    for J in range(len(substrain)):    
        if IgnoreLabel:
            ax.plot(NC_tmp,substrain[J],lw=1.5,ls=myline[J], color=mycolor[J]) #, marker='.'
            pos = subpos[J]
            ax.text(pos[0],pos[1],subname[J],rotation=label_rot,rotation_mode='anchor',size=20,c='k',\
                    horizontalalignment='center', verticalalignment='center')
        else:
            ax.plot(NC_tmp,substrain[J],lw=1.5, ls=myline[J],label=subname[J], color=mycolor[J])
    return 

def PlotExperimentalPoints(ax, data):
    """
    This function plots the Experimental points.

    Parameters
    ----------
    ax : Figure axes
        Figure axes.
    data : Tuple
        The experimental data as tuple (x-coordinate array, y-coordinatearray).

    Returns
    -------
    None.

    """
    ax.scatter(data[0], data[1], edgecolor='k',marker='*',label='Experiment',facecolors='k',s=100)
    return 

def PlotFillbetweenX(data, ax):
    """
    This function fill the error region using fill_betweenx().

    Parameters
    ----------
    data : Float tuple
        The tuple containing the Y, X1, X2.
    ax : Figure axes
         Figure axes.

    Returns
    -------
    ax : Figure axes
        Figure axes.

    """
    yy,xx,xxer = data[0], data[1], data[2]
    h = ax.fill_betweenx(yy,xx,xxer, hatch='++', alpha=0)
    
    return h

def ApplyFilter(filtered_arr, sigma=4,mode='nearest'):
    filtered_arr_fil = gaussian_filter(filtered_arr, sigma=sigma,mode=mode)
    return filtered_arr_fil

def DrawHeatmap(ax, fig, filtered_arr, extent,aspect='auto', origin='lower', \
                mycmap=plt.cm.RdYlBu_r, InterpolMethod='bicubic',cbar_horizontal=False, \
                    vmin=None, vmax=None, applyfilter=False):
    """
    This function is used to draw the heatmap and colorbar.

    Parameters
    ----------
    ax : Axes object
        Axes object of the heatmap figure.
    fig : Figure object 
        Heatmap figure object.
    filtered_arr : Float array
        The heatmap data.
    extent : Float list
        The limits of x and y axis in heatmap, (left, right, bottom, top).
    aspect : String/float, optional
        Aspect ratio control in image show. The default is 'auto'.
    origin : String, optional
        Origin of the heatmap image show. The default is 'lower'.
    mycmap : String or colormap, optional
        The colormap instance or registered colormap name used to map scalar data to colors.
        This parameter is ignored for RGB(A) data.. The default is plt.cm.RdYlBu_r.
    InterpolMethod : String, optional
        The interpolation method in heatmap. The default is 'bilinear'.
    cbar_horizontal : Bool, optional
        Whether to put the color bar horizontally. The default is False (vertical).
    vmin: float
        colorbar lower limit.
    vmax: float
        colorbar higher limit.

    Returns
    -------
    Image object.

    """
    if applyfilter: filtered_arr = ApplyFilter(filtered_arr)
    im = ax.imshow(filtered_arr, aspect=aspect,extent=extent, origin=origin,\
                    cmap=mycmap,interpolation=InterpolMethod,vmin=vmin,vmax=vmax)
    if cbar_horizontal:
        cbar = fig.colorbar(im,format='%.1f', ax=[ax],location='top',fraction=0.05)
        cbar.ax.set_ylabel('E$_{\mathrm{g}}$(eV)', fontsize = 24,\
                           rotation=0,labelpad=50,va='center_baseline')
    else:
        cbar = fig.colorbar(im,format='%.1f') #v = np.linspace(-.1, 2.0, 15), plt.colorbar(ticks=v)
        cbar.ax.set_ylabel('E$_{\mathrm{g}}$ (eV)', fontsize = 24) #, weight="bold")
    return im

def DrawScatter(ax, fig, filtered_arr, extent, mycmap=plt.cm.RdYlBu_r, cbar_horizontal=False,vmin=None,vmax=None):
    """
    This function is used to draw the strain points as scatter plot and colorbar.

    Parameters
    ----------
    ax : Axes object
        Axes object of the heatmap figure.
    fig : Figure object 
        Heatmap figure object.
    filtered_arr : Float array
        The data points and bandgap array.
    mycmap : String or colormap, optional
        The colormap instance or registered colormap name used to map scalar data to colors.
        This parameter is ignored for RGB(A) data.. The default is plt.cm.RdYlBu_r.
    cbar_horizontal : Bool, optional
        Whether to put the color bar horizontally. The default is False (vertical).

    Returns
    -------
    Image object.

    """

    im = ax.scatter(filtered_arr[0][:,0], filtered_arr[0][:,1],cmap=mycmap, c=filtered_arr[1],marker='s',s=20,vmin=vmin,vmax=vmax)
    ax.set_xlim(extent[0],extent[1])
    ax.set_ylim(extent[2],extent[3])
    ax.grid(visible=False)
    if cbar_horizontal:
        cbar = fig.colorbar(im,format='%.1f', ax=[ax],location='top',fraction=0.05)
        cbar.ax.set_ylabel('E$_{\mathrm{g}}$(eV)', fontsize = 18, weight="bold",\
                           rotation=0,labelpad=50,va='center_baseline')
    else:
        cbar = fig.colorbar(im,format='%.1f') #v = np.linspace(-.1, 2.0, 15), plt.colorbar(ticks=v)
        cbar.ax.set_ylabel('E$_{\mathrm{g}}$ (eV)', fontsize = 18, weight="bold")
    return im


def PlotBPD(NC, substrain, subname, subpos, subeg, mycolor, myline, Conc, DIT, filtered_arr, extent,  \
             Concfitdata=None, DITfitdata=None, DITfit=True, Concfitdataerr=None, DITfitdataerr=None, DITfitErr=True,\
             ErrLabel='', DrawSub=True, DrawDIT=True,label_rot=0, drawheatmap=True, DrawScatterPlot=False,\
                  ylabel=None, xlabel=None, title=None, figsize=(8,6), fignum=None, cmap=plt.cm.RdYlBu_r,\
                      DrawExp=False, ExpData=None, DrawFillBtw=False, filldata=None,applyfilter=False,\
                          DITText=True,Textpos=((70,0),(10,0)),mytext=['DIRECT','INDIRECT'],vmin=None, vmax=None):
    fig, ax = DefineFigureBPD(ylabel=ylabel, xlabel=xlabel, title=title, \
                              mytext=mytext, pos=Textpos,
                              DITText=DITText, figsize=figsize,fignum=fignum)
    if drawheatmap: 
        im = DrawHeatmap(ax, fig, filtered_arr, extent,vmin=vmin,vmax=vmax, mycmap=cmap,applyfilter=applyfilter)
        mycolor = [im.cmap(im.norm(x)) for x in subeg]
    elif DrawScatterPlot:
        im = DrawScatter(ax, fig, filtered_arr,extent,vmin=vmin,vmax=vmax, mycmap=cmap)
        mycolor = [im.cmap(im.norm(x)) for x in subeg]
        
    if DrawSub: 
        DrawSubstrate(ax, NC, substrain, subname, subpos, mycolor, myline, \
                      mycolorIgnore=True, mylineIgnore=True, IgnoreLabel=True,label_rot=label_rot,\
                          )
    if DrawDIT: PlotDIT_points(ax, Conc, DIT, pointcolor='k')
    if DITfit: PlotDIT_Fit(ax, Concfitdata, DITfitdata, linecolor='k')
    if DITfitErr: PlotDITerr_Fit(ax, Concfitdataerr, DITfitdataerr, ErrLabel, linesty='-.', linecolor='k')
    if DrawExp: PlotExperimentalPoints(ax, ExpData)
    handles, labels = ax.get_legend_handles_labels()
    if DrawFillBtw: 
        h = PlotFillbetweenX(filldata, ax)
        handles.append(h)
        labels.append("DIT-region")
        
    #ax.legend(ncol=3, handles=handles, labels=labels, columnspacing=0.5, labelspacing=0.2, handlelength=1, handletextpad=0.3)
    # ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0,-.55,1,0.35), loc="upper left",
    #             mode="expand", borderaxespad=0, ncol=3, columnspacing=0.8, labelspacing=0.2, handlelength=1, handletextpad=0.3)

    return mycolor, fig, ax, im
#%%
import matplotlib.cm as CM
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance_matrix, KDTree
import plotly.express as px
from plotly.offline import plot


#### Plot scatter plot
def PlotScatterData(X,Y,Z,ax=None,fig=None,titletxt=None,xlabeltxt=None,ylabeltxt=None,zlabeltxt=None,
                    ZLIM=None,XLIM=None,YLIM=None,ShowColorBar=True,marker='o',cmap=plt.cm.RdYlBu_r): #,markersize=50):    
    if ShowColorBar:
        if ZLIM is None: ZLIM = (min(Z)-0.1,max(Z)+0.1)
    else:
        ZLIM = (None, None)

    if XLIM is None:
        XLIM_SHIFT = (max(X)-min(X))/100 # 1% of the scale 
        XLIM = (min(X)-XLIM_SHIFT,max(X)+XLIM_SHIFT)
    if YLIM is None:
        YLIM_SHIFT = (max(Y)-min(Y))/100 # 1% of the scale
        YLIM = (min(Y)-YLIM_SHIFT,max(Y)+YLIM_SHIFT)
    
    if ax is None: 
        fig, ax = plt.subplots(figsize=(8,6))
        ax.set_xlabel(xlabeltxt)
        ax.set_ylabel(ylabeltxt)
        ax.set_title(titletxt)
        ax.set_xlim(XLIM[0],XLIM[1])
        ax.set_ylim(YLIM[0],YLIM[1])
    if fig is not None: fig=fig
    if Z is None or isinstance(Z, str): cmap=None
    im = ax.scatter(X, Y, c=Z, cmap=cmap, vmin=ZLIM[0], vmax=ZLIM[1], marker=marker) #, markersize=markersize)    
    if ShowColorBar:
        cbar = fig.colorbar(im,format='%.1f')
        cbar.ax.set_ylabel(zlabeltxt)
    plt.tight_layout()
    return fig, ax

#### Plot scatter plot
def PlotPlotlyScatterData(X,Y,Z,ax=None,fig=None,titletxt=None,xlabeltxt=None,ylabeltxt=None,zlabeltxt=None,
                          ZLIM=None,XLIM=None,YLIM=None,ShowColorBar=True,marker='o',savefig=True,figpath=None,
                          figname='PlotlyScatterPlot.html',cmap='viridis'):    
    if ShowColorBar:
        if ZLIM is None: ZLIM = (min(Z)-0.1,max(Z)+0.1)
    else:
        ZLIM = (None, None)

    if XLIM is None:
        XLIM_SHIFT = (max(X)-min(X))/100 # 1% of the scale 
        XLIM = (min(X)-XLIM_SHIFT,max(X)+XLIM_SHIFT)
    if YLIM is None:
        YLIM_SHIFT = (max(Y)-min(Y))/100 # 1% of the scale
        YLIM = (min(Y)-YLIM_SHIFT,max(Y)+YLIM_SHIFT)
        
    fig = px.scatter(x=X,y=Y,color=Z,labels=dict(x=xlabeltxt,y=ylabeltxt,color=zlabeltxt),color_continuous_scale=cmap,range_color=ZLIM)
    fig.update_xaxes(range=XLIM,title_font = {"size": 24}, tickfont={"size": 24})
    fig.update_yaxes(range=YLIM,title_font = {"size": 24},zeroline=True, tickfont={"size": 24},
                     zerolinewidth=2, zerolinecolor='black')
    fig.update_traces(marker=dict(size=20),selector=dict(mode='markers'))
    fig.update_coloraxes(colorbar_tickfont={"size": 24},colorbar_title_text="Eg (eV)")
    #fig.show()
    if savefig and figpath:
        fig.write_html(figpath+'/'+figname, auto_open=False)
    else:
        plot(fig)
    return fig

#### Plot tri angular plot
def PlotTripColor(X,Y,Z,ax=None,fig=None,titletxt=None,xlabeltxt=None,ylabeltxt=None,zlabeltxt=None,
                  ZLIM=None,XLIM=None,YLIM=None,ShowColorBar=True):    
    if ShowColorBar:
        if ZLIM is None: ZLIM = (min(Z)-0.1,max(Z)+0.1)
    else:
        ZLIM = (None, None)

    if XLIM is None: XLIM = (min(X),max(X))
    if YLIM is None: YLIM = (min(Y),max(Y))
    
    if ax is None: 
        fig, ax = plt.subplots(figsize=(8,6))
        ax.set_xlabel(xlabeltxt)
        ax.set_ylabel(ylabeltxt)
        ax.set_title(titletxt)
        ax.set_xlim(XLIM[0],XLIM[1])
        ax.set_ylim(YLIM[0],YLIM[1])
    if fig is not None: fig=fig
    im = ax.tripcolor(X,Y, Z, cmap=plt.cm.RdYlBu_r, vmin=ZLIM[0], vmax=ZLIM[1])   
    if ShowColorBar:
        cbar = fig.colorbar(im,format='%.1f')
        cbar.ax.set_ylabel(zlabeltxt)
    plt.tight_layout()
    return fig, ax

def ShowbatchInteractive(points1, points2, data, save_movie=False, cumulative=True, 
                         save_movie_path=None, xLimit=None, yLimit=None, xlabel=None, ylabel=None):

    fig, ax = plt.subplots()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xLimit)
    plt.ylim(yLimit)
    sc = ax.scatter(points1[:,0],points1[:,1],marker='o')
    ax.scatter(points2[:,0],points2[:,1],marker='.',c='k')
    
    def init():
        ax.set_title('Start')
        return sc

    def animate(i, cumulative=False):
        # print(i)
        if i == 0 and cumulative:       
            plt.cla()
            plt.xlim(xLimit)
            plt.ylim(yLimit)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            
        batch_samples = data[i]
        x = batch_samples[:,0]
        y = batch_samples[:,1]
        if cumulative:
            ax.scatter(x,y,marker='x')
        else:
            sc.set_offsets(np.c_[x,y])
        
        ax.set_title(f'Loop = {i}')
        return sc
        
    ani = FuncAnimation(fig, animate, init_func=init, frames=len(data), fargs=(cumulative,), interval=800, repeat_delay=100, repeat=True) 
    if save_movie:
        if save_movie_path:
            ani.save(save_movie_path)
            # plt.close()
        else:
            print('Please provide the save_movie_path.')
    else:
        plt.show()
    return ani
#------------------------------------------------------------------------------
#%%    
if  __name__ == '__main__':
    import numpy as np
    BANDGAP_type1= BANDGAP_type2= [np.linspace(1,10,10),np.linspace(1,10,10)]
    STRAIN = [np.linspace(1,10,10),np.linspace(2,20,10)]
    NC = ['hi','hii']
    
    plotEgStrain(STRAIN, BANDGAP_type1, BANDGAP_type2,NC)
