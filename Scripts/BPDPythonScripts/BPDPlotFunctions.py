#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 14:47:59 2021

@author: bmondal
"""
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import itertools
import holoviews as hv 
from holoviews import opts
from holoviews.operation.datashader import regrid
from bokeh.models.tools import *
hv.extension('bokeh')

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
                 Both=False, ylabel=None, xlabel=None, title=None):
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
    Type2=True if not Type1 and not Both else False
    fig, ax = DefineMyFigure(ylabel,xlabel,title)
    ax.axvline(color='k', ls='--')
    YY = bandgap_type2 if Type2 else bandgap_type1
    for J in range(len(mylabels)):
        #print(strain[J], YY[J], mylabels[J])
        p = ax.plot(strain[J], YY[J],'-',marker = next(marker),label=mylabels[J])
        if Both: 
            ax.plot(strain[J], bandgap_type2[J], color=p[0].get_color(), ls='--',lw=1)
    if Both: ax.plot([], [], color='k', ls='--',lw=1,label='Redefined')
    ax.legend(ncol=3)
   
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

def DefineFigureBPD(ylabel=None, xlabel=None, title=None, figsize=(8,6), fignum=None):
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
    #ax.text(10,-1.2,'DIRECT',rotation='vertical',size=24,c='b')
    #ax.text(70,-1.5,'INDIRECT',rotation=270,size=24,c='b')
    #ax1.axvline(x=30,ymin=0.2,ymax=0.7,color='k', ls='-.',lw=3) 
    #ax1.annotate("", xy=(25,-2), xytext=(25, -3.5),arrowprops=dict(arrowstyle="<->"))

    # ax.set_xticks([0,100])
    # ax.set_xticklabels(['As','P'])
    ax.set_ylim(-5,5)
    ax.set_xlim(0,100)
    return fig, ax

def PlotDIT_points(ax, Conc, DIT):
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

    Returns
    -------
    None.

    """
    ax.scatter(Conc, DIT, c='m',marker='o',label="DIT")
    return ax

def PlotDIT_Fit(ax, Conc, DIT):
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

    Returns
    -------
    None.

    """
    
    ax.plot(Conc, DIT, 'm-',label='DIT-fit')
    return

def PlotDITerr_Fit(ax, Conc, DIT, label):
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

    Returns
    -------
    None.

    """
    
    ax.plot(Conc, DIT, 'g-',label=label)
    return

def DrawSubstrate(ax, NC, substrain, subname, mycolor, myline):
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
    mycolor : Matplotlib pyplot color
        Defines the color for each substrate curve.
    myline : Matplotlib pyplot line style
        Defines the linestyle for each substrate curve.

    Returns
    -------
    None

    """
    for J in range(len(substrain)):    
        ax.plot(NC,substrain[J],lw=3, ls=myline[J],label=subname[J], color=mycolor[J])
    return 

def DrawHeatmap(ax, fig, filtered_arr, extent,aspect='auto', origin='lower', \
                mycmap=plt.cm.RdYlBu_r, InterpolMethod='bilinear',cbar_horizontal=False):
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

    Returns
    -------
    None.

    """
    im = ax.imshow(filtered_arr, aspect=aspect,extent=extent, origin=origin,\
                    cmap=mycmap,interpolation=InterpolMethod)
    if cbar_horizontal:
        cbar = fig.colorbar(im,format='%.1f', ax=[ax],location='top',fraction=0.05)
        cbar.ax.set_ylabel('E$_{\mathrm{g}}$(eV)', fontsize = 18, weight="bold",\
                           rotation=0,labelpad=50,va='center_baseline')
    else:
        cbar = fig.colorbar(im,format='%.1f') #v = np.linspace(-.1, 2.0, 15), plt.colorbar(ticks=v)
        cbar.ax.set_ylabel('E$_{\mathrm{g}}$ (eV)', fontsize = 18, weight="bold")
    return im

def PlotBPD(NC, substrain, subname, subeg, mycolor, myline, Conc, DIT, filtered_arr, extent,  \
             Concfitdata=None, DITfitdata=None, DITfit=True, Concfitdataerr=None, DITfitdataerr=None, DITfitErr=True,\
             ErrLabel='', DrawSub=True, DrawDIT=True,  drawheatmap=True, \
                  ylabel=None, xlabel=None, title=None, figsize=(8,6), fignum=None):
    fig, ax = DefineFigureBPD(ylabel=ylabel, xlabel=xlabel, title=title, figsize=figsize,fignum=fignum)
    if drawheatmap: 
        im = DrawHeatmap(ax, fig, filtered_arr, extent)
        mycolor = [im.cmap(im.norm(x)) for x in subeg]
    if DrawSub: DrawSubstrate(ax, NC, substrain, subname, mycolor, myline)
    if DrawDIT: PlotDIT_points(ax, Conc, DIT)
    if DITfit: PlotDIT_Fit(ax, Concfitdata, DITfitdata)
    if DITfitErr: PlotDITerr_Fit(ax, Concfitdataerr, DITfitdataerr, ErrLabel)
    ax.legend(ncol=3, columnspacing=0.5, labelspacing=0.2, handlelength=1, handletextpad=0.3)
    return 

#------------------------------------------------------------------------------
def web_dit_scatter(XX,YY,xlabel=None,ylabel=None,title=None,fontsize=20):
    """
    This function plots the DIT points for html.

    Parameters
    ----------
    XX : Float array
        X coordinate array.
    YY : Float array
        Y coordinate array.
    xlabel : String, optional
        x-axis label. The default is None.
    ylabel : String, optional
        y-axis label. The default is None.
    title : String, optional
        Title of the figure. The default is None.
    fontsize : Float, optional
        Font size. The default is 20.

    Returns
    -------
    DITs : Holoview Scatter plot object.

    """
    DITs = hv.Scatter((XX,YY),label = "DIT").opts(color='black', size=10,xlabel=xlabel,\
                                                  ylabel=ylabel, title = title, fontsize=fontsize)
    return DITs

def web_dit_fit(XX,YY, label='', fontsize=20):
    """
    This function is to plot the DIT fitting curve in html.

    Parameters
    ----------
    XX : Float array
        X coordinate array.
    YY : Float array
        Y coordinate array.
    label : String, optional
        Label for the fitting curve. The default is ''.
    fontsize : Float, optional
        Font size. The default is 20.

    Returns
    -------
    Holoview Curve plot object.

    """
    return hv.Curve((XX,YY), label=label).opts(line_color='black',fontsize=fontsize)

def web_diterr_fit(XX,YY, label='', fontsize=20):
    """
    This function is to plot the DIT fitting curve in html.

    Parameters
    ----------
    XX : Float array
        X coordinate array.
    YY : Float array
        Y coordinate array.
    label : String, optional
        Label for the fitting curve. The default is ''.
    fontsize : Float, optional
        Font size. The default is 20.

    Returns
    -------
    Holoview Curve plot object.

    """
    return hv.Curve((XX,YY), label=label).opts(line_color='green',fontsize=fontsize)

def web_bpd_text(x, y, text=None, fontsize=20):
    """
    This function used to add text in holoview figure.

    Parameters
    ----------
    x : Float
        x-coordinate of the text.
    y : Float
        y-coordinate of the text.
    text : String, optional
        The text string. The default is None.
    fontsize : Float, optional
        Font size. The default is 20.

    Returns
    -------
    Holoview Text object.

    """
    return hv.Text(x,y,text,fontsize=fontsize)

def web_eqm_line(yeqm=0):
    """
    This function is to add horizontal line.

    Parameters
    ----------
    yeqm : Float, optional
        Y-coordinate of the horizontal line. The default is 0.

    Returns
    -------
    Holoview Hline object.

    """
    return hv.HLine(yeqm).opts(color='black',line_width=2,line_dash='dotted')

def hook(plot, element):
    #https://docs.bokeh.org/en/latest/docs/reference/models/axes.html
    #http://holoviews.org/user_guide/Customizing_Plots.html#Plot-hooks
    plot.state.title.align = "center"
    #plot.handles['xaxis'].axis_label_text_color = 'green'
    #plot.handles['yaxis'].axis_label_text_color = 'red'
    plot.handles['yaxis'].major_label_text_font_size = '50px'


    
def web_heatmap(filtered_arr, extent, tooltips, clabel=None, cmap='RdYlBu_r', interpolation='bilinear' ):
    """
    This function plots the bandgap Heatmap in BPD html.

    Parameters
    ----------
    filtered_arr : Float array
        The array for the bandgap.
    extent : TYPE
        The x and y-axis extent, should math with the dimension of the filtered_arr.
    tooltips : Hover list
        Holoview hover tooltips.
    clabel : String, optional
        Label for the colorbar. The default is None.
    cmap : Colormap object, optional
        Colormap. The default is 'RdYlBu_r'.
    interpolation : String, optional
        Interpolation method. The default is 'bilinear'.

    Returns
    -------
    inter_img : Holoview Image object.

    """
    #fontsize={'xticks':'20pt','title':'30pt',}
    hover = HoverTool(tooltips=tooltips)
    img = hv.Image(filtered_arr, bounds=(extent[0],extent[2],extent[1],extent[3]))
    img.opts(cmap=cmap,tools=[hover],colorbar=True,clabel=clabel,yformatter='%.1f',xformatter='%.2f',\
             fontsize=20, hooks=[hook]) #,clim=(0.7,2.3))
    inter_img = regrid(img, upsample=True, interpolation='bilinear')
    return inter_img

def web_plot_substrate(NC, substrain, subname, myco, myline, extent):
    """
    This function is used to plot the substrate line in BPD html.

    Parameters
    ----------
    NC : Float array
        X-array.
    substrain : Float array
        Substrate strain array.
    subname : String list/array
        List of substrate name will be used for labeling.
    myco : Color object/name list.
        List of colors that will be used for substrate line.
    myline : Line style object/name list
        List of line style that will be used for substrate.
    extent : Float array/list
        The x and y axis extent [left, right, bottom, top].

    Returns
    -------
    Holoview Curve overlay object.

    """
    suboverlay = hv.NdOverlay({subname[I]: hv.Curve((NC,sub)).opts(line_color=myco[I],line_dash=myline[I]\
                                                                   ) for I,sub in enumerate(substrain)})

    return suboverlay.opts(xlim=(extent[0],extent[1]), ylim=(extent[2],extent[3]))

def web_overlay_all(overlay_all_obj_list):
    """
    This function is to overlay all different Holoview objects.

    Parameters
    ----------
    overlay_all_obj_list : List of holoview objects
        Holoview objects like Curve, Scatter etc.

    Returns
    -------
    fullfig : Holoview final rendered figure.

    """
    overlay_all = overlay_all_obj_list[0]
    for I in overlay_all_obj_list[1:]:
        overlay_all *= I
        
    overlay_all.opts(legend_position='bottom',responsive=True, legend_cols=5)
    fullfig = hv.render(overlay_all)
    return fullfig
    
if  __name__ == '__main__':
    import numpy as np
    BANDGAP_type1= BANDGAP_type2= [np.linspace(1,10,10),np.linspace(1,10,10)]
    STRAIN = [np.linspace(1,10,10),np.linspace(2,20,10)]
    NC = ['hi','hii']
    
    plotEgStrain(STRAIN, BANDGAP_type1, BANDGAP_type2,NC)
