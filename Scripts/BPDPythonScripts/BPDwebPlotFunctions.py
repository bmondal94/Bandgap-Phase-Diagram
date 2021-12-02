#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 19:27:20 2021

@author: bmondal
"""

#------------------------------------------------------------------------------
import holoviews as hv 
from holoviews import opts
from holoviews.operation.datashader import regrid
from bokeh.models.tools import *
hv.extension('bokeh')

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
                                                    ,line_width=4,) for I,sub in enumerate(substrain)})

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