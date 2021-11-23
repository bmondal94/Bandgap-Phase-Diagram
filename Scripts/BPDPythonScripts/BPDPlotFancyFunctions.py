#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 18:24:52 2021

@author: bmondal
"""

#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

#%%----------------------------------------------------------------------------
def rainbowarrow(basex,basey,longx,longy,hl,width=0.1,fraction=1/2,mm_min_lim=0,mm_max_lim=1,cmap=plt.cm.RdYlBu_r,vertical=False,reverse_ar=False,front=False,colorflip=False):
    '''
    Parameters
    ----------
    basex : float
        x-cordinate of arrow starting point. (left-bottom)
    basey : float
        y-cordinate of arrow starting point. (left-bottom)
    longx : float
        Length of arrow along x including arrow head.
    longy : float
        Length of arrow along y including arrow head.
    hl : float
        Arrow head length.
    width : float, optional
        Width of arrow. The default is 0.1.
    fraction : float, optional
        Fraction of arrow for color. The default is 1/2.
    mm_min_lim : float, optional
        Color scale minimum between 0-1. The default is 0.
    mm_max_lim : float, optional
        Color scale maxima between 0-1. The default is 1.
    cmap: color map, optional
        The default is plt.cm.RdYlBu_r
    vertical : bool, optional
        Arrow in vertical direction. The default is False.
    reverse_ar : bool, optional
        Arrow in reverse direction. The default is False.
    front : bool, optional
        Colors in front part of the arrow. The default is False.

    Returns
    -------
    None.

    '''
    colors=np.linspace(mm_max_lim, mm_min_lim,50)
    
    if reverse_ar:
        basex+=longx
        basey+=longy
        longx = -longx
        longy = -longy
        hl = -hl
        colors=np.flip(colors)
    if colorflip:
        colors=np.flip(colors)
    #plt.arrow(basex,basey+width*0.5,longx,longy,width=width,length_includes_head=True,head_length=hl,fc='k')
    
    
    if vertical:
        colors=np.flip(colors)
        frac=longy*fraction
        rect=plt.Rectangle((basex,basey),width, longy-hl,linewidth = 0,fc = 'k', antialiased=True)
        XY=[[basex+width*2,longy-hl+basey],[basex-width,longy-hl+basey],[basex+width*0.5,basey+longy]]
    else:
        frac=longx*fraction
        rect=plt.Rectangle((basex,basey),longx-hl,width, linewidth = 0,fc = 'k', antialiased=True)
        XY=[[basex+longx-hl,basey+width*2],[basex+longx-hl,basey-width],[basex+longx,basey+width*0.5]]
    plt.gca().add_patch(rect)
    
    c=cmap(colors)
    if front:
        calco=c[-1]
        if vertical:
            x=np.linspace(basey+frac,basey+longy-hl,50) 
        else:
            x=np.linspace(basex+frac,basex+longx-hl,50)
    else:
        calco='k'
        if vertical:
            x=np.linspace(basey, basey+frac,50)
        else:
            x=np.linspace(basex, basex+frac,50)
    
    trian=plt.Polygon(XY,linewidth = 0,fc = calco, antialiased=True)
    plt.gca().add_patch(trian)
    diffn = x[1]-x[0]
    for i in range(len(x)):
        if vertical:
            rect=plt.Rectangle((basex,x[i]),width, diffn,linewidth = 0,fc = c[i], antialiased=True) 
        else:
            rect=plt.Rectangle((x[i],basey),diffn,width, linewidth = 0,fc = c[i], antialiased=True)
        plt.gca().add_patch(rect)
        
def DrawArrow(basex,basey,longx,longy, harraow=True, varrow=False):
    if harraow:
    # Horizontal arrows
        rainbowarrow(13,0.32 ,50,0,hl=8,width=0.3,mm_max_lim=0.4,mm_min_lim=0.6,cmap=plt.cm.prism,vertical=False)
        rainbowarrow(13,-0.62,50,0,hl=8,width=0.3,mm_max_lim=0.4,mm_min_lim=0.6,cmap=plt.cm.prism,reverse_ar=True,front=True)  
    elif varrow:
    #verical arrows
        rainbowarrow(24,0,0,4,hl=0.5,width=2,mm_max_lim=0.4,mm_min_lim=0.6,cmap=plt.cm.prism,fraction=0.58,reverse_ar=False,front=False,vertical=True)
        rainbowarrow(24,-5,0,5,hl=0.5,width=2,mm_max_lim=0.4,mm_min_lim=0.56,cmap=plt.cm.prism,fraction=0.56,reverse_ar=True,front=False,vertical=True,colorflip=True)
    else:
       rainbowarrow(13,0.32 ,50,0,hl=8,width=0.3,mm_max_lim=0.4,mm_min_lim=0.6,cmap=plt.cm.prism,vertical=False)
       rainbowarrow(13,-0.62,50,0,hl=8,width=0.3,mm_max_lim=0.4,mm_min_lim=0.6,cmap=plt.cm.prism,reverse_ar=True,front=True)
       rainbowarrow(24,0,0,4,hl=0.5,width=2,mm_max_lim=0.4,mm_min_lim=0.6,cmap=plt.cm.prism,fraction=0.58,reverse_ar=False,front=False,vertical=True)
       rainbowarrow(24,-5,0,5,hl=0.5,width=2,mm_max_lim=0.4,mm_min_lim=0.56,cmap=plt.cm.prism,fraction=0.56,reverse_ar=True,front=False,vertical=True,colorflip=True)
