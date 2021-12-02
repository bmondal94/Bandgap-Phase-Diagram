#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:02:27 2021

@author: bmondal
"""
import numpy as np
from sklearn.svm import SVR
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata #, make_lsq_spline, BSpline, make_interp_spline 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#------------------------------------------------------------------------------
def Polyfit(XX, YY, order=3, xmin=None, xmax=None):
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


def SVRmodelFit(X, Y, kernel="poly", order=3, C=1, gamma="auto", epsilon=0.1,\
                datatransform=True, xmin=None, xmax=None):
    """
    This function uses Support Vector Regression ML method for fitting.

    Parameters
    ----------
    X : Float array
        X coordinates.
    Y : Float array
        Y coordinates.
    kernel : String, optional
        Type of SVR kernel. The default is "poly".
    order : Integer, optional
        Degree of polynomial for poly kernel. The default is 3.
    C : Float, optional
        Regularization parameter. The default is 1.
    gamma : {‘scale’, ‘auto’} or float, optional
        Kernel coefficient. The default is "auto".
    epsilon : Float, optional
        Epsilon in the epsilon-SVR model. The default is 0.1.
    datatransform: Bool, optional
        Whether to standarize the data or not. Default is True.
    xmin : Float, optional
        Minimum in X-data. The default is None.
    xmax : Float, optional
        Maximum in X-data. The default is None.

    Returns
    -------
    xx : Float array
        X coordinates after fitting.
    predict_y : Float array
        Y coordinate after fitting.

    """
    svr = SVR(kernel=kernel, C=C, gamma=gamma, degree=order, epsilon=epsilon, coef0=1)
    if datatransform:
        model = make_pipeline(StandardScaler(), svr)
    XX = X[:,None]
    if xmin is None: xmin=X[0]
    if xmax is None: xmax=X[-1]
    xx = np.linspace(xmin,xmax,100)[:,None]
    if datatransform:
        predict_y = model.fit(XX, Y).predict(xx)
    else:
        predict_y = svr.fit(XX, Y).predict(xx)
    return xx, predict_y

def HeatmapInterpolation(points, Bandgaparray, grid_x, grid_y, method='nearest',\
                         Sigma=2, G_filtering=True):
    """
    This function use interpolation in heatmap data.

    Parameters
    ----------
    points : 2d Float array
        The x,y coordinate sets of the data points.
    Bandgaparray : Float array
        The Bandgap at those x,y-points.
    grid_x : Float array
        The x grid array.
    grid_y : Float array
        The y grid array.
    method : String, optional
        The interpolation method. The default is 'nearest'.
    G_filtering : Bool, optional
        Gaussian filtering in the data. The default is True.
    Sigma : Scalar, optional
        Defines the sigma value in gaussian filtering. The default is 2.

    Returns
    -------
    filtered_arr : Float array
        Final heat map array.

    """
    grid_z0 = griddata(points,Bandgaparray, (grid_x, grid_y), method=method)
    filtered_arr=gaussian_filter(grid_z0.T, sigma=Sigma) if G_filtering else grid_z0.T
    return filtered_arr

def HeatmapInterpolationML(points, Bandgaparray, grid_x, grid_y, kernel="rbf", \
                           order=3, C=1, gamma="auto", epsilon=0.1):
    """
    This function use interpolation in heatmap data.

    Parameters
    ----------
    points : 2d Float array
       The x,y coordinate sets of the data points.
    Bandgaparray : Float array
       The Bandgap at those x,y-points.
    grid_x : Float array
       The x grid array.
    grid_y : Float array
       The y grid array.
    kernel : String, optional
        Type of SVR kernel. The default is "rbf".
    order : Integer, optional
        Degree of polynomial for poly kernel. The default is 3.
    C : Float, optional
        Regularization parameter. The default is 1.
    gamma : {‘scale’, ‘auto’} or float, optional
        Kernel coefficient. The default is "auto".
    epsilon : Float, optional
        Epsilon in the epsilon-SVR model. The default is 0.1.

    Returns
    -------
    filtered_arr : Float array
       Final heat map array.
    """
    xx = np.stack((grid_x.flatten(), grid_y.flatten()), axis=-1)
    svr = SVR(kernel=kernel, C=C, gamma=gamma, degree=order, epsilon=epsilon, coef0=1)
    clf = make_pipeline(StandardScaler(), svr)
    grid_z0 = clf.fit(points, Bandgaparray).predict(xx)
    filtered_arr = np.reshape(grid_z0,(100,40)).T
    return filtered_arr

def CreateHeatmapData(NC, STRAIN, BANDGAP, heatmap_bound=False, heatmap_strain_bound=None, \
                 Conc_bound=(0,100), G_filtering=True, Sigma=2, kernel="rbf", \
                     order=3, C=1, gamma="auto", epsilon=0.1,\
                     intermethod='nearest', InterpolationMethod = 'SVRML'):
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
    kernel : String, optional
        Type of SVR kernel. The default is "rbf".
    order : Integer, optional
        Degree of polynomial for poly kernel. The default is 3.
    C : Float, optional
        Regularization parameter. The default is 1.
    gamma : {‘scale’, ‘auto’} or float, optional
        Kernel coefficient. The default is "auto".
    epsilon : Float, optional
        Epsilon in the epsilon-SVR model. The default is 0.1.
    intermethod : String, optional
        The interpolation method. The default is 'nearest'.
    InterpolationMethod: String, optional
        Defines the interpolation technique. 
        SVRML: SVR machine learning model
        interpolation: Interpolation
        Default is SVRML.
    
        
    Returns
    -------
    filtered_arr : Float array
        Final heat map array. If no interpolation is used then it returns 2d array
        of coordinates with bandgap array (points, Bandgaparray).
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
    
    if InterpolationMethod == 'interpolation':
        filtered_arr = HeatmapInterpolation(points, Bandgaparray, grid_x, grid_y, \
                                            Sigma=Sigma, G_filtering=True, method=intermethod)
    elif InterpolationMethod == 'SVRML':
        filtered_arr = HeatmapInterpolationML(points, Bandgaparray, grid_x, grid_y, kernel=kernel, \
                                   order=order, C=C, gamma=gamma, epsilon=epsilon)
    else:
        filtered_arr=(points,Bandgaparray)
    
    
    return filtered_arr, [0,100,extenty_b,extenty_t]