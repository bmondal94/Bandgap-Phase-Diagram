#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:02:27 2021

@author: bmondal
"""
import numpy as np
from sklearn.svm import SVR
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata,SmoothBivariateSpline,RBFInterpolator,bisplev,bisplrep #, make_lsq_spline, BSpline, make_interp_spline 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_validate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error, classification_report, \
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import time
import pickle
import matplotlib.pyplot as plt

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

def HeatmapInterpolation_griddata(points, Bandgaparray, grid_x, grid_y, method='nearest',\
                                  Sigma=2, G_filtering=False):
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
    grid_points = (grid_x, grid_y)
    grid_z0 = griddata(points,Bandgaparray, grid_points, method=method,rescale=True)
    # print(grid_points[np.argwhere(np.isnan(grid_z0))])
    # print(grid_points)
    filtered_arr=gaussian_filter(grid_z0.T, sigma=Sigma) if G_filtering else grid_z0.T
    return filtered_arr

def HeatmapInterpolation_smoothspline(x,y,z, grid_x, grid_y,smoothing=None):
    """
    This function use interpolation in heatmap data.

    Parameters
    ----------
    x,y,z : 1d Float array
        The x,y coordinate sets of the data points. z value at each (x,y) point.
    grid_x : Float array
        The x grid array.
    grid_y : Float array
        The y grid array.
    Returns
    -------
    filtered_arr : Float array
        Final heat map array.

    """
    interp_my = SmoothBivariateSpline(x, y, z,s=smoothing)
    XXXX = grid_x.flatten() ; YYYY = grid_y.flatten() 
    RESULTS = interp_my.ev(XXXX,YYYY,dx=0,dy=0)
    print(f'The residual in SmoothBivariateSpline = {interp_my.get_residual():0.3f}')
    filtered_arr = RESULTS.reshape(np.shape(grid_x))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(XXXX,YYYY,c=RESULTS,vmin=0,vmax=2.5,cmap=plt.cm.RdYlBu_r)
    plt.figure()
    plt.scatter(x,y,c=z,vmin=0,vmax=2.5,cmap=plt.cm.RdYlBu_r)
    plt.colorbar()
    return filtered_arr.T


def HeatmapInterpolation_RBF(points, Bandgaparray, grid_x, grid_y, kernel="gaussian", \
                           order=3, smoothing=0, epsilon=0.1):
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
    epsilon : Float, optional
        Epsilon in the epsilon-SVR model. The default is 0.1.

    Returns
    -------
    filtered_arr : Float array
       Final heat map array.
    """
    # if kernel == 'gaussian': order=-1
    print('RBF interpolator is chosen.')
    print(f'kernel={kernel},smoothing={smoothing},epsilon={epsilon},degree={order}')
    xx = np.stack((grid_x.flatten(), grid_y.flatten()), axis=-1)
    grid_z0 = RBFInterpolator(points, Bandgaparray, kernel=kernel, neighbors=None,
                              smoothing=smoothing, degree=order, epsilon=epsilon)(xx)
    filtered_arr = np.reshape(grid_z0,np.shape(grid_x)).T
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
    print("Evaluate cross validation over whole data set.")
    kfold = KFold(n_splits=10, random_state=1, shuffle=True)  
    cv_results = cross_val_score(clf, points, Bandgaparray, cv=kfold, scoring='r2', n_jobs=-1)
    print(clf.get_params(deep=True))
    print(f"Score (r2): {cv_results.mean():.3f}+/-{cv_results.std():.3f}")

    filtered_arr = np.reshape(grid_z0,np.shape(grid_x)).T
    return filtered_arr

def Plot_Prediction_Actual_results(X, Y, wrongdata_cutoff=None, tsxt=None, save=False, savepath='.', marker='o',figname='TruePrediction.png'):
    
    plt.figure(figsize=(8, 8))
    plt.xlabel("True values (eV)")
    plt.ylabel("Predictions (eV)")
    plt.scatter(X, Y, marker=marker)
    

    lim_min, lim_max = min(X), max(X)
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.plot([lim_min, lim_max],[lim_min, lim_max], color='k')
    
    plt.title(tsxt)
    DIFF_ = abs(X - Y)
    DIFF_wring_id = None
    if wrongdata_cutoff:
        DIFF_wring_id = np.argwhere(DIFF_>wrongdata_cutoff)
        plt.scatter(X[DIFF_wring_id], Y[DIFF_wring_id], marker='x')
        print(f'Wrong samples have prediction error >= {wrongdata_cutoff} eV')
        
    # ax.plot([],[],' ',label=f"Max error = {max(DIFF_):.2f} eV \n {'MAE':>11} = {np.mean(DIFF_):.2f}±{np.std(DIFF_):.2f} eV")
    plt.plot([],[],' ',label=f"Max error = {max(DIFF_):.2f} eV \nMAE = {np.mean(DIFF_):.2f}±{np.std(DIFF_):.2f} eV")
    plt.legend(handlelength=0)
    plt.tight_layout()
    if save:
        plt.savefig(savepath+'/'+figname,bbox_inches = 'tight',dpi=300)
        plt.close()
    else:
        plt.show()
    return DIFF_wring_id
def HeatmapInterpolationML_V2(X, Y, grid_x, grid_y, kernel="rbf", refit = True, 
                               scoringfn='r2', njobs=-1,
                               save=False, filename='test.sav'):
    # Fit regression model
    # Radial Basis Function (RBF) kernel
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.2)
    
    # Define a Standard Scaler to normalize inputs
    scaler = StandardScaler()
    tol = 1e-5;cache_size=10000
    svr = SVR(kernel="rbf",tol=tol,cache_size=cache_size)   
    # Set pipeline
    pipe = Pipeline(steps=[("scaler", scaler), ("svrm", svr)])
    
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    C_range = [1e0, 1e1, 1e2, 1e3]
    gamma_range = np.logspace(-4, 2, 50)
    param_grid={"svrm__C": C_range, 
                "svrm__gamma": gamma_range}
    svrgrid = GridSearchCV(estimator=pipe,
                           param_grid=param_grid,
                           cv = 5,
                           scoring = scoringfn,
                           n_jobs=njobs,
                           refit=refit)
    
    t0 = time.time()
    svrgrid.fit(X_train, y_train)
    svr_fit = time.time() - t0
    print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit)
    print(f'Training dataset size: {X_train.shape[0]}')
   
    #print('All cv results:',svrgrid.cv_results_)
    
    # if PlotResults:
    #     _ = PlotCV_results(svrgrid, C_range, gamma_range)
    
    print("Best parameter (CV score=%0.3f):" % svrgrid.best_score_, svrgrid.best_params_)
    nsupports = svrgrid.best_estimator_['svrm'].n_support_
    print(f"Number of support vectors: Total={sum(nsupports)} ;  for each class=", nsupports)
    #print('* The scorer used:', inspect.getsource(svrgrid.scorer_))

    if refit:
        print('Refitting timing for the best model on the whole training dataset:%.3f s' % svrgrid.refit_time_)
        print("The number of cross-validation splits (folds/iterations): %d" % svrgrid.n_splits_)
    
        t0 = time.time()
        y_svr = svrgrid.predict(X_test)
        Y_predict_all = svrgrid.predict(X)
        svr_predict = time.time() - t0
        
        print(f"The out-of-sample r2_score for prediction: {r2_score(y_test, y_svr):.3f}")
        print(f"The out-of-sample mean_squared_error for prediction: {mean_squared_error(y_test, y_svr):.3f} eV2")
        print(f"The out-of-sample mean_absolute_error for prediction: {mean_absolute_error(y_test, y_svr):.3f} eV")
        print(f"The out-of-sample max_error for prediction: {max_error(y_test, y_svr):.3f} eV")
        print('\n')
        print(f"The all-data r2_score for prediction: {r2_score(Y,Y_predict_all):.3f}")
        print(f"The oall-data mean_squared_error for prediction: {mean_squared_error(Y,Y_predict_all):.3f} eV2")
        print(f"The all-data mean_absolute_error for prediction: {mean_absolute_error(Y,Y_predict_all):.3f} eV")
        print(f"The all-data max_error for prediction: {max_error(Y,Y_predict_all):.3f} eV")
        Plot_Prediction_Actual_results(y_test, y_svr, tsxt='Bandgap prediction on test set') #,save=save, savepath=savepath, figname='TestSet'+figname)     
        DIFF_wring_id = Plot_Prediction_Actual_results(Y,Y_predict_all, tsxt='Bandgap prediction over all data',wrongdata_cutoff=0.15) #,save=save, savepath=savepath, figname=figname)
        X_wrong = X[DIFF_wring_id]
        figg = plt.figure()
        # plt.scatter( X_wrong[:,0][:,0], X_wrong[:,0][:,1],c=abs(Y-Y_predict_all)[DIFF_wring_id][:,0])
        imm = plt.scatter( X[:,0], X[:,1],c=abs(Y-Y_predict_all))
        cbar = figg.colorbar(imm,format='%.2f')
        cbar.ax.set_ylabel('Error (eV)')
        # print(X[:,0])
        
        print("SVR prediction for %d (test) inputs in %.3f s" % (X_test.shape[0], svr_predict))
        # print(f"The r2_score for prediction: {r2_score(y_test, y_svr):.3f}")
        # print(f"The mean_squared_error for prediction: {mean_squared_error(y_test, y_svr):.3f}")
        model_best_scorer = scoringfn if isinstance(refit,bool)  else refit
        print(f"Score on prediction using the 'default' scorer ({model_best_scorer}): {svrgrid.score(X_test, y_test): .3f}")
        #print("Number of support vectors:", len(svrgrid.best_estimator_['svrm'].n_support_))
        BestModel, _ = svrgrid.best_estimator_,svrgrid.best_score_
        npipe = BestModel
    else:
        BestModel, _ = svrgrid.best_params_,svrgrid.best_score_
        # Define a Standard Scaler to normalize inputs
        scaler = StandardScaler()
        # Set model
        svr = SVR(kernel="rbf")
        # Set pipeline
        npipe = Pipeline(steps=[("scaler", scaler), ("svrm", svr)])        
        # Parameters of pipelines can be set using ‘__’ separated parameter names:
        npipe.set_params(**BestModel)
        
    print("-----------------------------------------------")
    # print('Model details:', npipe.get_params())
    # print("-----------------------------------------------")
    print("Evaluate cross validation over whole data set.")
    kfold = KFold(n_splits=10, random_state=1, shuffle=True)
    if isinstance(scoringfn, str):   
        cv_results = cross_val_score(npipe, X, Y, cv=kfold, scoring=scoringfn, n_jobs=njobs)
        print(f"Score ({scoringfn}): {cv_results.mean():.3f}+/-{cv_results.std():.3f}")
    else:
        cv_results = cross_validate(npipe, X, Y, cv=kfold, scoring=scoringfn, n_jobs=njobs)
        for I in scoringfn:
            print(f"Score ({I}): {cv_results['test_'+I].mean():.3f}+/-{cv_results['test_'+I].std():.3f}")
    
    npipe.fit(X, Y)
    xx = np.stack((grid_x.flatten(), grid_y.flatten()), axis=-1)
    grid_z0 = npipe.predict(xx)
    filtered_arr = np.reshape(grid_z0,np.shape(grid_x)).T

    if save:
        pickle.dump(npipe, open(filename, 'wb'))
        print(f"* Model saved: {filename}")
        
    print("***************************************************************************\n")
    return filtered_arr

def CreateHeatmapData(NC, STRAIN, BANDGAP, preprocessing=True, heatmap_bound=False, heatmap_strain_bound=None, \
                      Conc_bound=(0,100), StrainBound=None, G_filtering=True, Sigma=2, kernel="rbf", \
                          order=3, C=1, gamma="auto", epsilon=0.1,smoothing=None,\
                              intermethod='nearest', InterpolationMethod = 'SVRML', \
                                  refit = True, scoringfn='r2', save=False, filename='test.sav'):
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
        SVRML_V2: SVR machine learning model with best parameter search
        griddata: Interpolation
        SmoothBivariateSpline: Interpolation
        Default is SVRML.
    refit: Bool
        Refitting for SVRML_V2 model. Defaulr is True.
    scoringfn:
        Scoring function for SVRML_V2 model. Default is "r2".
    save: Bool
        Save the trained best model. Default=False.
    filename:
        Filepath to save the model. Default is "test.sav".
        
    
        
    Returns
    -------
    filtered_arr : Float array
        Final heat map array. If no interpolation is used then it returns 2d array
        of coordinates with bandgap array (points, Bandgaparray).
    list [x_left, x_right, y_bottom, y_top]
        Extent of x and y axis.

    """
    if preprocessing:
        Xarray=[]
        for ii in range(0,len(NC)):
            xxaxis = [NC[ii]]*len(STRAIN[ii])
            Xarray+= xxaxis
        
        Xarray = np.array(Xarray, dtype=float)
        strainarray = np.concatenate(STRAIN, axis=0)
        Bandgaparray = np.concatenate(BANDGAP, axis=0) 
    else:
        Xarray = NC; strainarray = STRAIN; Bandgaparray=BANDGAP
        
    
    if heatmap_bound:
        if heatmap_strain_bound:
            strainarray_cutoff_ind = np.argwhere(abs(strainarray)>heatmap_strain_bound)
    
        if Conc_bound:
            NC_index = np.concatenate((np.argwhere(Xarray<Conc_bound[0]),np.argwhere(Xarray>Conc_bound[1])))  
            if heatmap_strain_bound:
                strainarray_cutoff_ind = np.concatenate((strainarray_cutoff_ind,NC_index))
            else:
                strainarray_cutoff_ind = NC_index
        
        strainarray = np.delete(strainarray, strainarray_cutoff_ind)
        Bandgaparray = np.delete(Bandgaparray, strainarray_cutoff_ind)
        Xarray = np.delete(Xarray, strainarray_cutoff_ind)

    extenty_b = np.amin(strainarray) if StrainBound is None else StrainBound[0]
    extenty_t = np.amax(strainarray) if StrainBound is None else StrainBound[1]
    Xesolution = complex(0,abs(Conc_bound[1]-Conc_bound[0])*10+1)
    Yresolution = complex(0,abs(extenty_t-extenty_b)*10+1)
    grid_x, grid_y = np.mgrid[Conc_bound[0]:Conc_bound[1]:Xesolution, extenty_b:extenty_t:Yresolution]

    points = np.stack((Xarray, strainarray),axis=-1)
    
    if InterpolationMethod == 'griddata':
        filtered_arr = HeatmapInterpolation_griddata(points, Bandgaparray, grid_x, grid_y, \
                                                     Sigma=Sigma, G_filtering=G_filtering, method=intermethod)
    elif InterpolationMethod == 'RBFInterpolator':
        if kernel not in ['linear','thin_plate_spline','cubic','quintic','multiquintic','inverse_multiquadric','inverse_quadratic']:
            kernel = 'gaussian' 
        filtered_arr = HeatmapInterpolation_RBF(points, Bandgaparray, grid_x, grid_y, kernel=kernel, \
                                                order=order, smoothing=smoothing, epsilon=epsilon)
    elif InterpolationMethod == 'SmoothBivariateSpline':
        filtered_arr = HeatmapInterpolation_smoothspline(Xarray,strainarray,Bandgaparray,grid_x, grid_y,smoothing=smoothing)
        
    elif InterpolationMethod == 'SVRML':
        filtered_arr = HeatmapInterpolationML(points, Bandgaparray, grid_x, grid_y, kernel=kernel, \
                                              order=order, C=C, gamma=gamma, epsilon=epsilon)
    elif InterpolationMethod == 'SVRML_V2':
        filtered_arr = HeatmapInterpolationML_V2(points, Bandgaparray, grid_x, grid_y, kernel=kernel, 
                                                 refit=refit, scoringfn=scoringfn,save=save, filename=filename)
    else:
        filtered_arr=(points,Bandgaparray)
        
    
    return filtered_arr, [Conc_bound[0],Conc_bound[1],extenty_b,extenty_t]


def CreateHeatmapData_new(NC, STRAIN, BANDGAP, preprocessing=True, heatmap_bound=False, heatmap_strain_bound=None, \
                      Conc_bound=(0,100), StrainBound=None, G_filtering=True, Sigma=2, kernel="rbf", \
                          order=3, C=1, gamma="auto", epsilon=0.1,smoothing=None,\
                              intermethod='nearest', InterpolationMethod = 'SVRML', \
                                  refit = True, scoringfn='r2', save=False, filename='test.sav'):
    if preprocessing:
        Xarray=[]
        for ii in range(0,len(NC)):
            xxaxis = [NC[ii]]*len(STRAIN[ii])
            Xarray+= xxaxis
        
        Xarray = np.array(Xarray, dtype=float)
        strainarray = np.concatenate(STRAIN, axis=0)
        Bandgaparray = np.concatenate(BANDGAP, axis=0) 
    else:
        Xarray = NC; strainarray = STRAIN; Bandgaparray=BANDGAP
        
    
    if heatmap_bound:
        if heatmap_strain_bound:
            strainarray_cutoff_ind = np.argwhere(abs(strainarray)>heatmap_strain_bound)
    
        if Conc_bound:
            NC_index = np.concatenate((np.argwhere(Xarray<Conc_bound[0]),np.argwhere(Xarray>Conc_bound[1])))  
            if heatmap_strain_bound:
                strainarray_cutoff_ind = np.concatenate((strainarray_cutoff_ind,NC_index))
            else:
                strainarray_cutoff_ind = NC_index
        
        strainarray = np.delete(strainarray, strainarray_cutoff_ind)
        Bandgaparray = np.delete(Bandgaparray, strainarray_cutoff_ind)
        Xarray = np.delete(Xarray, strainarray_cutoff_ind)

    extenty_b = np.amin(strainarray) if StrainBound is None else StrainBound[0]
    extenty_t = np.amax(strainarray) if StrainBound is None else StrainBound[1]
    Xesolution = complex(0,abs(Conc_bound[1]-Conc_bound[0])*10+1)
    Yresolution = complex(0,abs(extenty_t-extenty_b)*1+1)
    grid_x, grid_y = np.mgrid[Conc_bound[0]:Conc_bound[1]:Xesolution, extenty_b:extenty_t:Yresolution]

    interp_my = SmoothBivariateSpline(Xarray,strainarray,Bandgaparray,s=smoothing)
    UNIQUE_conc = np.unique(Xarray)
    
    XXXX = grid_x.flatten() ; YYYY = grid_y.flatten() 
    RESULTS = interp_my.ev(XXXX,YYYY,dx=0,dy=0)
    print(f'The residual in SmoothBivariateSpline = {interp_my.get_residual():0.3f}')
    filtered_arr = RESULTS.reshape(np.shape(grid_x))
    filtered_arr = HeatmapInterpolation_smoothspline(Xarray,strainarray,Bandgaparray,grid_x, grid_y,smoothing=smoothing)

        
    
    return filtered_arr, [Conc_bound[0],Conc_bound[1],extenty_b,extenty_t]

#%%
#%%----------------- iterative interpolation image processing -----------------
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata,SmoothBivariateSpline,RBFInterpolator
from scipy.spatial import distance_matrix, KDTree, Delaunay
import itertools

def CheckRescaleCutoff(cutoff, scale, gridresolutionX, gridresolutionY):
    MAXgridresolution = 1/min(gridresolutionX, gridresolutionY) if cutoff is None else cutoff
    tmp = MAXgridresolution / scale
    # print(cutoff, scale, tmp )
    return max(tmp)+0.0001
      
def Rescaleing(points, Offset=None, Scale=None):
    # scale to unit cube centered at 0
    offset = np.mean(points, axis=0) if Offset is None else Offset 
    points = points - offset
    scale = points.ptp(axis=0) if Scale is None else Scale
    scale[~(scale > 0)] = 1.0  # avoid division by 0
    points /= scale
    return points, (offset, scale)

def ReverseRescaleing(points, offset, scale):
    # scale[~(scale > 0)] = 1.0  
    points = (points*scale) + offset
    return points

def BoundHeatMap(X, Y, Z, Xbound=None, Ybound=None):    
    X_cutoff_ind = np.array([]); Y_cutoff_ind = np.array([])
    if Xbound is not None:
        assert len(Xbound)>1, 'The bound should be list/array/tuple of 2 values.'
        X_cutoff_ind = np.concatenate((np.argwhere(X<Xbound[0]),np.argwhere(X>Xbound[1])))
    
    if Ybound is not None:
        assert len(Ybound)>1, 'The bound should be list/array/tuple of 2 values.'
        Y_cutoff_ind = np.concatenate((np.argwhere(Y<Ybound[0]),np.argwhere(Y>Ybound[1])))
        
    if (Xbound is None) and (Ybound is None):
        print('Warning: Nothing to bound by. Returning original array.')
        
    else:
        Total_cutoff_ind = np.concatenate((X_cutoff_ind,Y_cutoff_ind))
        X = np.delete(X, Total_cutoff_ind)
        Y = np.delete(Y, Total_cutoff_ind)
        Z = np.delete(Z, Total_cutoff_ind)
        
    return X, Y, Z

def CreateFullInterpolationGrid(Xbound, Ybound, factorX=10, factorY=10):
    assert len(Xbound)>1, 'The x-bound should be list/array/tuple of 2 values.'
    assert len(Ybound)>1, 'The y-bound should be list/array/tuple of 2 values.'
    Xesolution = complex(0,int(abs(Xbound[1]-Xbound[0])*factorX+1))
    Yresolution = complex(0,int(abs(Ybound[1]-Ybound[0])*factorY+1))
    print(f'Composition resolution in grid: {1/factorX:.3f} %')
    print(f'Strain resolution in grid: {1/factorY:.3f} %')
    grid_x, grid_y = np.mgrid[Xbound[0]:Xbound[1]:Xesolution, Ybound[0]:Ybound[1]:Yresolution]
    return grid_x, grid_y

def FindNeighbours(points1, points2, cutoff, loop, methodt='griddata', submethodt='nearest',pltottriplot=0): 
    # build the KDTree using the *larger* points array
    tree = KDTree(points1)
    groups = tree.query_ball_point(points2, cutoff,workers=-1)
    indices = np.unique(list(itertools.chain.from_iterable(groups)))
    if (methodt=='griddata') and (submethodt in ['linear','cubic']):
        tri = Delaunay(points2)
        out_side_points_id = (np.argwhere(tri.find_simplex(points1) == -1)).flatten()
        mask = np.isin(indices, out_side_points_id, invert=True)
        indices = indices[mask]
        if pltottriplot:
            plt.figure()
            plt.title(f'Loop={loop}')
            plt.triplot(points2[:,0], points2[:,1], tri.simplices)
            plt.plot(points2[:,0], points2[:,1], 'o')
            plt.plot(points1[out_side_points_id][:,0], points1[out_side_points_id][:,1], 'o',c='b')
            plt.plot(points1[indices][:,0], points1[indices][:,1], 'o',c='r')
            plt.show() 
    return indices

def PerformInterpolation(points,values,xi,methodt='griddata',submethodt='linear',smoothingt=None):
    if methodt=='griddata':
        assert submethodt in ['linear','nearest','cubic'], 'giddata() does not support supplied interpolation method.'
        grid_z0 = griddata(points,values,xi,method=submethodt)
    elif methodt=='RBFInterpolator':
        grid_z0 = RBFInterpolator(points,values)
    elif methodt=='SmoothBivariateSpline':
        interp_my = SmoothBivariateSpline(points[:,0], points[:,1], values,s=smoothingt)
        grid_z0 = interp_my.ev(xi[:,0],xi[:,1],dx=0,dy=0)
        print(f'The residual in SmoothBivariateSpline = {interp_my.get_residual():0.3f}')  
    elif methodt=='BivariateSpline':
        # tck, fp, ier, msg = bisplrep(points[:,0], points[:,1], values,s=smoothingt,nxest=len(points[:,0]),nyest=len(points[:,1]),full_output=1)
        tck, fp, ier, msg = bisplrep(points[:,0], points[:,1], values,s=smoothingt,nxest=50,nyest=50,full_output=1)
        print(f"fp={fp}\nier={ier}\n{msg}\nnx={len(tck[0])},\nny={len(tck[1])}")
        # print(tck,len(np.unique(xi[:,0])),len(np.unique(xi[:,1])))
        grid_z0 = bisplev(np.unique(xi[:,0]),np.unique(xi[:,1]),tck)
    return grid_z0

def UsingIterativeInterpolation(points_grid, points_data, Z, cutoff=0.1, methodt='griddata',IterativeInterpolation=True,
                                submethodt='cubic',smoothingt=None,fillval=np.nan,FillwithNearestNeighbor=True):
    points_grid_ = points_grid.copy()
    points_in_cutoff = points_data.copy()
    points_data_loop = points_data.copy()
    Z_val = Z.copy()
    points_iterative, points_iterative_Z_val = [points_data], [Z]
    if IterativeInterpolation and (methodt in ['RBFInterpolator', 'SmoothBivariateSpline', 'BivariateSpline']):
        print(f"For {methodt} our iterative interpolation scheme can not be performed. Switching to single-shot interpolation.")
        IterativeInterpolation = False
        
    if IterativeInterpolation:
        loop = 1
        TotalPoints = len(points_grid_)
        FormatInt = len(str(TotalPoints))
        print(f'Total grid points to cover = {TotalPoints}')
        while len(points_grid_)>0:
            points_in_cutoff_indices = FindNeighbours(points_grid_, points_data_loop, cutoff, loop, methodt=methodt, submethodt=submethodt) # find neighbours indices inside cutoff
            if len(points_in_cutoff_indices)<1: 
                if FillwithNearestNeighbor:
                    print('Warning: Rest of the points can not be covered with convex hull. Those points will be nearest-neighbor interpolated.')
                    points_in_cutoff_indices = np.s_[:]
                    submethodt = 'nearest'
                else:
                    print('Warning: Rest of the points can not be covered with convex hull. Those points will be filled with fill-value.')
                    points_iterative.append(points_grid_) # Save the points where interpolation can not be done
                    points_iterative_Z_val.append(np.array([fillval]*len(points_grid_))) # Save the Z-value of points where interpolation can not be done with fill value
                    break
            points_in_cutoff = points_grid_[points_in_cutoff_indices] # Neighbors inside cutoff
            grid_z0 = PerformInterpolation(points_data_loop,Z_val,points_in_cutoff,methodt=methodt,submethodt=submethodt,smoothingt=smoothingt) # Do interpolation
            points_grid_ = np.delete(points_grid_, points_in_cutoff_indices, axis=0) # update points where to find next interpolations
            
            points_iterative.append(points_in_cutoff) # Save the points where interpolation is done
            points_iterative_Z_val.append(grid_z0) # Save the Z-value of points where interpolation is done
            
            # points_data_loop = points_in_cutoff.copy() # update points which will be used to find neighbors and do interpolations
            # Z_val = grid_z0.copy()  # update Z-value of points which will be used to find neighbors and do interpolations
            points_data_loop = np.concatenate(points_iterative)
            Z_val = np.concatenate(points_iterative_Z_val)
            print(f'Loop = {loop: <4}, Points cover = {len(points_in_cutoff): >{FormatInt}}, Left points = {len(points_grid_): >{FormatInt}}')
            loop += 1
    else:
        points_iterative.append(points_grid_)
        grid_z0 = PerformInterpolation(points_data_loop,Z_val,points_grid_,methodt=methodt,submethodt=submethodt,smoothingt=smoothingt) # Do interpolation
        NANpos = np.argwhere(np.isnan(grid_z0)).flatten()
        if len(NANpos)>0:
            print(f"Warning: For {len(NANpos)} points nearest neighbor interpolation is performed.")
            if FillwithNearestNeighbor:
                NotNANpos = np.argwhere(np.isfinite(grid_z0)).flatten()
                grid_z0_nn = PerformInterpolation(points_grid_[NotNANpos],grid_z0[NotNANpos],points_grid_[NANpos],methodt='griddata',submethodt='nearest')
                # print(grid_z0[NANpos],grid_z0_nn)
                grid_z0[NANpos] = grid_z0_nn
            else:
                grid_z0[NANpos] = fillval       
        points_iterative_Z_val.append(grid_z0)
    return points_iterative[1:], points_iterative_Z_val[1:]

def ImageProcessing(X,Y,Z,gridextent, Xbound=(None,None),Ybound=(None,None),IterativeInterpolation=True,
                    gridresolutionX=10,gridresolutionY=10,FillwithNearestNeighbor=True,
                    cutoff=None,method='griddata',submethod='cubic',fillval=np.nan,smoothing=None):
    #### Bound the data
    X_b, Y_b, Z_b = BoundHeatMap(X, Y, Z, Xbound=Xbound, Ybound=Ybound)

    #### Create full grid for prediction
    fgridx, fgridy = CreateFullInterpolationGrid(gridextent[0:2], gridextent[2:], factorX=gridresolutionX, factorY=gridresolutionY)
    
    #### Rescale the data
    points_data, (offset, scale) = Rescaleing(np.c_[X_b.ravel(), Y_b.ravel()])
    points_grid, _ = Rescaleing(np.c_[fgridx.ravel(), fgridy.ravel()], Offset=offset, Scale=scale)
    
    #### Normalize and Check the cutoff
    cutoff = CheckRescaleCutoff(cutoff, scale, gridresolutionX, gridresolutionY)
    print(f"cutoff (rescaled) = {cutoff:.2f}, scale = {scale}") # Note: this is rescaled cutoff
    
    #### Create interpolation iteratively 
    points_iterative, points_iterative_Z_val = UsingIterativeInterpolation(points_grid, points_data, Z_b, 
                                                                           methodt=method,submethodt=submethod,IterativeInterpolation=IterativeInterpolation,
                                                                           cutoff=cutoff,smoothingt=smoothing,fillval=fillval,
                                                                           FillwithNearestNeighbor=FillwithNearestNeighbor)
    #### Reverse rescale 
    points_iterative = [ReverseRescaleing(POINTS, offset, scale) for POINTS in points_iterative]
    
    AFTER_ITER_P = np.concatenate(points_iterative)
    AFTER_ITER_P_Z = np.concatenate(points_iterative_Z_val)
    #### Reshaping
    if method == 'BivariateSpline':
        AFTER_ITER_P_Z_reshape = points_iterative_Z_val[0].copy()
    else:
        sort_id = AFTER_ITER_P[:,0].argsort()
        sort_id_ = ((AFTER_ITER_P[sort_id][:,1]).reshape(np.shape(fgridx))).argsort(axis=1)
        AFTER_ITER_P_Z_reshape = np.take_along_axis(AFTER_ITER_P_Z[sort_id].reshape(np.shape(fgridx)), sort_id_, axis=1)
    
    return (X_b, Y_b, Z_b), (offset, scale), (fgridx, fgridy), \
        (points_iterative, points_iterative_Z_val), (AFTER_ITER_P, AFTER_ITER_P_Z, AFTER_ITER_P_Z_reshape), np.shape(fgridx)

        
