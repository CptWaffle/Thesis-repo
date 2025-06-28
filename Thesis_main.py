#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis_main.py

Purpose:
    Main work file of the thesis.
    General functions and methods can be imported from folder

Date:
    YYYY/MM/DD

Author:
    Sander Tromp
"""
###########################################################
### Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as opt
import scipy as sp
from timeit import default_timer as timer

### Import python files
# from Utilities.lib.grad import *
###########################################################
### Make class for storage
class cMaximizeLikelihood:
    def __init__(self):
            self.x0 = []
            self.x = []
            self.tx0 = []
            self.tx = []
            self.likelihoodvalue = []
            self.tcovariancematrix = []
            self.covariancematrix = []
            self.standarderrors = []
            self.tstandarderrors = []
            self.tCI = ()
            self.success = False

###########################################################
### dY= emptyfunc(vX)
def RMSE_ARpanel(mY, dRho):

    """
    Purpose:
        Compute the RMSE for an AR-panel model.
        For now only capable of using an estimated auto-regressive component.
    Inputs:
        mY              matrix of dep. variable (iN x iT)
        dRho            double, estimated autoregressive parameter

    Return value:
        dRMSE            double, mean squared error of the estimated parameter
    """
    # Initialize regression matrices
    mY1, mX0 = Lag_AR1(mY)
    vE = mY1 - mX0 * dRho
    dRMSE = np.sqrt(np.sum(np.square(vE)))
    return dRMSE

###########################################################
### dY= emptyfunc(vX)
def Monte_Carlo_Panel(vT, vKappa, vGamma, vTheta, iM, vInference, vBool, sInit):

    """
    Purpose:
        Perform a pre-specified Monte Carlo routine to estimate different model
        specifications with the considered estimators.
        Considered estimators:
            - FDMLE
            - FE
            - HPJ
    Inputs:
        vT              vector, dimension of the time simulation
        vKappa          vector, kappa indicates N = T**(1/kappa)
        vGamma          vector, values to determine dRho0
        vTheta          vector, parameters used in simulation
        iM              integer, indicates the number of Monte Carlo simulations
        vInference      vector, holding the selected Inference options
        vBool           vector, configuration for simulation
        sInit           string, indicates initialization condition

    Return value:
        (Various results)
    """
    start_timer = timer()
    # Start multi-loops over the parameter combinations
    iPsi = len(vT) * len(vKappa) * len(vGamma)
    iEst = 4            # integer to track number of estimators
    mParam_hat = np.zeros((iM,iPsi,iEst))
    mCov_hat = np.zeros((iM,iPsi,iEst))
    mT_hat = np.zeros((iM,iPsi,iEst))
    mBias = np.zeros_like(mParam_hat)
    iTrack = 0          # indicator to track progress through Monte Carlo designs
    mRMSE = np.zeros_like(mParam_hat)
    mCI = np.zeros((iM,iPsi,iEst,2))
    mInf = np.zeros_like(mParam_hat)
    dCrit = st.chi2.ppf(0.95, 1)
    for gamma in vGamma:
        for kappa in vKappa:
            for iT in vT:
                iN = round(iT ** (1/kappa))
                dRho0 = np.exp(-iT**(-gamma))
                vTheta0 = np.hstack([dRho0, vTheta])
                # Simulate Dataset
                mY = Sim_AR_Panel(iM, iN, iT, vTheta0, vBool, sInit)
                for iSim in range(iM):
                    dTracker = iTrack * iM + iSim
                    dThreshold = np.ceil((iM * iPsi)*0.001)
                    if (dTracker %dThreshold == 0) & (dTracker != 0):
                        split_timer = timer() - start_timer
                        dProgress = np.round(dTracker/(iM * iPsi)*100,2)
                        dRemaining = 1/((dProgress)/100)-1
                        dEstTime = np.round(split_timer * dRemaining/60,2)
                        print(dProgress, "% ---", dEstTime, 'min remaining...', end='\r')
                    # Start Estimation
                    cARpanel_FD = MaxLikelihood(mY[:,:,iSim], bDisplay=False)
                    cARpanel_TMLE = Transformed_MaxLikelihood_Root(mY[:,:,iSim])
                    mParam_hat[iSim, iTrack,0] = cARpanel_FD.tx[0]
                    mParam_hat[iSim, iTrack,1] = cARpanel_TMLE.tx[0]
                    mCov_hat[iSim, iTrack, 0] = cARpanel_FD.tcovariancematrix[0,0]
                    mCov_hat[iSim, iTrack, 1] = cARpanel_TMLE.tcovariancematrix[0,0]
                    if vInference[0]:
                        (mCI[iSim, iTrack,0,0], mCI[iSim, iTrack,0,1]) = cARpanel_FD.tCI
                        (mCI[iSim, iTrack,1,0], mCI[iSim, iTrack,1,1]) = cARpanel_TMLE.tCI
                        mParam_hat[iSim, iTrack,2], (mCI[iSim, iTrack,2,0], mCI[iSim, iTrack,2,1]),mCov_hat[iSim, iTrack,2] = estimate_FE_WG(mY[:,:,iSim], vInference[1])
                        mParam_hat[iSim, iTrack,3], (mCI[iSim, iTrack,3,0], mCI[iSim, iTrack,3,1]),mCov_hat[iSim, iTrack,3] = estimate_HPJ(mY[:,:,iSim], vInference[2])
                    else:
                        mParam_hat[iSim, iTrack,2], mCov_hat[iSim, iTrack,2] = estimate_FE_WG(mY[:,:,iSim], 'Asymptotic')
                        mParam_hat[iSim, iTrack,3], mCov_hat[iSim, iTrack,3] = estimate_HPJ(mY[:,:,iSim], 'Asymptotic')
                    # Evaluate RMSE/Bias
                    for k in range(iEst):
                        mRMSE[iSim, iTrack, k] = RMSE_ARpanel(mY[:,:,iSim], mParam_hat[iSim, iTrack,k])
                        mBias[iSim, iTrack, k] = (mParam_hat[iSim, iTrack,k] - dRho0) * np.sqrt(iN * iT) / np.sqrt(mCov_hat[iSim, iTrack, k])
                    # Inference
                    if vInference[0]:
                        for k in range(0,2):
                            mInf[iSim, iTrack,k] = 1 - int(mCI[iSim, iTrack,k,0] <= dRho0) * int(dRho0 <= mCI[iSim, iTrack,k,1])
                        mT_hat[iSim, iTrack,2] = np.sqrt(iN*(iT-1)) * (mParam_hat[iSim, iTrack,2] - dRho0) / np.sqrt(mCov_hat[iSim, iTrack,2])
                        mT_hat[iSim, iTrack,3] = np.sqrt(iN*(iT-1)) * (mParam_hat[iSim, iTrack,3] - dRho0) / np.sqrt(mCov_hat[iSim, iTrack,3])
                        for k in range(2,iEst):
                            mInf[iSim, iTrack,k] = 1 - int(mCI[iSim, iTrack,k,0] <= mT_hat[iSim, iTrack,k]) * int(mT_hat[iSim, iTrack,k] <= mCI[iSim, iTrack,k,1])
                iTrack += 1
    print('')
    end_timer = timer()
    print('100%',np.round((end_timer-start_timer)/60,2), 'min elapsed.')
    return mParam_hat, mBias, mRMSE, mCI, mInf

###########################################################
### dY= emptyfunc(vX)
def Cross_sectional_Bootstrap(estimate_fun, mY, dRho, dSignif, *args):
    """
    Purpose:
        Implementation of the cross-sectional bootstrap for Panel data methods.
        Returns the CI of the test statistic

    Inputs:
        estimate_fun,           function, desired estimation function to implement the bootstrap procedure
        mY                      matrix of dep. variable (iN x iT)
        dRho                    double, original parameter estimate (This is always included in the bootstrap)
        dSignif,                 double, confidence level for the bootstrap
        iBoot                   integer, number of bootstrap draws
        *args                   Additional input arguments for "estimate_fun"

    Return value:
        vCI                     vector, (2 x 1) denoting the lower and upper bound of the confidence interval
    """
    iBoot = 199
    # Obtain sizes
    iN, iT = mY.shape
    vRho_boot = np.zeros(iBoot)
    vCov_boot = np.zeros(iBoot)
    vT_boot = np.zeros(iBoot+1)
    for Boot in range(iBoot):
        # Draw with replacement
        vBoot = np.random.choice(iN,iN,replace=True)
        # Select new sample
        mYBoot = mY[vBoot,:]
        vRho_boot[Boot], vCov_boot[Boot] = estimate_fun(mYBoot, 'Asymptotic')
    # Obtain bootstrapped test statistics
    vT_boot[1:] = np.sqrt(iN*iT) * (vRho_boot- dRho) / np.sqrt(vCov_boot)
    dLevel = round((1-dSignif)/2 * iBoot)
    vT_boot_sort = np.sort(vT_boot)
    dCiL = vT_boot_sort[dLevel-1]
    dCiU = vT_boot_sort[-dLevel]
    return (dCiL, dCiU)

###########################################################
### dY= emptyfunc(vX)
def Recursive_Bootstrap(estimate_fun, mY, dRho, dSignif, *args):
    """
    Purpose:
        Implementation of the Recursive bootstrap for Panel data methods.
        From Gonçalves and Kaffo (2025)

    Inputs:
        estimate_fun,           function, desired estimation function to implement the bootstrap procedure
        mY                      matrix of dep. variable (iN x iT)
        dRho                    double, original parameter estimate (This is always included in the bootstrap)
        dSignif,                 double, confidence level for the bootstrap
        iBoot                   integer, number of bootstrap draws
        *args                   Additional input arguments for "estimate_fun"

    Return value:
        vCI                     vector, (2 x 1) denoting the lower and upper bound of the confidence interval
    """
    iBoot = 199
    # Obtain sizes
    iN, iT = mY.shape
    vRho_boot = np.zeros(iBoot)
    vCov_boot = np.zeros(iBoot)
    vT_boot = np.zeros(iBoot+1)
    mPseudo = np.zeros((iN,iT,iBoot))
    # initialize
    mY0, mX0 = Lag_AR1(mY)
    vEta_hat = np.mean(mY0 - mX0 * dRho, axis=1).reshape(-1,1)     # Estimate fixed effects
    # mPseudo[:,0,:] = mY[:,0].reshape(-1,1)                        # Initialize Pseudo observations (stationary)
    mPseudo[:,0,:] = vEta_hat.reshape(-1,1) / (1-dRho)             # Initialize Pseudo observations (stationary)
    mE_hat = mY0 - vEta_hat - mX0 * dRho                           # Obtain estimated residuals
    for Boot in range(iBoot):
        mDraw_Rademacher = np.random.randint(0, 2, (iN,iT-1))
        vI = mDraw_Rademacher == 0
        mDraw_Rademacher[vI] = -1
        mE_pseudo = mE_hat * mDraw_Rademacher                       # Generate pseudo residuals
        # mE_pseudo = mE_hat * np.random.randn(iN,iT-1)               # Generate pseudo residuals
        for t in range(1,iT):
            # Generate pseudo observations
            mPseudo[:,t,Boot] = vEta_hat[:,0] + dRho * mPseudo[:,t-1,Boot] + mE_pseudo[:,t-1]
        # Obtain estimate
        vRho_boot[Boot], vCov_boot[Boot] = estimate_fun(mPseudo[:,:,Boot], 'Asymptotic', *args)
    # Obtain bootstrapped test statistics
    vT_boot[1:] = np.sqrt(iN*iT) * (vRho_boot- dRho) / np.sqrt(vCov_boot)
    dLevel = round((1-dSignif)/2 * (iBoot+1))
    vT_boot_sort = np.sort(vT_boot)
    dCiL = vT_boot_sort[dLevel-1]
    dCiU = vT_boot_sort[-dLevel]
    return (dCiL, dCiU)

###########################################################
### dY= emptyfunc(vX)
def Lag_AR1(mY):
    """
    Purpose:
        Construct the dependent variable and its regressor, for the AR(1) panel model.
        Y_t = rho * Y_{t-1} + epsilon_t

    Inputs:
        mY          matrix of dep. variable (iN x iT)

    Return value:
        mY_new      matrix of dep. variable (iN x (iT-1))
        mX_new      matrix of lagged dep. variable (iN x (iT-1))
    """
    iN, iT = mY.shape
    mY_new = mY[:,1:]
    mX_new = mY[:,:-1]
    return mY_new, mX_new

###########################################################
### dY= emptyfunc(vX)
def estimate_HPJ(mY, sSE_type = 'None'):
    """
    Purpose:
        Estimates parameters by applying the HPJ estimator: (only for AR panel)
        Y = X beta + epsilon
        (for panel data)
        Also returns standard FE estimate

    Inputs:
        mY          matrix of dep. variable (iN x iT)
        sSE_type    string to denote the type of standard errors
    Return value:
        vBeta   vector of estimated beta
    """
    # Init
    iN, iT = mY.shape
    iNT = iN * iT
    mYt, mXt = Lag_AR1(mY)
    iHPJ_0 = int(iT/2)
    # Estimate using FE-WG
    mYta, mXta = (mYt[:,:iHPJ_0], mXt[:,:iHPJ_0])
    mYtb, mXtb =(mYt[:,iHPJ_0:], mXt[:,iHPJ_0:])
    vBeta_hat = estimate_FE_WG((mYt, mXt), bHPJ=True)
    vBeta_hat0 = estimate_FE_WG((mYta, mXta), bHPJ=True)
    vBeta_hat1 = estimate_FE_WG((mYtb, mXtb), bHPJ=True)
    vBeta_HPJ = 2 * vBeta_hat - 0.5 * (vBeta_hat0 + vBeta_hat1)
    if sSE_type == 'CS-bootstrap':
        tCI_HPJ = Cross_sectional_Bootstrap(estimate_HPJ, mY, vBeta_HPJ, 0.95)
        mCov = estimate_HPJ_Cov(mYt, mXt, mYta, mXta, mYtb, mXtb, vBeta_HPJ)
        return vBeta_HPJ, tCI_HPJ, mCov
    if sSE_type == 'RD-bootstrap':
        tCI_HPJ = Recursive_Bootstrap(estimate_HPJ, mY, vBeta_HPJ, 0.95)
        mCov = estimate_HPJ_Cov(mYt, mXt, mYta, mXta, mYtb, mXtb, vBeta_HPJ)
        return vBeta_HPJ, tCI_HPJ, mCov
    if sSE_type == 'Asymptotic':
        # Estimate Cov and return
        mCov = estimate_HPJ_Cov(mYt, mXt, mYta, mXta, mYtb, mXtb, vBeta_HPJ)
        return vBeta_HPJ, mCov
    return vBeta_HPJ

###########################################################
### dY= emptyfunc(vX)
def estimate_HPJ_Cov(mYt, mXt,mYta, mXta, mYtb, mXtb, vBeta_HPJ):
    """
    Purpose:
        Function to estimate the covariance matrix of estimated parameters
        for the HPJ estimator.

    Inputs:

    Return value:

    """
    # Init
    iN, iT = mYt.shape
    mOmega = np.zeros((1,1))
    # Demean
    mYq = mYt - np.mean(mYt, axis=1).reshape(-1,1)
    mXq = mXt - np.mean(mXt, axis=1).reshape(-1,1)
    mXqa = mXta - np.mean(mXta, axis=1).reshape(-1,1)
    mXqb = mXtb - np.mean(mXtb, axis=1).reshape(-1,1)
    # Construct matrix with d
    mD_a = 2*mXq[:,:mXqa.shape[1]] - mXqa
    mD_b = 2*mXq[:,(mXqb.shape[1]):] - mXqb
    mD = np.zeros(mXt.shape)
    mD[:, :mXqa.shape[1]] = mD_a
    mD[:,(mXqb.shape[1]):] = mD_b
    # Obtain W_hat
    vXq = mXq.reshape(-1,1)
    mW_hati = np.linalg.inv(vXq.T @ vXq / (iN * iT))
    # Obtain residuals
    mE = mYq - mXq * vBeta_HPJ
    for i0 in range(iN):
        vD = mD[i0,:].reshape(-1,1)
        vY = mYq[i0,:].reshape(-1,1)
        vE = mE[i0,:].reshape(-1,1)
        mOmega = mOmega + vD.T @ vE @ vE.T @ vD
    mCov = mW_hati * mOmega * mW_hati / (iN * iT)
    return mCov[0,0]

###########################################################
### dY= emptyfunc(vX)
def estimate_FE_WG(mYt, sSE_type = 'None', bHPJ= False):
    """
    Purpose:
        Estimates parameters by applying the FE/WG estimator:
        Y = X beta + epsilon
        (for panel data)
        Computes the standard deviation and respective t-stat to test
        for beta == 0.

    Inputs:
        mYi          matrix of dep. variable (t indicates not transformed)
        sSE_type    string to denote the type of standard errors
    Return value:
        vBeta   vector of estimated \beta
    """
    # Initialize matrices
    if bHPJ == False:
        mY, mX = Lag_AR1(mYt)
        iN, iT = mY.shape
    else:
        mY, mX = mYt
    # Demean
    mYq = mY - np.mean(mY, axis=1).reshape(-1,1)
    mXq = mX - np.mean(mX, axis=1).reshape(-1,1)
    vYq = mYq.reshape(-1,1)
    vXq = mXq.reshape(-1,1)
    # Apply OLS to find parameter estimates
    vBeta_hat = np.sum(vXq * vYq) / np.sum(vXq * vXq)
    if sSE_type == 'CS-bootstrap':
        tCI_WG = Cross_sectional_Bootstrap(estimate_FE_WG, mY, vBeta_hat, 0.95)
        mCov_hat = estimate_FE_Cov(mYq, mXq, vBeta_hat)
        return vBeta_hat, tCI_WG, mCov_hat
    if sSE_type == 'RD-bootstrap':
        tCI_WG = Recursive_Bootstrap(estimate_FE_WG, mY, vBeta_hat, 0.95)
        mCov_hat = estimate_FE_Cov(mYq, mXq, vBeta_hat)
        return vBeta_hat, tCI_WG, mCov_hat
    if (sSE_type == 'cluster') or (sSE_type == 'Asymptotic'):
        mCov_hat = estimate_FE_Cov(mYq, mXq, vBeta_hat)
        if sSE_type == 'Asymptotic':
            return vBeta_hat, mCov_hat
        # Using covariance matrix of parameter estimates, obtain SE
        vSigma_hat = np.sqrt(np.diag(mCov_hat))
        tCI_WG = (vBeta_hat - 1.96 * vSigma_hat[0] / np.sqrt(iN * iT), vBeta_hat + 1.96 * vSigma_hat[0] / np.sqrt(iN * iT))
        return vBeta_hat, tCI_WG
    return vBeta_hat

###########################################################
### dY= emptyfunc(vX)
def estimate_BG(mY):
    """
    Purpose:
        Estimates parameters by applying the Between-Group estimator:
        Y = X beta + epsilon
        (for panel data)


    Inputs:
        mY              matrix of dep. variable

    Return value:
        vBeta   vector of estimated \beta
    """
    # Create functional matrices
    iN, iT = mY.shape
    mYt, mYLt = Lag_AR1(mY)
    mYt_mean = np.mean(mYt, axis=1).reshape(-1,1)
    mYLt_mean = np.mean(mYLt, axis=1).reshape(-1,1)
    mYdd = mYt_mean - mY[:,0].reshape(-1,1)
    mYddL = mYLt_mean - mY[:,0].reshape(-1,1)
    dRho = np.sum(mYdd * mYddL) / np.sum(mYddL * mYddL)
    return dRho

###########################################################
### dY= emptyfunc(vX)
def estimate_FE_Cov(mYq, mXq, vBeta_hat):
    """
    Purpose:
        Function to estimate the covariance matrix of estimated parameters
        for the FE estimator.
        Estimated the Cluster-Robust-Covariance matrix

    Inputs:

    Return value:

    """
    # Cluster residuals across individuals
    iK=1
    iN, iT = mYq.shape
    mOmega = np.zeros((iK,iK))
    vXq = mXq.reshape(-1,1)
    mXi = np.linalg.inv(vXq.T @ vXq / (iN * iT))
    mE = mYq - mXq * vBeta_hat
    vEq = mE.reshape(-1,1)
    for i0 in range(iN):
        vX = mXq[i0,:].reshape(-1,1)
        vY = mYq[i0,:].reshape(-1,1)
        vE = mE[i0,:].reshape(-1,1)
        mOmega = mOmega + vX.T @ vE @ vE.T @ vX
    mCov_hat = mXi @ mOmega @ mXi / (iN * iT)
    return mCov_hat[0,0]

###########################################################
### dY= emptyfunc(vX)
def FDMLE_backout(mY, dRho):
    """
    Purpose:
        Fuction to back out the parameters given the value of dRho.
        The underlying model is the AR(1) Panel data model, under the
        FD Framework.
        This implies that given a value of dRho, one can recover the respective
        values of dSigma2

    Inputs:
        mY          matrix of dep. variable (iN x iT)

    Return value:
        mY_new      matrix of dep. variable (iN x (iT-1))
        mX_new      matrix of lagged dep. variable (iN x (iT-1))
    """
    # Unpack vParams
    iN, iT = mY.shape
    mDn = First_diff_matrix(1,iT)
    mV = np.zeros((iT,iT))
    for i in range(iT):
        for j in range(iT):
            mV[i,j] = dRho**(abs(i-j)) / (1-dRho**2)
    mDVD = mDn @ mV @ mDn.T
    mDVDi = np.linalg.inv(mDVD)
    vQ = np.zeros(iN)
    for i in range(iN):
        vQ[i] = ((mDn @ mY[i,:]).T @ mDVDi @ (mDn @ mY[i,:]))
    # Estimate dSigma2
    dSigma2 = 1/ (iN * (iT-1)) * np.sum(vQ)
    return dSigma2

###########################################################
###
def CompLikelihood_FD(mY, mDn, vParam, iN, iT):
    """
    Purpose:
        Compute LogLikelihood vector for the FDMLE
    Inputs:
        vY              ((iN x iT) x 1)     Vector containing the data
        mDn             matrix, first-difference format for individual n
        vParam          Vector, containing the (transformed) parameters of interest.
        iN              integer, number of cross-sectional units
        iT              integer, number of time-series units
    Returns:
        vLikelihoodValues       vector, containing the individual-average log-likelihood values
    """
    # Unpack vParams
    dRho = vParam[0]
    mV = np.zeros((iT,iT))
    for i in range(iT):
        for j in range(iT):
            mV[i,j] = dRho**(abs(i-j)) / (1-dRho**2)
    mDVD = mDn @ mV @ mDn.T
    mDVDi = np.linalg.inv(mDVD)
    mDVDd = np.linalg.det(mDVD)
    vQ = np.zeros(iN)
    for i in range(iN):
        vQ[i] = ((mDn @ mY[i,:]).T @ mDVDi @ (mDn @ mY[i,:]))
    # Estimate dSigma2
    dSigma2 = 1/ (iN * (iT-1)) * np.sum(vQ)
    # Add likelihood constants
    vLikelihoodValues = (-vQ/(2* dSigma2) - 0.5 * (iT-1) * np.log(2 * np.pi) - 0.5 * (iT-1) * np.log(dSigma2) - 0.5 * np.log(mDVDd))/iT
    return vLikelihoodValues

###########################################################
###
def MaxLikelihood(mY, bDisplay=True, bJac = True):
    """
    Purpose:
        Maximum likelihood estimator of the given model class.
        Information is stored in cMaximumlikelihood class.
        In this implementation: The parameters are given by:
            dRho | vBeta | dSigma2
        Note: This is the maximum likelihood estimator used for the FDML in thesis
    Inputs:
        mY              (iN x iT)     Vector containing the data
        bDisplay        bool, indicate whether to indicate intermediate results
        bJac            bool, indicate whether analytical Jacobian should be used
    Returns:
        cReturnValue    Class object containing all information about the estimation
    """

    ###########################################################
    ###
    def Objective(vTheta, bForAll=False):
        """
        Purpose:
            Local objective function for maximum likelihood estimation.
            Parameters MUST be given in the correct order:
                dRho | dSigma2
        Inputs:
            vTheta      Vector, containing the (transformed) parameters of interest.
            bForAllT    Boolean, True if the vector of likelihoods must be return.      False for average negative log likelihood.
        Returns:
            dObjValue   Vector or Double, containin the vector of loglikelihood values or the average negative log likelihood.
        """
        # initialize the parameter values
        vParam = ParameterTransform(vTheta)
        # Compute likelihood value here
        vLikelihoodValues = CompLikelihood_FD(mY, mDn, vParam, iN, iT)
        if bDisplay:
            # Give sign of life
            print(".",end="")
        if (bForAll == True):
            # Return loglikelihood vector for numerical derivatives
            dObjValue = vLikelihoodValues.reshape(-1,)
        else:
            # Return negative mean of LL for optimizaiton (standard)
            dObjValue = -np.mean(vLikelihoodValues)
            vScore = Analytical_Score(vTheta)
        if bJac:
            return dObjValue, vScore
        else:
            return dObjValue
    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def ComputeCovarianceMatrix(vTheta, bInfEq=MASTER_bInfEq):
        """
        Purpose:
            Local covariance matrix function to compute the covariance matrix for the univariate garch model
            Parameters are expected to be given in the same order:
        Inputs:
            vTheta      Vector, containing the parameters of interest.
            bInfEq      Boolean, use the information matrix equality
        Returns:
            mCov        Matrix, iK x iK containing the covariance matrix of the estimated parameters
        """
        # compute the inverse hessian of the average log likelihood
        mH= hessian_2sided(Objective, vTheta)
        # print(mH)
        mCov = np.linalg.inv(mH)
        mCov = (mCov +  mCov.T)/2       #  Force to be symmetric
        if bInfEq:
            return mCov
        # compute the outer product of gradients of the average log likelihood
        mG = jacobian_2sided(Objective, vTheta, True)
        mG = mG.T @ mG / mG.shape[0]
        mCov = mCov @ mG @ mCov
        return mCov

    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def ComputeCovarianceMatrix_analytic(vTheta, bInfEq=MASTER_bInfEq):
        """
        Purpose:
            Local covariance matrix function to compute the covariance matrix for the AR(1) panel model
        Inputs:
            vTheta      Vector, containing the parameters of interest.
            bInfEq      Boolean, apply the information matrix inequality or not
        Returns:
            mCov        Matrix, iK x iK containing the covariance matrix of the estimated parameters
        """
        # compute the inverse hessian of the average log likelihood
        mHessian = Analytical_Hessian(vTheta)
        mHinv = np.linalg.inv(mHessian)
        if bInfEq:
            return mHinv
        mGrad = Analytical_Score(vTheta, True)
        mG = mGrad.T @ mGrad / (iT*iN)
        mCov = mHinv @ mG @ mHinv
        return mCov

    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def Analytical_Score(vTheta, bGrad=False):
        """
        Purpose:
            Local function to obtain the analytical score of the AR(1) panel model.
            Analytical score is used by BFGS minimize as the jacobian of the optimization schemes.

        Inputs:
            vTheta      Vector, containing the (transformed) parameters of interest.
            bGrad       Boolean, True if the Gradient (per cross-section) should be returned.

        Returns:
            vScore      Vector, (iK,) holding the score of the parameters
        """
        # initialize the parameter values
        iN, iT = mY.shape
        vParam = ParameterTransform(vTheta)
        dRho = vParam[0]
        dRho_tilde = vTheta[0]
        vScore = np.zeros((2,))
        mGrad = np.zeros((iN,2))
        # Score for dRho
        vY0 = np.zeros(iN)
        vY1 = np.zeros(iN)
        vI = np.zeros(iN)
        dA0 = (1-dRho) / ((1-dRho)*iT + 2*dRho)
        dA1 = 1/(((1-dRho)*iT + 2*dRho)**2)
        dL0 = iN * (iT-1) / ((1+dRho)*((1-dRho)*iT + 2*dRho))
        # Score for dSigma2
        mV = np.zeros((iT,iT))
        for i in range(iT):
            for j in range(iT):
                mV[i,j] = dRho**(abs(i-j)) / (1-dRho**2)
        mDVD = mDn @ mV @ mDn.T
        mDVDi = np.linalg.inv(mDVD)
        vQ = np.zeros(iN)
        for i in range(iN):
            vQ[i] = ((mDn @ mY[i,:]).T @ mDVDi @ (mDn @ mY[i,:]))
        # Estimate dSigma2
        dSigma2 = 1/ (iN * (iT-1)) * np.sum(vQ)
        vScore[1] = -iN * (iT-1) / (dSigma2*2) + np.sum(vQ) / (dSigma2**2 *2)
        mGrad[:,1] = -1 * (iT-1) / (dSigma2*2) + vQ / (dSigma2**2 *2)
        # Score for dRho
        for i in range(iN):
            vY0[i] = mY[i,0]**2 + mY[i,-1]**2 + 2 * mY[i,0] * mY[i,-1] + 2*(1-dRho) * (mY[i,0] + mY[i,-1]) * np.sum(mY[i,1:(iT-1)]) + (1-dRho)**2 * np.sum(mY[i,1:(iT-1)]) **2
            vY1[i] = (mY[i,0] + mY[i,-1]) * np.sum(mY[i,1:(iT-1)]) + (1-dRho) * np.sum(mY[i,1:(iT-1)]) **2
            vI[i] = dRho * np.sum(mY[i,1:(iT-1)]**2) - np.sum(mY[i,1:(iT)] * mY[i,:(iT-1)])
        vScore[0] = dL0 - np.sum(vI + dA1 * vY0 + dA0 * vY1) / dSigma2
        mGrad[:,0] = dL0 / iN - (vI + dA1 * vY0 + dA0 * vY1) / dSigma2
        # Adjust score for transformation and divide by (iT * iN)
        dPartialRho = 2*(iT+1)/(iT-1) * (np.exp(-dRho_tilde)/(1+np.exp(-dRho_tilde))**2)
        vScore_tf = np.zeros_like(vScore)
        mGrad_tf = np.zeros_like(mGrad)
        vScore_tf[0] = vScore[0] / (iT * iN) * dPartialRho
        # vScore_tf[1] = vScore[1] / (iT * iN)
        mGrad_tf[:,0] = mGrad[:,0] * dPartialRho
        mGrad_tf[:,1] = mGrad[:,1]
        if bGrad:
            return -mGrad_tf
        return -vScore_tf[0]

    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def Analytical_Hessian(vTheta):
        """
        Purpose:
            Local function to obtain the analytical Hessian of the AR(1) panel model.
            The analytical Hessian is used for the Sandwhich covariance estimator
            of the parameters.

        Inputs:
            vTheta      Vector, containing the (transformed) parameters of interest.
        Returns:
            mHessian    Matrix, (iK,iK) holding the Hessian of the parameters
        """
        # initialize the parameter values
        iN, iT = mY.shape
        vParam = ParameterTransform(vTheta)
        dRho = vParam[0]
        dRho_tilde = vTheta[0]
        vScore = np.zeros((2,))
        mHessian = np.zeros((2, 2))
        # Hessian for dSigma2, dSigma2
        mV = np.zeros((iT,iT))
        for i in range(iT):
            for j in range(iT):
                mV[i,j] = dRho**(abs(i-j)) / (1-dRho**2)
        mDVD = mDn @ mV @ mDn.T
        mDVDi = np.linalg.inv(mDVD)
        vQ = np.zeros(iN)
        for i in range(iN):
            vQ[i] = ((mDn @ mY[i,:]).T @ mDVDi @ (mDn @ mY[i,:]))
        # Estimate dSigma2
        dSigma2 = 1/ (iN * (iT-1)) * np.sum(vQ)
        mHessian[1,1] = iN * (iT-1) / (dSigma2**2*2) - np.sum(vQ) / (dSigma2**3)
        vScore[1] = -iN * (iT-1) / (dSigma2*2) + np.sum(vQ) / (dSigma2**2 *2)
        # Hessian for dRho, dRho
        vY0 = np.zeros(iN)
        vY1 = np.zeros(iN)
        vI = np.zeros(iN)
        vI0 = np.zeros(iN)
        vI1 = np.zeros(iN)
        dA0 = (1-dRho) / ((1-dRho)*iT + 2*dRho)
        dA1 = 1/((1-dRho)*iT + 2*dRho)**2
        dA2 = (iT-2)/((1-dRho)*iT + 2*dRho)**(3)
        dL0 = iN * (iT-1) / ((1+dRho)*((1-dRho)*iT + 2*dRho))
        dL1 = 2 * iN * (iT-1) * ( dRho * (iT-2)-1) / ((1+dRho)**2*((1-dRho)*iT + 2*dRho)**2)
        for i in range(iN):
            vY0[i] = mY[i,0]**2 + mY[i,-1]**2 + 2 * mY[i,0] * mY[i,-1] + 2*(1-dRho) * (mY[i,0] + mY[i,-1]) * np.sum(mY[i,1:(iT-1)]) + (1-dRho)**2 * np.sum(mY[i,1:(iT-1)]) **2
            vY1[i] = (mY[i,0] + mY[i,-1]) * np.sum(mY[i,1:(iT-1)]) + (1-dRho) * np.sum(mY[i,1:(iT-1)]) **2
            vI[i] = dRho * np.sum(mY[i,1:(iT-1)]**2) - np.sum(mY[i,1:(iT)] * mY[i,:(iT-1)])
            vI0[i] = np.sum(mY[i,1:(iT-1)]) **2
            vI1[i] = np.sum(mY[i,1:(iT-1)] **2)
        mHessian[0,0] = dL1 - np.sum(4 * dA2 * vY0 - 8 * dA1 * vY1 - 2 * dA0 * vI0 + 2 * vI1) / (dSigma2*2)
        vScore[0] = dL0 - np.sum(vI + dA1 * vY0 + dA0 * vY1) / dSigma2
        # Hessian for dSigma2, dRho
        mHessian[[0,1],[1,0]] =  np.sum(vI + dA1 * vY0 + dA0 * vY1) / dSigma2**2
        # Adjust Hessian for transformation and divide by (iT * iN)
        mHessian_tf = np.zeros_like(mHessian)
        dPartialRho = 2*(iT+1)/(iT-1) * (np.exp(-dRho_tilde)/(1+np.exp(-dRho_tilde))**2)
        dPartialRho2 = 2*(iT+1)/(iT-1) * (2* np.exp(-2*dRho_tilde) / (1+np.exp(-dRho_tilde))**3 - np.exp(-dRho_tilde) / (1+np.exp(-dRho_tilde))**2)
        mHessian_tf[0,0] = mHessian[0,0] / (iT * iN) * dPartialRho**2 + vScore[0] / (iT * iN) * dPartialRho2
        # mHessian_tf[0,0] = mHessian[0,0] / (iT * iN)
        mHessian_tf[[0,1],[1,0]] = mHessian[0,1] / (iT * iN) * dPartialRho
        # mHessian_tf[[0,1],[1,0]] = mHessian[0,1] / (iT * iN)
        mHessian_tf[1,1] = mHessian[1,1] / (iT * iN)
        return -mHessian_tf

    ###############################################
    ### main()
    def ParameterTransform(vTheta):
        """
        Purpose:
            Obtain true parameters by applying parameter transformation.
            Will produce parameter restrictions:
                -(iT+1)/(iT-1) < dBeta < +(iT+1)/(iT-1)
                dSigma2 > 0
            Parameters are expected to be given in the same order:

        Inputs:
            vTheta                  Vector, containing the transformed parameters of interest
        Returns:
            vParam                  Vector, containing the true parameters.
        """
        vParam = np.copy(vTheta)
        vParam[0] = (2/(1+np.exp(-vTheta[0])) - 1)*(iT+1)/(iT-1)
        return vParam
    # initialize starting values and return value
    cReturnValue = cMaximizeLikelihood()
    iN, iT = mY.shape
    mDn = First_diff_matrix(1,iT)
    vY = mY.reshape(-1,1)
    # Add dSigma2 as parameter
    vTheta = np.ones(1)
    iNT = vY.shape[0]
    dgtol = 1e-4
    # Select reasonable starting values | Use estimate of FE
    dFE = estimate_FE_WG(mY)
    dFrac = (iT-1) / (iT + 1)
    vTheta[0] = -np.log(2/(dFE * dFrac + 1)-1)        # Transformed starting value
    cReturnValue.x0 = vTheta
    cReturnValue.tx0 = ParameterTransform(vTheta)
    # Start the optimization
    tSol = opt.minimize(Objective, vTheta, method='BFGS', jac=bJac,
        options={'disp': bDisplay, 'maxiter':1000, 'gtol': dgtol})
    cReturnValue.success = tSol['success']
    # check for success and store results
    if (tSol['success'] != True):
        print("*** no true convergence: ",tSol['message'])
    dRho_tilde = tSol['x'][0]
    dRho_hat = ParameterTransform([dRho_tilde])[0]
    dSigma2_hat = FDMLE_backout(mY, dRho_hat)
    cReturnValue.x = [dRho_tilde, dSigma2_hat]
    cReturnValue.tx = np.round(ParameterTransform(cReturnValue.x),4)
    cReturnValue.likelihoodvalue = -iT * iN * tSol['fun']
    cReturnValue.covariancematrix = ComputeCovarianceMatrix_analytic(cReturnValue.x)
    dPartialRho = 2*(iT+1)/(iT-1) * (np.exp(-cReturnValue.x[0])/(1+np.exp(-cReturnValue.x[0]))**2)
    mJt = np.diag([dPartialRho, 1])
    cReturnValue.tcovariancematrix = mJt @ cReturnValue.covariancematrix @ mJt.T
    cReturnValue.tstandarderrors = np.sqrt(np.diag(cReturnValue.tcovariancematrix)/iNT)
    cReturnValue.tCI = (cReturnValue.tx[0] - 1.96 * cReturnValue.tstandarderrors[0],cReturnValue.tx[0] + 1.96 * cReturnValue.tstandarderrors[0])
    return cReturnValue

###########################################################
### dY= emptyfunc(vX)
def TMLE_backout(mY, dRho):
    """
    Purpose:
        Fuction to back out the parameters given the value of dRho.
        The underlying model is the AR(1) Panel data model, under the
        TML Framework.
        This implies that given a value of dRho, one can recover the respective
        values of dSigma2 and dGamma2

    Inputs:
        mY          matrix of dep. variable (iN x iT)

    Return value:
        mY_new      matrix of dep. variable (iN x (iT-1))
        mX_new      matrix of lagged dep. variable (iN x (iT-1))
    """
    # Create functional matrices
    iN, iT = mY.shape
    mYt, mYLt = Lag_AR1(mY)
    mYt_mean = np.mean(mYt, axis=1).reshape(-1,1)
    mYLt_mean = np.mean(mYLt, axis=1).reshape(-1,1)
    mYq = mYt - mYt_mean
    mYqL = mYLt - mYLt_mean
    mYdd = mYt_mean - mY[:,0].reshape(-1,1)
    mYddL = mYLt_mean - mY[:,0].reshape(-1,1)
    mE = mYq - dRho * mYqL
    vEdd = (mYdd - dRho * mYddL).reshape(-1)
    # Back out parameters
    dSigma2 = 1/ (iN* (iT-1)) * np.sum(mE * mE)
    dGamma2 = iT / iN * np.sum(vEdd * vEdd)
    return dSigma2, dGamma2

###########################################################
### dY= emptyfunc(vX)
def TMLE_FD_backout(mY, dRho):
    """
    Purpose:
        Fuction to back out the parameters given the value of dRho.
        The underlying model is the AR(1) Panel data model, under the
        TML Framework, with the imposed parameter values as in FD.
        This implies that given a value of dRho, one can recover the respective
        values of dSigma2 and dGamma2

    Inputs:
        mY          matrix of dep. variable (iN x iT)

    Return value:
        mY_new      matrix of dep. variable (iN x (iT-1))
        mX_new      matrix of lagged dep. variable (iN x (iT-1))
    """
    # initialize the parameter values
    iN, iT = mY.shape
    mGrad = np.zeros((iN,1))
    # Create functional matrices
    mYt, mYLt = Lag_AR1(mY)
    mYt_mean = np.mean(mYt, axis=1).reshape(-1,1)
    mYLt_mean = np.mean(mYLt, axis=1).reshape(-1,1)
    mYq = mYt - mYt_mean
    mYqL = mYLt - mYLt_mean
    mYdd = mYt_mean - mY[:,0].reshape(-1,1)
    mYddL = mYLt_mean - mY[:,0].reshape(-1,1)
    mE = mYq - dRho * mYqL
    vEdd = (mYdd - dRho * mYddL).reshape(-1)
    # Back out parameters implied by dRho
    dFrac = 1/(1+ iT * ((1-dRho)/(1+dRho)))
    dSigma2 = 1/ (iN* iT) * np.sum(mE * mE) + 1/ iN * dFrac * np.sum(vEdd * vEdd)
    return dSigma2

###########################################################
###
def CompLikelihood_TMLE(mY, vParam):
    """
    Purpose:
        Compute LogLikelihood vector for the TMLE
    Inputs:
        mY              (iN x iT) matrix, containing the data
        vParam          Vector, containing the (transformed) parameters of interest.
        iN              integer, number of cross-sectional units
        iT              integer, number of time-series units
    Returns:
        vLikelihoodValues       vector, containing the individual-average log-likelihood values
    """
    # Unpack vParams
    dRho = vParam[0]
    # Create functional matrices
    iN, iT = mY.shape
    mYt, mYLt = Lag_AR1(mY)
    mYt_mean = np.mean(mYt, axis=1).reshape(-1,1)
    mYLt_mean = np.mean(mYLt, axis=1).reshape(-1,1)
    mYq = mYt - mYt_mean
    mYqL = mYLt - mYLt_mean
    mYdd = mYt_mean - mY[:,0].reshape(-1,1)
    mYddL = mYLt_mean - mY[:,0].reshape(-1,1)
    # Initialize likelihood and compute disturbances
    vLikelihoodValues = np.zeros(iN)
    mE = mYq - dRho * mYqL
    vEdd = (mYdd - dRho * mYddL).reshape(-1)
    dSigma2 = 1/ (iN * (iT-1)) * np.sum(mE * mE)
    dGamma2 = iT/ iN * np.sum(vEdd * vEdd)
    for i in range(iN):
        vLikelihoodValues[i] = -1/ (2*dSigma2) * np.sum(mE[i,:] * mE[i,:]) - iT/ (2*dGamma2) * vEdd[i] * vEdd[i]
    # Add likelihood constants
    vLikelihoodValues = (vLikelihoodValues - 0.5 * (iT-1) * np.log(2 * np.pi) - 0.5 * (iT-1) * np.log(dSigma2) - 0.5 * np.log(dGamma2)) / iT
     # Return likelihood contributions
    return vLikelihoodValues

###########################################################
###
def CompLikelihood_TMLE_FD(vParam):
    """
    Purpose:
        Compute LogLikelihood vector for the TMLE
    Inputs:
        mY              (iN x iT) matrix, containing the data
        vParam          Vector, containing the (transformed) parameters of interest.
        iN              integer, number of cross-sectional units
        iT              integer, number of time-series units
    Returns:
        vLikelihoodValues       vector, containing the individual-average log-likelihood values
    """
    # Unpack vParams
    dRho, dSigma2 = vParam
    # Create functional matrices
    iN, iT = mY.shape
    mYt, mYLt = Lag_AR1(mY)
    mYt_mean = np.mean(mYt, axis=1).reshape(-1,1)
    mYLt_mean = np.mean(mYLt, axis=1).reshape(-1,1)
    mYq = mYt - mYt_mean
    mYqL = mYLt - mYLt_mean
    mYdd = mYt_mean - mY[:,0].reshape(-1,1)
    mYddL = mYLt_mean - mY[:,0].reshape(-1,1)
    # Initialize likelihood and compute disturbances
    vLikelihoodValues = np.zeros(iN)
    mE = mYq - dRho * mYqL
    vEdd = (mYdd - dRho * mYddL).reshape(-1)
    dGamma2 = dSigma2 * (1 + iT * ((1-dRho)/(1+dRho)))
    for i in range(iN):
        vLikelihoodValues[i] = -1/ (2*dSigma2) * np.sum(mE[i,:] * mE[i,:]) - iT/ (2*dGamma2) * vEdd[i] * vEdd[i]
    # Add likelihood constants
    vLikelihoodValues = (vLikelihoodValues - 0.5 * (iT-1) * np.log(2 * np.pi) - 0.5 * (iT-1) * np.log(dSigma2) - 0.5 * np.log(dGamma2)) / iT
     # Return likelihood contributions
    return -np.mean(vLikelihoodValues)

###########################################################
###
def Transformed_MaxLikelihood(mY, bDisplay=True, bJac = True, bTMLE = True):
    """
    Purpose:
        Transformed Maximum likelihood estimator (TMLE) for AR(1)
        panel methods.
        Information is stored in cMaximumlikelihood class.
        In this implementation: The parameters are given by:
            dRho | dSigma2 | dGamma2
        Note: This implementation is not used to obtain results in the thesis.
        See below for the root finding alternative to this function.

    Inputs:
        mY              (iN x iT)     Vector containing the data
        bDisplay        bool, indicate whether to indicate intermediate results
        bJac            bool, indicate whether analytical Jacobian should be used
        bTMLE           bool, indicate to use the TML assumption
    Returns:
        cReturnValue    Class object containing all information about the estimation
    """

    ###########################################################
    ###
    def Objective(vTheta, bForAll=False):
        """
        Purpose:
            Local objective function for maximum likelihood estimation.
            Parameters MUST be given in the correct order:
                dRho | vBeta | dSigma2
        Inputs:
            vTheta      Vector, containing the (transformed) parameters of interest.
            bForAllT    Boolean, True if the vector of likelihoods must be return.      False for average negative log likelihood.
        Returns:
            dObjValue   Vector or Double, containin the vector of loglikelihood values or the average negative log likelihood.
        """
        # initialize the parameter values
        vParam = ParameterTransform(vTheta)
        # Compute likelihood value here
        vLikelihoodValues = CompLikelihood_TMLE(mY, vParam)
        if bDisplay:
            # Give sign of life
            print(".",end="")
        if (bForAll == True):
            # Return loglikelihood vector for numerical derivatives
            dObjValue = vLikelihoodValues.reshape(-1,)
        else:
            # Return negative mean of LL for optimizaiton (standard)
            dObjValue = -np.mean(vLikelihoodValues)
        if bJac:
            vScore = Analytical_Score(vTheta)
            return dObjValue, vScore
        else:
            return dObjValue
    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def ComputeCovarianceMatrix(vTheta, bInfEq=True):
        """
        Purpose:
            Local covariance matrix function to compute the covariance matrix for the univariate garch model
            Parameters are expected to be given in the same order:
        Inputs:
            vTheta      Vector, containing the parameters of interest.
            bInfEq      Boolean, use the information matrix equality
        Returns:
            mCov        Matrix, iK x iK containing the covariance matrix of the estimated parameters
        """
        # compute the inverse hessian of the average log likelihood
        mH= hessian_2sided(Objective, vTheta)
        # print(mH)
        mCov = np.linalg.inv(mH)
        mCov = (mCov +  mCov.T)/2       #  Force to be symmetric
        if bInfEq:
            return mCov
        # compute the outer product of gradients of the average log likelihood
        mG = jacobian_2sided(Objective, vTheta, True)
        mG = mG.T @ mG / mG.shape[0]
        mCov = mCov @ mG @ mCov
        return mCov

    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def ComputeCovarianceMatrix_analytic(vTheta, bInfEq=MASTER_bInfEq):
        """
        Purpose:
            Local covariance matrix function to compute the covariance matrix for the AR(1) panel model
        Inputs:
            vTheta      Vector, containing the parameters of interest.
            bInfEq      Boolean, apply the information matrix inequality or not
        Returns:
            mCov        Matrix, iK x iK containing the covariance matrix of the estimated parameters
        """
        # compute the inverse hessian of the average log likelihood
        mHessian = Analytical_Hessian(vTheta)
        mHinv = np.linalg.inv(mHessian)
        if bInfEq:
            return mHinv
        mGrad = Analytical_Score(vTheta, True)
        mG = mGrad.T @ mGrad / (iT*iN)
        mCov = mHinv @ mG @ mHinv
        return mCov

    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def Analytical_Score(vTheta, bGrad=False):
        """
        Purpose:
            Local function to obtain the analytical score of the AR(1) panel model.
            Analytical score is used by BFGS minimize as the jacobian of the optimization schemes.

        Inputs:
            vTheta      Vector, containing the (transformed) parameters of interest.
            bGrad       Boolean, True if the Gradient (per cross-section) should be returned.

        Returns:
            vScore      Vector, (iK,) holding the score of the parameters
        """
        # initialize the parameter values
        iN, iT = mY.shape
        vParam = ParameterTransform(vTheta)
        dRho = vParam[0]
        dRho_tilde = vTheta[0]
        vScore = np.zeros((3,))
        mGrad = np.zeros((iN,3))
        # Create functional matrices
        mYt, mYLt = Lag_AR1(mY)
        mYt_mean = np.mean(mYt, axis=1).reshape(-1,1)
        mYLt_mean = np.mean(mYLt, axis=1).reshape(-1,1)
        mYq = mYt - mYt_mean
        mYqL = mYLt - mYLt_mean
        mYdd = mYt_mean - mY[:,0].reshape(-1,1)
        mYddL = mYLt_mean - mY[:,0].reshape(-1,1)
        mE = mYq - dRho * mYqL
        vEdd = (mYdd - dRho * mYddL).reshape(-1)
        dSigma2 = 1/ (iN * (iT-1)) * np.sum(mE * mE)
        dGamma2 = iT/ iN * np.sum(vEdd * vEdd)
        # Gradient:
        for i in range(iN):
            mGrad[i,0] = 1/ dSigma2 * np.sum(mYqL[i,:] * (mYq[i,:] - dRho * mYqL[i,:])) + iT/ dGamma2 * mYddL[i,0] * (mYdd[i,0] - dRho * mYddL[i,0])
            mGrad[i,1] = - 1 * (iT-1) / (2 * dSigma2)  + np.sum(np.square(mYq[i,:] - dRho * mYqL[i,:])) / (2* dSigma2**2)
            mGrad[i,2] = - 1 / (2 * dGamma2) + iT * np.square(mYdd[i,0] - dRho * mYddL[i,0]) / (2* dGamma2**2)
        vScore[0] = np.sum(mGrad[:,0]) / (iN * iT)
        vScore[1] = np.sum(mGrad[:,1]) / (iN * iT)
        vScore[2] = np.sum(mGrad[:,2]) / (iN * iT)
        # Adjust score for parameter bounds
        vScore_tf = np.zeros_like(vScore)
        mGrad_tf = np.zeros_like(mGrad)
        dPartialRho = 2*(iT+1)/(iT-1) * (np.exp(-dRho_tilde)/(1+np.exp(-dRho_tilde))**2)
        vScore_tf[0] = vScore[0] * dPartialRho
        vScore_tf[1] = vScore[1]
        vScore_tf[2] = vScore[2]
        mGrad_tf[:,0] = mGrad[:,0] * dPartialRho
        mGrad_tf[:,1] = mGrad[:,1]
        mGrad_tf[:,2] = mGrad[:,2]
        if bGrad:
            return -mGrad_tf
        return -vScore_tf[0]

    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def Analytical_Hessian(vTheta):
        """
        Purpose:
            Local function to obtain the analytical Hessian of the AR(1) panel model.
            The analytical Hessian is used for the Sandwhich covariance estimator
            of the parameters.

        Inputs:
            vTheta      Vector, containing the (transformed) parameters of interest.
        Returns:
            mHessian    Matrix, (iK,iK) holding the Hessian of the parameters
        """
        # initialize the parameter values
        iN, iT = mY.shape
        vParam = ParameterTransform(vTheta)
        dRho = vParam[0]
        dRho_tilde = vTheta[0]
        vScore = np.zeros((3,))
        mGrad = np.zeros((iN,3))
        mHessian = np.zeros((3, 3))
        # Create functional matrices
        mYt, mYLt = Lag_AR1(mY)
        mYt_mean = np.mean(mYt, axis=1).reshape(-1,1)
        mYLt_mean = np.mean(mYLt, axis=1).reshape(-1,1)
        mYq = mYt - mYt_mean
        mYqL = mYLt - mYLt_mean
        mYdd = mYt_mean - mY[:,0].reshape(-1,1)
        mYddL = mYLt_mean - mY[:,0].reshape(-1,1)
        mE = mYq - dRho * mYqL
        vEdd = (mYdd - dRho * mYddL).reshape(-1)
        dSigma2 = 1/ (iN * (iT-1)) * np.sum(mE * mE)
        dGamma2 = iT/ iN * np.sum(vEdd * vEdd)
        # Gradient:
        for i in range(iN):
            mGrad[i,0] = 1/ dSigma2 * np.sum(mYqL[i,:] * (mYq[i,:] - dRho * mYqL[i,:])) + iT/ dGamma2 * mYddL[i,0] * (mYdd[i,0] - dRho * mYddL[i,0])
            mGrad[i,1] = - 1 * (iT-1) / (2 * dSigma2)  + np.sum(np.square(mYq[i,:] - dRho * mYqL[i,:])) / (2* dSigma2**2)
            mGrad[i,2] = - 1 / (2 * dGamma2) + iT * np.square(mYdd[i,0] - dRho * mYddL[i,0]) / (2* dGamma2**2)
        mGrad = mGrad.reshape(3,-1) / (iN * iT)
        vScore[0] = np.sum(mGrad[:,0])
        vScore[1] = np.sum(mGrad[:,1])
        vScore[2] = np.sum(mGrad[:,2])
        # Create Hessian
        mHessian[0,0] = - 1/ dSigma2 * np.sum(np.square(mYqL)) - iT / dGamma2 *  np.sum(np.square(mYddL))
        mHessian[[0,1], [1,0]] = - 1/ dSigma2**2 * np.sum(mYqL * (mYq - dRho * mYqL))
        mHessian[[0,2], [2,0]] = - 1/ dGamma2**2 * np.sum(mYddL * (mYdd - dRho * mYddL))
        mHessian[1,1] = iN * (iT-1) / (2 * dSigma2**2) - np.sum(np.square(mYq - dRho * mYqL)) / dSigma2**3
        mHessian[2,2] = iN / (2 * dGamma2**2) - iT * np.sum(np.square(mYdd - dRho * mYddL)) / dGamma2**3
        mHessian = mHessian / (iN * iT)
        # Adjust Hessian for transformation and divide by (iT * iN)
        mHessian_tf = np.zeros_like(mHessian)
        dPartialRho = 2*(iT+1)/(iT-1) * (np.exp(-dRho_tilde)/(1+np.exp(-dRho_tilde))**2)
        dPartialRho2 = 2*(iT+1)/(iT-1) * (2* np.exp(-2*dRho_tilde) / (1+np.exp(-dRho_tilde))**3 - np.exp(-dRho_tilde) / (1+np.exp(-dRho_tilde))**2)
        mHessian_tf[0,0] = mHessian[0,0] * dPartialRho**2 + vScore[0] * dPartialRho2
        mHessian_tf[1,1] = mHessian[1,1]
        mHessian_tf[2,2] = mHessian[2,2]
        mHessian_tf[[0,1],[1,0]] = mHessian[0,1] * dPartialRho
        mHessian_tf[[0,2],[2,0]] = mHessian[0,2] * dPartialRho
        return -mHessian_tf

    ###############################################
    ### main()
    def ParameterTransform(vTheta):
        """
        Purpose:
            Obtain true parameters by applying parameter transformation.
            Will produce parameter restrictions:
                -(iT+1)/(iT-1) < dBeta < +(iT+1)/(iT-1)
                dSigma2 > 0
            Parameters are expected to be given in the same order:

        Inputs:
            vTheta                  Vector, containing the transformed parameters of interest
        Returns:
            vParam                  Vector, containing the true parameters.
        """
        vParam = np.copy(vTheta)
        vParam[0] = (2/(1+np.exp(-vTheta[0])) - 1)*(iT+1)/(iT-1)
        return vParam
    # initialize starting values and return value
    cReturnValue = cMaximizeLikelihood()
    iN, iT = mY.shape
    vY = mY.reshape(-1,1)
    # Only dRho needs numerical optimization
    vTheta = np.ones(1)
    iNT = vY.shape[0]
    dgtol = 1e-4
    # Select reasonable starting values | Use FE as starting point
    dFE = estimate_FE_WG(mY)
    dFrac = (iT-1) / (iT + 1)
    vTheta[0] = -np.log(2/(dFE * dFrac + 1)-1)        # Transformed starting value
    cReturnValue.x0 = vTheta
    cReturnValue.tx0 = ParameterTransform(vTheta)
    # Start the optimization
    tSol = opt.minimize(Objective, vTheta, method='BFGS', jac=bJac,
        options={'disp': bDisplay, 'maxiter':1000, 'gtol': dgtol})
    cReturnValue.success = tSol['success']
    # check for success and store results
    if (tSol['success'] != True):
        print("*** no true convergence: ",tSol['message'])
    dRho_hat = tSol['x'][0]
    dSigma2_hat, dGamma2_hat = TMLE_backout(mY, dRho)
    cReturnValue.x = [dRho_hat, dSigma2_hat, dGamma2_hat]
    cReturnValue.tx = np.round(ParameterTransform(cReturnValue.x),4)
    cReturnValue.covariancematrix = ComputeCovarianceMatrix_analytic(cReturnValue.x)
    cReturnValue.likelihoodvalue = np.mean(CompLikelihood_TMLE(mY, cReturnValue.tx)) * iN * iT
    dPartialRho = 2*(iT+1)/(iT-1) * (np.exp(-cReturnValue.x[0])/(1+np.exp(-cReturnValue.x[0]))**2)
    mJt = np.diag([dPartialRho, 1, 1])
    cReturnValue.tcovariancematrix = mJt @ cReturnValue.covariancematrix @ mJt.T
    cReturnValue.tstandarderrors = np.sqrt(np.diag(cReturnValue.tcovariancematrix)/iNT)
    cReturnValue.tCI = (cReturnValue.tx[0] - 1.96 * cReturnValue.tstandarderrors[0],cReturnValue.tx[0] + 1.96 * cReturnValue.tstandarderrors[0])
    return cReturnValue

###########################################################
### dY= emptyfunc(vX)
def Transformed_MaxLikelihood_Root(mY):
    """
    Purpose:
        Transformed Maximum likelihood estimator (TMLE) for AR(1)
        panel methods.
        Estimation proceeds by root finding instead of minimization.
        In this implementation: The parameters are given by:
            dRho | dSigma2 | dGamma2
        Note: This is the maximum likelihood estimator used for the TML in thesis.
        Root finding implementation is much faster compared to numerical optimization.
    Inputs:
        mY              (iN x iT)     Vector containing the data

    Returns:
        cReturnValue    Class object containing all information about the estimation
    """
    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def Objective(dRho):
        """
        Purpose:
            Objective function for the root finding procedure.
            This is the Score with respect to dRho
            In this implementation: The parameters are given by:
                dRho | dSigma2 | dGamma2
        Inputs:
            mY              (iN x iT)     Vector containing the data

        Returns:
            cReturnValue    Class object containing all information about the estimation
        """
        # initialize the parameter values
        iN, iT = mY.shape
        mGrad = np.zeros((iN,1))
        # Create functional matrices
        mYt, mYLt = Lag_AR1(mY)
        mYt_mean = np.mean(mYt, axis=1).reshape(-1,1)
        mYLt_mean = np.mean(mYLt, axis=1).reshape(-1,1)
        mYq = mYt - mYt_mean
        mYqL = mYLt - mYLt_mean
        mYdd = mYt_mean - mY[:,0].reshape(-1,1)
        mYddL = mYLt_mean - mY[:,0].reshape(-1,1)
        mE = mYq - dRho * mYqL
        vEdd = (mYdd - dRho * mYddL).reshape(-1)
        # Back out parameters implied by dRho
        dSigma2 = 1/ (iN* (iT-1)) * np.sum(mE * mE)
        dGamma2 = iT / iN * np.sum(vEdd * vEdd)
        # Gradient:
        for i in range(iN):
            mGrad[i,0] = 1/ dSigma2 * np.sum(mYqL[i,:] * (mYq[i,:] - dRho * mYqL[i,:])) + iT/ dGamma2 * mYddL[i,0] * (mYdd[i,0] - dRho * mYddL[i,0])
        dScore = np.sum(mGrad) / (iN * iT)
        return dScore
    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def ComputeCovarianceMatrix_analytic(dRho, bInfEq=MASTER_bInfEq):
        """
        Purpose:
            Local covariance matrix function to compute the covariance matrix for the AR(1) panel model
        Inputs:
            vTheta      Vector, containing the parameters of interest.
            bInfEq      Boolean, apply the information matrix inequality or not
        Returns:
            mCov        Matrix, iK x iK containing the covariance matrix of the estimated parameters
        """
        # compute the inverse hessian of the average log likelihood
        mHessian = Analytical_Hessian(dRho)
        mHinv = np.linalg.inv(mHessian)
        if bInfEq:
            return mHinv
        mGrad = Analytical_Score(dRho, True)
        mG = mGrad.T @ mGrad / (iT*iN)
        mCov = mHinv @ mG @ mHinv
        return mCov

    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def Analytical_Score(dRho, bGrad=False):
        """
        Purpose:
            Local function to obtain the analytical score of the AR(1) panel model.
            Analytical score is used by BFGS minimize as the jacobian of the optimization schemes.

        Inputs:
            vTheta      Vector, containing the (transformed) parameters of interest.
            bGrad       Boolean, True if the Gradient (per cross-section) should be returned.

        Returns:
            vScore      Vector, (iK,) holding the score of the parameters
        """
        # initialize the parameter values
        iN, iT = mY.shape
        vScore = np.zeros((3,))
        mGrad = np.zeros((iN,3))
        # Create functional matrices
        mYt, mYLt = Lag_AR1(mY)
        mYt_mean = np.mean(mYt, axis=1).reshape(-1,1)
        mYLt_mean = np.mean(mYLt, axis=1).reshape(-1,1)
        mYq = mYt - mYt_mean
        mYqL = mYLt - mYLt_mean
        mYdd = mYt_mean - mY[:,0].reshape(-1,1)
        mYddL = mYLt_mean - mY[:,0].reshape(-1,1)
        mE = mYq - dRho * mYqL
        vEdd = (mYdd - dRho * mYddL).reshape(-1)
        # Back out parameters implied by dRho
        dSigma2 = 1/ (iN* (iT-1)) * np.sum(mE * mE)
        dGamma2 = iT / iN * np.sum(vEdd * vEdd)
        # Gradient:
        for i in range(iN):
            mGrad[i,0] = 1/ dSigma2 * np.sum(mYqL[i,:] * (mYq[i,:] - dRho * mYqL[i,:])) + iT/ dGamma2 * mYddL[i,0] * (mYdd[i,0] - dRho * mYddL[i,0])
            mGrad[i,1] = - 1 * (iT-1) / (2 * dSigma2)  + np.sum(np.square(mYq[i,:] - dRho * mYqL[i,:])) / (2* dSigma2**2)
            mGrad[i,2] = - 1 / (2 * dGamma2) + iT * np.square(mYdd[i,0] - dRho * mYddL[i,0]) / (2* dGamma2**2)
        vScore[0] = np.sum(mGrad[:,0]) / (iN * iT)
        vScore[1] = np.sum(mGrad[:,1]) / (iN * iT)
        vScore[2] = np.sum(mGrad[:,2]) / (iN * iT)
        if bGrad:
            return -mGrad
        return -vScore

    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def Analytical_Hessian(dRho):
        """
        Purpose:
            Local function to obtain the analytical Hessian of the AR(1) panel model.
            The analytical Hessian is used for the Sandwhich covariance estimator
            of the parameters.

        Inputs:
            vTheta      Vector, containing the (transformed) parameters of interest.
        Returns:
            mHessian    Matrix, (iK,iK) holding the Hessian of the parameters
        """
        # initialize the parameter values
        iN, iT = mY.shape
        vScore = np.zeros((3,))
        mGrad = np.zeros((iN,3))
        mHessian = np.zeros((3, 3))
        # Create functional matrices
        mYt, mYLt = Lag_AR1(mY)
        mYt_mean = np.mean(mYt, axis=1).reshape(-1,1)
        mYLt_mean = np.mean(mYLt, axis=1).reshape(-1,1)
        mYq = mYt - mYt_mean
        mYqL = mYLt - mYLt_mean
        mYdd = mYt_mean - mY[:,0].reshape(-1,1)
        mYddL = mYLt_mean - mY[:,0].reshape(-1,1)
        mE = mYq - dRho * mYqL
        vEdd = (mYdd - dRho * mYddL).reshape(-1)
        # Back out parameters implied by dRho
        dSigma2 = 1/ (iN* (iT-1)) * np.sum(mE * mE)
        dGamma2 = iT / iN * np.sum(vEdd * vEdd)
        # Create Hessian
        mHessian[0,0] = - 1/ dSigma2 * np.sum(np.square(mYqL)) - iT / dGamma2 *  np.sum(np.square(mYddL))
        mHessian[[0,1], [1,0]] = - 1/ dSigma2**2 * np.sum(mYqL * (mYq - dRho * mYqL))
        mHessian[[0,2], [2,0]] = - 1/ dGamma2**2 * np.sum(mYddL * (mYdd - dRho * mYddL))
        mHessian[1,1] = iN * (iT-1) / (2 * dSigma2**2) - np.sum(np.square(mYq - dRho * mYqL)) / dSigma2**3
        mHessian[2,2] = iN / (2 * dGamma2**2) - iT * np.sum(np.square(mYdd - dRho * mYddL)) / dGamma2**3
        mHessian = mHessian / (iN * iT)
        return -mHessian
    # Initialize class:
    cReturnValue = cMaximizeLikelihood
    iN, iT = mY.shape
    iNT = iN * iT
    # Obtain lower and upper bracket
    dRho_min = estimate_FE_WG(mY)
    dRho_max = estimate_BG(mY)
    # Use root finding algorithm
    tSol = opt.root_scalar(Objective, bracket=[dRho_min, dRho_max])
    # Check for convergence
    if tSol.converged == False:
        print("*** no true convergence: ",tSol.flag)
    ### Store results
    dSigma2, dGamma2 = TMLE_backout(mY, tSol.root)
    cReturnValue.tx = np.round([tSol.root, dSigma2, dGamma2], 4)
    cReturnValue.likelihoodvalue = np.mean(CompLikelihood_TMLE(mY, cReturnValue.tx)) * iN * iT
    cReturnValue.tcovariancematrix = ComputeCovarianceMatrix_analytic(tSol.root)
    cReturnValue.tstandarderrors = np.sqrt(np.diag(cReturnValue.tcovariancematrix)/iNT)
    cReturnValue.tCI = (cReturnValue.tx[0] - 1.96 * cReturnValue.tstandarderrors[0],cReturnValue.tx[0] + 1.96 * cReturnValue.tstandarderrors[0])
    return cReturnValue

###########################################################
### dY= emptyfunc(vX)
def Transformed_MaxLikelihood_Root_FD(mY):
    """
    Purpose:
        Transformed Maximum likelihood estimator (TMLE) for AR(1)
        panel methods, under the conditions as in the FDMLE method.
        Estimation proceeds by root finding instead of minimization.
        In this implementation: The parameters are given by:
            dRho | dSigma2
        Note: This function is used for testing purposes. Function estimates the
        FDML estimates as expected. Original function used to obtain results.
    Inputs:
        mY              (iN x iT)     Vector containing the data

    Returns:
        cReturnValue    Class object containing all information about the estimation
    """
    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def Objective(dRho):
        """
        Purpose:
            Objective function for the root finding procedure.
            This is the Score with respect to dRho
            In this implementation: The parameters are given by:
                dRho | dSigma2 | dGamma2
        Inputs:
            mY              (iN x iT)     Vector containing the data

        Returns:
            cReturnValue    Class object containing all information about the estimation
        """
        # initialize the parameter values
        iN, iT = mY.shape
        vScore = np.zeros((2,))
        mGrad = np.zeros((iN,2))
        # Create functional matrices
        mYt, mYLt = Lag_AR1(mY)
        mYt_mean = np.mean(mYt, axis=1).reshape(-1,1)
        mYLt_mean = np.mean(mYLt, axis=1).reshape(-1,1)
        mYq = mYt - mYt_mean
        mYqL = mYLt - mYLt_mean
        mYdd = mYt_mean - mY[:,0].reshape(-1,1)
        mYddL = mYLt_mean - mY[:,0].reshape(-1,1)
        mE = mYq - dRho * mYqL
        vEdd = (mYdd - dRho * mYddL).reshape(-1,1)
        # Back out parameters implied by dRho
        dFrac = 1/(1 + iT * ((1-dRho)/(1+dRho)))
        dSigma2 = 1/ (iN* iT) * np.sum(mE * mE) + 1/ iN * dFrac * np.sum(vEdd * vEdd)
        # Gradient:
        for i in range(iN):
            mGrad[i,0] = -iT / ((1+dRho)*((iT-1)*dRho - iT - 1)) + 1/ dSigma2 * np.sum(mYqL[i,:] * (mYq[i,:] - dRho * mYqL[i,:])) + dFrac * iT/ dSigma2 * (mYddL[i,0] * (mYdd[i,0] - dRho * mYddL[i,0])) - iT**2 / (dSigma2 *((iT-1)*dRho - iT - 1)**2) * vEdd[i,0] * vEdd[i,0]
            mGrad[i,1] = - 1 * iT / (2*dSigma2) + 1/(2* dSigma2**2) * (np.sum(mE[i,:] * mE[i,:]) + dFrac * iT * vEdd[i,0] * vEdd[i,0])
        vScore[0] = np.sum(mGrad[:,0]) / (iN * iT)
        vScore[1] = np.sum(mGrad[:,1]) / (iN * iT)
        return vScore[0]

    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def ComputeCovarianceMatrix_analytic(dRho, bInfEq=MASTER_bInfEq):
        """
        Purpose:
            Local covariance matrix function to compute the covariance matrix for the AR(1) panel model
        Inputs:
            vTheta      Vector, containing the parameters of interest.
            bInfEq      Boolean, apply the information matrix inequality or not
        Returns:
            mCov        Matrix, iK x iK containing the covariance matrix of the estimated parameters
        """
        # compute the inverse hessian of the average log likelihood
        mHessian = Analytical_Hessian_FD(dRho)
        mHinv = np.linalg.inv(mHessian)
        if bInfEq:
            return mHinv
        mGrad = Analytical_Score_FD(dRho, True)
        mG = mGrad.T @ mGrad / (iT*iN)
        mCov = mHinv @ mG @ mHinv
        return mCov

    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def Analytical_Score_FD(dRho, bGrad=False):
        """
        Purpose:
            Local function to obtain the analytical score of the AR(1) panel model.
            Analytical score is used by BFGS minimize as the jacobian of the optimization schemes.

        Inputs:
            vTheta      Vector, containing the (transformed) parameters of interest.
            bGrad       Boolean, True if the Gradient (per cross-section) should be returned.

        Returns:
            vScore      Vector, (iK,) holding the score of the parameters
        """
        # initialize the parameter values
        iN, iT = mY.shape
        vScore = np.zeros((2,))
        mGrad = np.zeros((iN,2))
        # Create functional matrices
        mYt, mYLt = Lag_AR1(mY)
        mYt_mean = np.mean(mYt, axis=1).reshape(-1,1)
        mYLt_mean = np.mean(mYLt, axis=1).reshape(-1,1)
        mYq = mYt - mYt_mean
        mYqL = mYLt - mYLt_mean
        mYdd = mYt_mean - mY[:,0].reshape(-1,1)
        mYddL = mYLt_mean - mY[:,0].reshape(-1,1)
        mE = mYq - dRho * mYqL
        vEdd = (mYdd - dRho * mYddL).reshape(-1,1)
        # Back out parameters implied by dRho
        dFrac = 1/(1 + iT * ((1-dRho)/(1+dRho)))
        dSigma2 = 1/ (iN* iT) * np.sum(mE * mE) + 1/ iN * dFrac * np.sum(vEdd * vEdd)
        # Gradient:
        for i in range(iN):
            # mGrad[i,0] = -iT / ((1+dRho)*((iT-1)*dRho - iT - 1)) + 1/ dSigma2 * np.sum(mYqL[i,:] * (mYq[i,:] - dRho * mYqL[i,:])) + dFrac**2 * iT/ dSigma2 * (mYddL[i,0] * (mYdd[i,0] - dRho * mYddL[i,0]) * (1+ iT * ((1-dRho)/(1+dRho))) - iT / (1+dRho)**2 * vEdd[i] * vEdd[i])
            mGrad[i,0] = -iT / ((1+dRho)*((iT-1)*dRho - iT - 1)) + 1/ dSigma2 * np.sum(mYqL[i,:] * (mYq[i,:] - dRho * mYqL[i,:])) + dFrac * iT/ dSigma2 * (mYddL[i,0] * (mYdd[i,0] - dRho * mYddL[i,0])) - iT**2 / (dSigma2 *((iT-1)*dRho - iT - 1)**2) * vEdd[i,0] * vEdd[i,0]
            mGrad[i,1] = - 1 * iT / (2*dSigma2) + 1/(2* dSigma2**2) * (np.sum(mE[i,:] * mE[i,:]) + dFrac * iT * vEdd[i,0] * vEdd[i,0])
        vScore[0] = np.sum(mGrad[:,0]) / (iN * iT)
        vScore[1] = np.sum(mGrad[:,1]) / (iN * iT)
        if bGrad:
            return -mGrad
        return -vScore

    ###########################################################
    ### cMaxLikelihood = MaxLikelihood(vData)
    def Analytical_Hessian_FD(dRho):
        """
        Purpose:
            Local function to obtain the analytical Hessian of the AR(1) panel model.
            The analytical Hessian is used for the Sandwhich covariance estimator
            of the parameters.

        Inputs:
            vTheta      Vector, containing the (transformed) parameters of interest.
        Returns:
            mHessian    Matrix, (iK,iK) holding the Hessian of the parameters
        """
        # initialize the parameter values
        iN, iT = mY.shape
        vScore = np.zeros((2,))
        mGrad = np.zeros((iN,2))
        mHessian = np.zeros((2, 2))
        # Create functional matrices
        mYt, mYLt = Lag_AR1(mY)
        mYt_mean = np.mean(mYt, axis=1).reshape(-1,1)
        mYLt_mean = np.mean(mYLt, axis=1).reshape(-1,1)
        mYq = mYt - mYt_mean
        mYqL = mYLt - mYLt_mean
        mYdd = mYt_mean - mY[:,0].reshape(-1,1)
        mYddL = mYLt_mean - mY[:,0].reshape(-1,1)
        mE = mYq - dRho * mYqL
        vEdd = (mYdd - dRho * mYddL).reshape(-1,1)
        # Back out parameters implied by dRho
        dFrac = 1/(1+ iT * ((1-dRho)/(1+dRho)))

        dSigma2 = 1/ (iN * iT) * np.sum(mE * mE) + 1/ iN * dFrac * np.sum(vEdd * vEdd)
        # Create Hessian
        dI0 = 2 * iN * iT * ((iT-1)*dRho - 1) / ((1+dRho)**2 * ((iT-1)*dRho - iT- 1)**2)
        dI1 = 2 * iT / ((1+dRho)**2)
        dI21 = 1 / (((iT-1) * dRho - iT - 1)**2)
        dI22 = (iT-1) / (((iT-1)*dRho - iT - 1)**3)
        dI3 = 1 / (((iT-1) * dRho - iT - 1)**2)
        mHessian[0,0] = dI0 - 1/ dSigma2 * np.sum(mYqL * mYqL) + iT / dSigma2 * dFrac**2 * ( dI1 * np.sum(mYddL * vEdd) - (1+ iT * ((1-dRho)/(1+dRho))) * np.sum(mYddL * mYddL)) + 2 * iT**2 / dSigma2 * (dI21 * np.sum(mYddL * vEdd) + dI22 * np.sum(vEdd * vEdd))

        mHessian[[0,1], [1,0]] = - 1/ (dSigma2**2) * ( np.sum(mYqL * mE) + dFrac * iT * np.sum(mYddL * vEdd) - iT**2 * dI3 * np.sum(vEdd * vEdd) )
        mHessian[1,1] = iN * iT / (2 * dSigma2**2) - (np.sum(mE * mE) + dFrac * iT * np.sum(vEdd * vEdd)) / (dSigma2**3)
        mHessian_tf = mHessian / (iN * iT)
        return -mHessian_tf

    # Initialize class:
    cReturnValue = cMaximizeLikelihood
    iN, iT = mY.shape
    iNT = iN * iT
    # Obtain lower and upper bracket
    dRho_min = estimate_FE_WG(mY)
    dRho_max = estimate_BG(mY)*(1+ iT * ((1-dRho)/(1+dRho)))
    # Use root finding algorithm
    tSol = opt.root_scalar(Objective, bracket=[dRho_min, dRho_max])
    # Check for convergence
    if tSol.converged == False:
        print("*** no true convergence: ",tSol.flag)
    ### Store results
    dSigma2 = TMLE_FD_backout(mY, tSol.root)
    cReturnValue.tx = np.round([tSol.root, dSigma2], 4)
    cReturnValue.tcovariancematrix = ComputeCovarianceMatrix_analytic(tSol.root)
    cReturnValue.tstandarderrors = np.sqrt(np.diag(cReturnValue.tcovariancematrix)/iNT)
    cReturnValue.tCI = (cReturnValue.tx[0] - 1.96 * cReturnValue.tstandarderrors[0],cReturnValue.tx[0] + 1.96 * cReturnValue.tstandarderrors[0])
    return cReturnValue

###########################################################
### dY= emptyfunc(vX)
def Sim_AR_Panel(iM,iN,iT,vTheta, vBool = [False, False], sInit = 'Cov-Stationary'):
    """
    Purpose:
        Monte Carlo simulation of a panel AR(1) model.
        vTheta contains the parameters to simulate the model.
        Can use the vBools vector to hand the simulation specification.
        dRho    =   vTheta[0]
        dSigma  =   vTheta[1]
        dNu     =   vTheta[2]
        dPhi    =   vTheta[3]
        dOmega  =   vTheta[-3]
        dBeta  =    vTheta[-2]
        dSigXi =   vTheta[-1]
        Configuratio of vBools:
        vBools[0]: Simulate from student-t with dNu DoF
        vBools[1]: Use stochastic volatility for volatility process

    Inputs:
        iM          integer, number of simulations
        iN          integer, number of cross sectional units
        iT          integer, number of time units
        vTheta      (iK) vector of parameters
        vBool      vector, to configure simulation
    Return value:
        mY          ((iN x iT) x iM)
    """
    # Unpack parameters
    dRho = vTheta[0]
    dSigma = vTheta[1]
    # Initialize values
    iT = iT + 1         # This makes sure that when taking first differences, we have T observations!
    mY = np.zeros((iN,iT,iM))
    mMu = np.random.randn(iN,iM)        # Simulate fixed effects from N(0,1)
    if vBool[0] == True:
        # Student-t case
        dNu = vTheta[2]
        mEpsilon_ns = np.random.standard_t(dNu, (iN,iT,iM))
    else:
        # Standard case
        mEpsilon_ns = np.random.randn(iN,iT,iM)
    if vBool[1] == True:
        # Stochastic volatility case (heteroskedasticity)
        dOmega  =   vTheta[-3]
        dBeta  =   vTheta[-2]
        dSigEta =   vTheta[-1]
        mXi = np.random.randn(iT,iM) * dSigmaXi * np.sqrt((1-dBeta**2))
        mSigma = np.zeros((iN, iT, iM))
        mSigma[:,0,:] = dOmega + mXi[0,:] / np.sqrt((1-dBeta**2))
        for t in range(1,iT):
            mSigma[:,t,:] = dOmega * (1-dBeta) + dBeta * mSigma[:,t-1,:] + mXi[t,:]
        mEpsilon = mEpsilon_ns * mSigma
    else:
        # Standard case (homoskedasticity)
        mEpsilon = mEpsilon_ns * dSigma
    # Initial value | psi_i
    if sInit == 'Cov-Stationary':
        ### Covariance stationary option
        mY[:,0,:] = mEpsilon[:,0,:]  / np.sqrt((1-dRho**2)) + mMu
    elif sInit == 'Zero-Residual':
        ### Non-covariance stationary option (set initial error at zero)
        mY[:,0,:] = mMu
    elif sInit == 'Increased-Cov':
        ### Non-covariance stationary option (increase initial variance for disturbance)
        mY[:,0,:] = np.sqrt(2) * mEpsilon[:,0,:]  / np.sqrt((1-dRho**2)) + mMu
    elif sInit == 'Phi-Residual':
        ### Non-covariance stationary option (Adjust magnitude of fixed effects)
        dPhi = vTheta[3]
        mY[:,0,:] = mEpsilon[:,0,:]  / np.sqrt((1-dRho**2)) + mMu * dPhi
    else:
        print('Incorrect initial condition specified ...')
        return None
    # Recursively construct mY
    for t in range(1,iT):
        mY[:,t,:] = dRho * mY[:,t-1,:] + mMu * (1-dRho) + mEpsilon[:,t,:]
    return mY

###########################################################
### dY= emptyfunc(vX)
def First_diff_matrix(iN,iT):
    """
    Purpose:
        Quick function to obtain the first-difference matrix for a given iN and iT.
        Supposes that the data is structured as follows:
        iN number of cross sectional units, for iT units of time sequentially placed in the vector.

    Inputs:
        iN          integer, number of cross-sectional units
        iT          integer, number of time-series units

    Return value:
        mD          ( (iN x iT) x (iN x (iT-1)) ) First-difference matrix
    """
    mD_init = np.zeros((iT-1,iT))
    for t in range(iT-1):
        mD_init[t,t+1] = 1
        mD_init[t,t] = -1
    mD = np.kron(np.eye(iN), mD_init)
    return mD

###########################################################
### dY= emptyfunc(vX)
def Plot_HistEstimators(mPlot, vT, vKappa, vGamma, vLegend, sName):
    """
    Purpose:
        Creates histogram of the provided matrix.
        iPsi number of simulation schemes
        TODO: Add dashed line in the center of the individual distributions

    Inputs:
        mPlot       matrix (iM, iPsi, iL)
        vKappa,     vector, indicating monte carlo design
        vT,         vector, indicating monte carlo design
        vLegend,    list, to add in the legend
        sName,      string, unique identifier for the plot

    Return value:
        vList       list, containing the monte carlo designs in strings
    """
    vList = []
    iTrack = 0
    for gamma in vGamma:
        fig, ax = plt.subplots(len(vKappa),len(vT),figsize=(25,25))
        for i, kappa in enumerate(vKappa):
            for j, iT in enumerate(vT):
                ax[i,j].set_xlim([-5,5])
                if 'CS' in sName:
                    ax[i,j].set_xlim([-9,9])
                ax[i,j].hist(mPlot[:,iTrack,:3], bins=100, density=True)
                iN = round(iT ** (1/kappa))
                dRho = np.exp(-iT**(-gamma))
                sTitle = "T:" +str(iT)+" N:" + str(iN) + r" $\rho_0$:" + str(round(dRho,2))
                vList.append(sTitle)
                ax[i,j].set_title(sTitle)
                ax[i,j].legend(vLegend, loc='upper left')
                if 'limiting' in sName:
                    ax[i,j].axvline(0, color='black', linestyle='dotted')
                    for iE in range(mPlot.shape[2]):
                        ax[i,j].axvline(np.mean(mPlot[:,iTrack, iE]), color=cols[iE], linestyle='dotted')
                iTrack += 1
        sName_fig = sName + str(gamma)
        fig.suptitle(r" g:" + str(gamma), fontsize=16)
        fig.savefig(r'TI\Thesis\figs\HistEstimators_'+sName_fig+'.jpeg')
    return vList

###########################################################
### dY= emptyfunc(vX)
def Monte_Carlo_Results(vT, vKappa, vGamma, vTheta, iM, vInference, vBool, sInit, vLegend, sName):
    """
    Purpose:
        Overarching function to obtain results from the Monte Carlo designs
        Simulation is handled by the Monte_Carlo_Panel function.
        vTheta contains all static parameters.

    Inputs:
        vT              vector, dimension of the time simulation
        vKappa          vector, kappa indicates N = T**(1/kappa)
        vGamma          vector, values to determine dRho0
        vTheta          vector, parameters used in simulation
        iM              integer, indicates the number of Monte Carlo simulations
        vInference      vector, to select whether inference is of interest
        vBool           vector, configuration for simulation
        vLegend,    list, to add in the legend
        sName,      string, unique identifier for the plot

    Return value:
        vList       list, containing the monte carlo designs in strings
    """
    if vBool[0] == True:
        print('Student-t residuals selected')
    else:
        print('Normal residuals selected')
    if vBool[1] == True:
        print('Heteroskedasticity selected')
    else:
        print('Homoskedasticity selected')
    print('Initial condition:', sInit)
    # Monte Carlo simulations
    mParam_hat, mBias, mRMSE, mCI, mInf = Monte_Carlo_Panel(vT, vKappa, vGamma, vTheta, iM, vInference, vBool, sInit)
    mAvg_RMSE = np.mean(mRMSE, axis=0)
    # Plot results
    # vList = Plot_HistEstimators(mParam_hat, vT, vKappa, vGamma, vLegend, sName + '_reg')
    vList = Plot_HistEstimators(mBias, vT, vKappa, vGamma, vLegend, sName + '_limiting')
    # Print RMSE
    df_res = pd.DataFrame(np.round(mAvg_RMSE,2), columns = vLegend, index= vList)
    df_res.to_latex(buf = r'TI\Thesis\dat\RMSE_'+sName+ '.txt' ,float_format="{:.2f}".format)
    # Check whether Inference results should be produced:
    if vInference[0]:
        dfInf = pd.DataFrame(np.round(np.mean(mInf,axis=0),2), columns = vLegend, index= vList)
        dfInf.to_latex(buf = r'TI\Thesis\dat\Inf_'+sName+ '.txt' ,float_format="{:.2f}".format)
    return mParam_hat, mBias, mRMSE, mCI, mInf

###########################################################
### main
def main():
    # Magic numbers
    MASTER_bInfEq = True # Master key to apply Information matrix inequality: Keep at True
    iM = 1000
    dGamma = 0.95
    dSigma2 = 1
    dSigma = dSigma2**(1/2)
    dNu = 5
    dOmega = 1
    dBeta = 0.6
    dSigmaXi = 1
    dPhi = 1                    # Governs the magnitude of the fixed effect component at the initial observation
    vTheta_sim = np.array([dSigma, dNu, dOmega, dPhi, dBeta, dSigmaXi])
    vT = [20,50,100]
    vKappa = [0.8, 1, 1.2]              # Remove 0.6 out of consideration for now, too much computing.
    vGamma =    [0.5]
    cols = ['C0', 'C1', 'C2', 'C3']
    vLegend = ['FDMLE', 'TMLE', 'FE', 'HPJ']

    ### Estimation only
    mParam_hat, mBias, mRMSE, mCI, mInf = Monte_Carlo_Results(vT, vKappa, vGamma, vTheta_sim, iM, [False, False, False], [False, False], 'Cov-Stationary', vLegend, 'CS-S')
    mParam_hat, mBias, mRMSE, mCI, mInf = Monte_Carlo_Results(vT, vKappa, vGamma, vTheta_sim, iM, [False, False, False], [False, False], 'Increased-Cov', vLegend, 'CS-S-IC')
    mParam_hat, mBias, mRMSE, mCI, mInf = Monte_Carlo_Results(vT, vKappa, vGamma, vTheta_sim, iM, [False, False, False], [False, False], 'Zero-Residual', vLegend, 'CS-S-ZR')

    mParam_hat, mBias, mRMSE, mCI, mInf = Monte_Carlo_Results(vT, vKappa, vGamma, vTheta_sim, iM, [False, False, False], [False, True], 'Cov-Stationary', vLegend, 'CS-SV')
    mParam_hat, mBias, mRMSE, mCI, mInf = Monte_Carlo_Results(vT, vKappa, vGamma, vTheta_sim, iM, [False, False, False], [False, True], 'Increased-Cov', vLegend, 'CS-SV-IC')
    mParam_hat, mBias, mRMSE, mCI, mInf = Monte_Carlo_Results(vT, vKappa, vGamma, vTheta_sim, iM, [False, False, False], [False, True], 'Zero-Residual', vLegend, 'CS-SV-ZR')
    ### Inference
    ## Standard normal | Homoskedastic case
    # Covariance stationary
    mParam_hat, mBias, mRMSE, mCI, mInf = Monte_Carlo_Results(vT, vKappa, vGamma, vTheta_sim, iM, [True, 'CS-bootstrap', 'CS-bootstrap'], [False, False], 'Cov-Stationary', vLegend, 'CS-S')
    # Non-Covariance stationarity | Increased initial variance:
    mParam_hat, mBias, mRMSE, mCI, mInf = Monte_Carlo_Results(vT, vKappa, vGamma, vTheta_sim, iM, [True, 'CS-bootstrap', 'CS-bootstrap'], [False, False], 'Increased-Cov', vLegend, 'CS-S-IC')
    # Non-Covariance stationarity | Zero-Residual
    mParam_hat, mBias, mRMSE, mCI, mInf = Monte_Carlo_Results(vT, vKappa, vGamma, vTheta_sim, iM, [True, 'CS-bootstrap', 'CS-bootstrap'], [False, False], 'Zero-Residual', vLegend, 'CS-S-ZR')
    ## Stoch-vol | Heteroskedastic case
    # Covariance stationary
    mParam_hat, mBias, mRMSE, mCI, mInf = Monte_Carlo_Results(vT, vKappa, vGamma, vTheta_sim, iM, [True, 'CS-bootstrap', 'CS-bootstrap'], [False, True], 'Cov-Stationary', vLegend, 'CS-SV')
    # Non-Covariance stationarity | Increased initial variance:
    mParam_hat, mBias, mRMSE, mCI, mInf = Monte_Carlo_Results(vT, vKappa, vGamma, vTheta_sim, iM, [True, 'CS-bootstrap', 'CS-bootstrap'], [False, True], 'Increased-Cov', vLegend, 'CS-SV-IC')
    # Non-Covariance stationarity | Zero-Residual
    mParam_hat, mBias, mRMSE, mCI, mInf = Monte_Carlo_Results(vT, vKappa, vGamma, vTheta_sim, iM, [True, 'CS-bootstrap', 'CS-bootstrap'], [False, True], 'Zero-Residual', vLegend, 'CS-SV-ZR')

###########################################################
### start main
if __name__ == "__main__":
    main()
