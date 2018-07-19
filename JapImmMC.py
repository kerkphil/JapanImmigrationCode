# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:30:29 2018

@author: kerkp
"""
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

name = 'UN_MC'

def mdefs(Xp, X, z, params):
    
    # unpack params
    [g, beta, delta, gamma, a, b, c, d, f, HSD, HUD, HSI, HUI, nu, mu, \
            ss, q, rho, sigma, beta, nx, ny, nz] = params
    
    [KSD, KUD, BSD, BUD] = X
    [KSDp, KUDp, BSDp, BUDp] = Xp
    
    K = KSD + KUD
    Kp = KSDp + KUDp
    B = BSD + BUD
    Bp = BSDp + BUDp
    NS = HSD + HSI
    NU = HUD + HUI
    W = (c*(f*NS)**((d-1)/d) + (1-c)*NU**((d-1)/d))**(d/(d-1))
    Y = (a*K**((b-1)/b) + (1-a)*(np.exp(z)*W)**((b-1)/b))**(b/(b-1))
    X = Bp*(1+g) - (1+ss)*B
    r = a*(Y/K)**(1/b)
    wS = f*(1-a)*(Y/W)**(1/b)*c*(W/NS)**(1/d)
    wU = (1-a)*(Y/W)**(1/b)*(1-c)*(W/NU)**(1/d)
    sS = ss - nu*(BSD/wS) - mu*((BSDp-BSD)**2/Y)
    sU = ss - nu*(BUD/wU) - mu*((BUDp-BUD)**2/Y)
    CSD = wS + ((1+r-delta)*KSD + (1+sS)*BSD - (1+g)*(KSDp + BSDp))/HSD
    CUD = wU + ((1+r-delta)*KUD + (1+sU)*BUD - (1+g)*(KUDp + BUDp))/HUD
    CSI = wS
    CUI = wU
    CSD = max(.0001, CSD)
    CUD = max(.0001, CUD)
    CSI = max(.0001, CSI)
    CUI = max(.0001, CUI)
    if gamma == 1.:
        USD = np.log(CSD)
        UUD = np.log(CUD)
        USI = np.log(CSI)
        UUI = np.log(CUI)
    else:
        USD = (CSD**(1-gamma)-1)/(1-gamma)
        UUD = (CUD**(1-gamma)-1)/(1-gamma)
        USI = (CSI**(1-gamma)-1)/(1-gamma)
        UUI = (CUI**(1-gamma)-1)/(1-gamma)    
    C = HSD*CSD + HUD*CUD + HSI*CSI + HUI*CUI
    I = Kp - (1-delta)*K
    
    return K, B, NS, NU, W, Y, sS, sU, X, r, wS, wU, CSD, CUD, CSI, CUI, USD, \
        UUD, USI, UUI, I, C


def mdyn(theta, params):
    # unpack theta
    [KSDpp, KUDpp, BSDpp, BUDpp, KSDp, KUDp, BSDp, BUDp, KSD, KUD, BSD, BUD, \
     zp, z] = theta
     
    [g, beta, delta, gamma, a, b, c, d, f, HSD, HUD, HSI, HUI, nu, mu, \
            ss, q, rho, sigma, beta, nx, ny, nz] = params
    
    K, B, NS, NU, W, Y, r, sS, sU, X, wS, wU, CSD, CUD, CSI, CUI, USD, UUD, \
        USI, UUI, I, C = mdefs(KSDp, KUDp, BSDp, BUDp, KSD, KUD, BSD, BUD, z, \
        params)
        
    Kp, Bp, NSp, NUp, Wp, Yp, sSp, sUp, Xp, rp, wSp, wUp, CSDp, CUDp, CSIp, \
        CUIp, USDp, UUDp, USIp, UUIp, Ip, Cp = mdefs(KSDpp, KUDpp, BSDpp, \
        BUDpp, KSDp, KUDp, BSDp, BUDp, z, params)
    # print('Cp:', CSDp, CUDp, CSIp, CUIp)
    
    ESD1 = CSD**(-gamma) - beta*(CSDp*(1+g))**(-gamma)*(1+sSp)
    EUD1 = CUD**(-gamma) - beta*(CUDp*(1+g))**(-gamma)*(1+sUp)
    ESD2 = CSD**(-gamma) - beta*(CSDp*(1+g))**(-gamma)*(1+rp-delta)
    EUD2 = CUD**(-gamma) - beta*(CUDp*(1+g))**(-gamma)*(1+rp-delta)
    
    Earray = np.array([ESD1, EUD1, ESD2, EUD2])
    # print(Earray)
    return Earray


def generateLIN(X, z, args):
    from LinApp_Sim import LinApp_Sim
    
    '''
    This function generates values of k next period and ell this period given
    values for k and z this period.
    
    Inputs
    X - X this period
    z - z this period
    args - lists of linear coeffiecients and the steady state values.
    
    Outputs
    Xp - X next period
    ell - ell this period
    '''
    
    # unpack args
    (coeffs, Xbar) = args
    (PP, QQ, UU, RR, SS, VV) = coeffs
    
    # inputs must be 1D numpy arrays and deviation from SS values
    Xtil = X - Xbar
    ztil = np.array([z])
    Xptil, temp = LinApp_Sim(Xtil, ztil, PP, QQ, UU, RR, SS, VV)
    Xp = Xptil + Xbar
    
    return Xp


def polsim(simargs):
    
    # unpack
    (initial, nobs, ts, funcname, args, argsn, params, paramsn) = simargs
    '''
    Generates a history of k & ell with a switch in regime in period ts
    
    This function reads values from the following pkl files:
        ILAfindss.pkl - steady state values and parameters
    
    Inputs
    -----------    
    initial: list of values for k & z (k0, z0) in the first period.
    nobs: number of periods to simulate.
    ts: period in which the shift occurs.
    args1: is a list of arguments needed by the solution method in baseline.
        For example, with linearization these are:
        coeffs1: list of (PP, QQ, UU, RR, SS, VV) under the baseline regime.
        XYbar1: numpy array of X & Y SS values under the baseline regime.
    args2: is a list of arguments needed by the solution method after change    
    params1: list of parameters under the baseline regime.
    params2: list of parameters under the new regime.
    
    Returns
    --------
    For the following variables x in (k, ell, z, Y, w, r, T, c, i, u):
        xhist: history of simultated values
        xfhist: history of one-period-ahed forecasts
        MsqEerr: root mean squared Euler errors
    '''
    # unpack params
    [g, beta, delta, gamma, a, b, c, d, f, HSD, HUD, HSI, HUI, nu, mu, \
            ss, q, rho, sigma, beta, nx, ny, nz] = params
    [g, beta, delta, gamma, a, b, c, d, f, HSDn, HUDn, HSIn, HUIn, nu, mu, ss, q, \
        rho, sigma, beta, nx, ny, nz] = paramsn
    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
        
    # preallocate histories
    XXhist = np.zeros((nobs+1, nx))
    Khist = np.zeros(nobs)
    Bhist = np.zeros(nobs)
    NShist = np.zeros(nobs)
    NUhist = np.zeros(nobs)
    Whist = np.zeros(nobs)
    Yhist = np.zeros(nobs)
    sShist = np.zeros(nobs)
    sUhist = np.zeros(nobs)
    Xhist = np.zeros(nobs)
    rhist = np.zeros(nobs)
    wShist = np.zeros(nobs)
    wUhist = np.zeros(nobs)
    CSDhist = np.zeros(nobs)
    CUDhist = np.zeros(nobs)
    CSIhist = np.zeros(nobs)
    CUIhist = np.zeros(nobs)
    USDhist = np.zeros(nobs)
    UUDhist = np.zeros(nobs)
    USIhist = np.zeros(nobs)
    UUIhist = np.zeros(nobs)
    Ihist = np.zeros(nobs)
    Chist = np.zeros(nobs)
    zhist = np.zeros(nobs)
    
    # set starting values
    (XX0, z0) = initial
    XXhist[0,:] = XX0
    zhist[0] = z0
    
    # generate history of random shocks
    for t in range(1, nobs):
        zhist[t] = rho*zhist[t-1] + sigma*np.random.normal(0., 1.)
        
    # generate histories for the first ts-1 periods
    for t in range(0, ts-1):
        XXhist[t+1,:] = funcname(XXhist[t,:], zhist[t], args)
        Khist[t], Bhist[t], NShist[t], NUhist[t], Whist[t], Yhist[t], \
        sShist[t], sUhist[t], Xhist[t], rhist[t], wShist[t], wUhist[t], \
        CSDhist[t], CUDhist[t], CSIhist[t], CUIhist[t], USDhist[t], \
        UUDhist[t], USIhist[t], UUIhist[t], Ihist[t], Chist[t] \
        = mdefs(XXhist[t+1,:], XXhist[t,:], zhist[t], params)

    # generate histories for the remaning periods        
    for t in range(ts-1, nobs):
        XXhist[t+1,:] = funcname(XXhist[t,:], zhist[t], argsn)
        Khist[t], Bhist[t], NShist[t], NUhist[t], Whist[t], Yhist[t], \
        sShist[t], sUhist[t], Xhist[t], rhist[t], wShist[t], wUhist[t], \
        CSDhist[t], CUDhist[t], CSIhist[t], CUIhist[t], USDhist[t], \
        UUDhist[t], USIhist[t], UUIhist[t], Ihist[t], Chist[t] \
        = mdefs(XXhist[t+1,:], XXhist[t,:], zhist[t], paramsn)

    # unpack Xhist
    KSDhist = XXhist[:,0]
    KUDhist = XXhist[:,1]    
    BSDhist = XXhist[:,2]
    BUDhist = XXhist[:,3]    
        
    return KSDhist, KUDhist, BSDhist, BUDhist, Khist, Bhist, NShist, NUhist, \
        Whist, Yhist, sShist, sUhist, Xhist, rhist, wShist, wUhist, CSDhist, \
        CUDhist, CSIhist, CUIhist, USDhist, UUDhist, USIhist, UUIhist, Ihist, \
        Chist

def runmc(simargs, nsim, nobs, repincr):
    '''
    This function returns all the results from a set of Monte Carlo simulations
    of the Simple ILA model.
    
    This function reads values from the following pkl files:
        ??.pkl - steady state values and parameters
    
    Inputs:
    -----------  
    funcname: name of the policy simulation function to be used.
        The function must be set up to take a single argument which is a list
    args: the list of arguments to be used by funcname
    nsim: the number of Monte Carlo simulations to run
    nobs: the number of observations in each simulation
    repincr:  the increment between MC reports (helps to see how fast the
        simulations run)
    
    Outputs:
    -----------  
    mcdata: a list of numpy arrays with simulations in the rows and
        observations in the columns
    histdata: a list of 1-dimensional numpy arrays for the final simulation 
    '''
    
    # load steady state values and parameters
    infile = open('UN.pkl', 'rb')
    (params, paramsn, bars, barsn, Momdf, IRFserlist, LinCoeffs, LinCoeffsn) \
        = pkl.load(infile)
    infile.close()
    
    # unpack
    [KSDbar, KUDbar, BSDbar, BUDbar, \
        Kbar, Bbar, NSbar, NUbar, Wbar, GDPbar, sSbar, sUbar, EXbar, rbar, wSbar, \
        wUbar,  CSDbar,  CUDbar, CSIbar, CUIbar, USDbar, UUDbar, USIbar, UUIbar, \
        Ibar, Cbar] = bars
    [KSDbarn, KUDbarn, BSDbarn, BUDbarn, \
        Kbarn, Bbarn, NSbarn, NUbarn, Wbarn, GDPbarn, sSbarn, sUbarn, EXbarn, \
        rbarn, wSbarn, wUbarn,  CSDbarn,  CUDbarn, CSIbarn, CUIbarn, USDbarn, \
        UUDbarn, USIbarn, UUIbarn, Ibarn, Cbarn] = barsn
    [g, beta, delta, gamma, a, b, c, d, f, HSD, HUD, HSI, HUI, nu, mu, \
            ss, q, rho, sigma, beta, nx, ny, nz] = params
    [g, beta, delta, gamma, a, b, c, d, f, HSDn, HUDn, HSIn, HUIn, nu, mu, ss, q, \
        rho, sigma, beta, nx, ny, nz] = paramsn
    (PP, QQ, UU, RR, SS, VV) = LinCoeffs
    (PPn, QQn, UUn, RRn, SSn, VVn) = LinCoeffsn
    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
    
    (initial, nobs, ts, generateLIN, args1, args2, params1, params2) = simargs
    
    # preallocate mc matrices
    KSDmc = np.zeros((nsim, nobs+1))
    KUDmc = np.zeros((nsim, nobs+1))
    BSDmc = np.zeros((nsim, nobs+1))
    BUDmc = np.zeros((nsim, nobs+1))
    Kmc   = np.zeros((nsim, nobs))
    Bmc   = np.zeros((nsim, nobs))
    NSmc  = np.zeros((nsim, nobs)) 
    NUmc  = np.zeros((nsim, nobs))
    Wmc   = np.zeros((nsim, nobs))
    Ymc   = np.zeros((nsim, nobs)) 
    sSmc  = np.zeros((nsim, nobs)) 
    sUmc  = np.zeros((nsim, nobs))
    Xmc   = np.zeros((nsim, nobs)) 
    rmc   = np.zeros((nsim, nobs)) 
    wSmc  = np.zeros((nsim, nobs)) 
    wUmc  = np.zeros((nsim, nobs))
    CSDmc = np.zeros((nsim, nobs))
    CUDmc = np.zeros((nsim, nobs)) 
    CSImc = np.zeros((nsim, nobs)) 
    CUImc = np.zeros((nsim, nobs)) 
    USDmc = np.zeros((nsim, nobs)) 
    UUDmc = np.zeros((nsim, nobs)) 
    USImc = np.zeros((nsim, nobs)) 
    UUImc = np.zeros((nsim, nobs)) 
    Imc   = np.zeros((nsim, nobs)) 
    Cmc   = np.zeros((nsim, nobs)) 
    
    # run simulations                                
    for i in range(0, nsim):
        if np.fmod(i, repincr) == 0.:
            print('mc #:', i, 'of', nsim)
        KSDhist, KUDhist, BSDhist, BUDhist, Khist, Bhist, NShist, NUhist, \
        Whist, Yhist, sShist, sUhist, Xhist, rhist, wShist, wUhist, CSDhist, \
        CUDhist, CSIhist, CUIhist, USDhist, UUDhist, USIhist, UUIhist, Ihist, \
        Chist = polsim(simargs)
                        
        # store results in Monte Carlo matrices
        KSDmc[i,:] = KSDhist
        KUDmc[i,:] = KUDhist
        BSDmc[i,:] = BSDhist
        BUDmc[i,:] = BUDhist
        Kmc[i,:]   = Khist
        Bmc[i,:]   = Bhist
        NSmc[i,:]  = NShist
        NUmc[i,:]  = NUhist
        Wmc[i,:]   = Whist
        Ymc[i,:]   = Yhist
        sSmc[i,:]  = sShist
        sUmc[i,:]  = sUhist
        Xmc[i,:]   = Xhist
        rmc[i,:]   = rhist
        wSmc[i,:]  = wShist
        wUmc[i,:]  = wUhist
        CSDmc[i,:] = CSDhist
        CUDmc[i,:] = CUDhist
        CSImc[i,:] = CSIhist
        CUImc[i,:] = CUIhist
        USDmc[i,:] = USDhist
        UUDmc[i,:] = UUDhist
        USImc[i,:] = USIhist
        UUImc[i,:] = UUIhist
        Imc[i,:]   = Ihist
        Cmc[i,:]   = Chist

        mcdata = (KSDmc, KUDmc, BSDmc, BUDmc, Kmc, Bmc, NSmc, \
        NUmc, Wmc, Ymc, sSmc, sUmc, Xmc, rmc, wSmc, wUmc, \
        CSDmc, CUDmc, CSImc, CUImc, USDmc, UUDmc, USImc, \
        UUImc, Imc, Cmc)
        
        histdata = (KSDhist, KUDhist, BSDhist, BUDhist, Khist, Bhist, NShist, \
        NUhist, Whist, Yhist, sShist, sUhist, Xhist, rhist, wShist, wUhist, \
        CSDhist, CUDhist, CSIhist, CUIhist, USDhist, UUDhist, USIhist, \
        UUIhist, Ihist, Chist)
        
    return mcdata, histdata


def mcanalysis(mcdata, bardata, histdata, nsim):
    '''
    This function finds confidence bands for data from the Monte Carlo
    simulations.  It also plots predictions and with confidence bands, and 
    predictions versus the final simulation as an example.
    
    Inputs:
    -----------  
    mcdata: a list of numpy arrays with simulations in the rows and
        observations in the columns
    bardata: a list of steady state values from the baseline
    histdata: a list of 1-dimensional numpy arrays for the final simulation 
    name: a string that is used when saving the plots and other files
    nsim: the number of Monte Carlo simulations that have been run
    
    Outputs:
    -----------  
    avgdata: list of 1-dimensional numpy arrays containing the average values 
        from the simulations for each time period
    uppdata: list of 1-dimensional numpy arrays containing the upper confidence
        bands from the simulations for each time period
    lowdata: list of 1-dimensional numpy arrays containing the lower confidence
        bands from the simulations for each time period
    '''    
    
    #unpack data
    (KSDmc, KUDmc, BSDmc, BUDmc, Kmc, Bmc, NSmc, \
        NUmc, Wmc, Ymc, sSmc, sUmc, Xmc, rmc, wSmc, wUmc, \
        CSDmc, CUDmc, CSImc, CUImc, USDmc, UUDmc, USImc, \
        UUImc, Imc, Cmc) = mcdata
    (KSDhist, KUDhist, BSDhist, BUDhist, Khist, Bhist, NShist, \
        NUhist, Whist, Yhist, sShist, sUhist, Xhist, rhist, wShist, wUhist, \
        CSDhist, CUDhist, CSIhist, CUIhist, USDhist, UUDhist, USIhist, \
        UUIhist, Ihist, Chist) = histdata
          
    # now sort the Monte Carlo matrices over the rows
    KSDmc = np.sort(KSDmc, axis = 0)
    KUDmc = np.sort(KUDmc, axis = 0)
    BSDmc = np.sort(BSDmc, axis = 0)
    BUDmc = np.sort(BUDmc, axis = 0)
    Kmc   = np.sort(Kmc, axis = 0)
    Bmc   = np.sort(Bmc, axis = 0)
    NSmc  = np.sort(NSmc, axis = 0)
    NUmc  = np.sort(NUmc, axis = 0)
    Wmc   = np.sort(Wmc, axis = 0)
    Ymc   = np.sort(Ymc, axis = 0)
    sSmc  = np.sort(sSmc, axis = 0)
    sUmc  = np.sort(sUmc, axis = 0)
    Xmc   = np.sort(Xmc, axis = 0)
    rmc   = np.sort(rmc, axis = 0)
    wSmc  = np.sort(wSmc, axis = 0)
    wUmc  = np.sort(wUmc, axis = 0)
    CSDmc = np.sort(CSDmc, axis = 0)
    CUDmc = np.sort(CUDmc, axis = 0)
    CSImc = np.sort(CSImc, axis = 0)
    CUImc = np.sort(CUImc, axis = 0)
    USDmc = np.sort(USDmc, axis = 0)
    UUDmc = np.sort(UUDmc, axis = 0)
    USImc = np.sort(USImc, axis = 0)
    UUImc = np.sort(UUImc, axis = 0)
    Imc   = np.sort(Imc, axis = 0)
    Cmc   = np.sort(Cmc, axis = 0)

    
    # find the average values for each variable in each time period across 
    # Monte Carlos
    KSDavg = np.mean(KSDmc, axis = 0)
    KUDavg = np.mean(KUDmc, axis = 0)
    BSDavg = np.mean(BSDmc, axis = 0)
    BUDavg = np.mean(BUDmc, axis = 0)
    Kavg   = np.mean(Kmc, axis = 0)
    Bavg   = np.mean(Bmc, axis = 0)
    NSavg  = np.mean(NSmc, axis = 0)
    NUavg  = np.mean(NUmc, axis = 0)
    Wavg   = np.mean(Wmc, axis = 0)
    Yavg   = np.mean(Ymc, axis = 0)
    sSavg  = np.mean(sSmc, axis = 0)
    sUavg  = np.mean(sUmc, axis = 0)
    Xavg   = np.mean(Xmc, axis = 0)
    ravg   = np.mean(rmc, axis = 0)
    wSavg  = np.mean(wSmc, axis = 0)
    wUavg  = np.mean(wUmc, axis = 0)
    CSDavg = np.mean(CSDmc, axis = 0)
    CUDavg = np.mean(CUDmc, axis = 0)
    CSIavg = np.mean(CSImc, axis = 0)
    CUIavg = np.mean(CUImc, axis = 0)
    USDavg = np.mean(USDmc, axis = 0)
    UUDavg = np.mean(UUDmc, axis = 0)
    USIavg = np.mean(USImc, axis = 0)
    UUIavg = np.mean(UUImc, axis = 0)
    Iavg   = np.mean(Imc, axis = 0)
    Cavg   = np.mean(Cmc, axis = 0)

    
    # find the rows for desired confidence bands
    conf = .1
    low = int(np.floor((conf/2)*nsim))
    high = nsim - low
    
    # find the upper and lower confidence bands for each variable
    KSDupp = KSDmc[high,:]
    KUDupp = KUDmc[high,:]
    BSDupp = BSDmc[high,:]
    BUDupp = BUDmc[high,:]
    Kupp   = Kmc[high,:]
    Bupp   = Bmc[high,:]
    NSupp  = NSmc[high,:]
    NUupp  = NUmc[high,:]
    Wupp   = Wmc[high,:]
    Yupp   = Ymc[high,:]
    sSupp  = sSmc[high,:]
    sUupp  = sUmc[high,:]
    Xupp   = Xmc[high,:]
    rupp   = rmc[high,:]
    wSupp  = wSmc[high,:]
    wUupp  = wUmc[high,:]
    CSDupp = CSDmc[high,:]
    CUDupp = CUDmc[high,:]
    CSIupp = CSImc[high,:]
    CUIupp = CUImc[high,:]
    USDupp = USDmc[high,:]
    UUDupp = UUDmc[high,:]
    USIupp = USImc[high,:]
    UUIupp = UUImc[high,:]
    Iupp   = Imc[high,:]
    Cupp   = Cmc[high,:]

    KSDlow= KSDmc[low,:]
    KUDlow= KUDmc[low,:]
    BSDlow= BSDmc[low,:]
    BUDlow= BUDmc[low,:]
    Klow  = Kmc[low,:]
    Blow  = Bmc[low,:]
    NSlow = NSmc[low,:]
    NUlow = NUmc[low,:]
    Wlow  = Wmc[low,:]
    Ylow  = Ymc[low,:]
    sSlow = sSmc[low,:]
    sUlow = sUmc[low,:]
    Xlow  = Xmc[low,:]
    rlow  = rmc[low,:]
    wSlow = wSmc[low,:]
    wUlow = wUmc[low,:]
    CSDlow= CSDmc[low,:]
    CUDlow= CUDmc[low,:]
    CSIlow= CSImc[low,:]
    CUIlow= CUImc[low,:]
    USDlow= USDmc[low,:]
    UUDlow= UUDmc[low,:]
    USIlow= USImc[low,:]
    UUIlow= UUImc[low,:]
    Ilow  = Imc[low,:]
    Clow  = Cmc[low,:]

    
    # create lists of data to return
    avgdata = (KSDavg, KUDavg, BSDavg, BUDavg, Kavg, Bavg, NSavg, \
        NUavg, Wavg, Yavg, sSavg, sUavg, Xavg, ravg, wSavg, wUavg, \
        CSDavg, CUDavg, CSIavg, CUIavg, USDavg, UUDavg, USIavg, \
        UUIavg, Iavg, Cavg) 
    uppdata = (KSDupp, KUDupp, BSDupp, BUDupp, Kupp, Bupp, NSupp, \
        NUupp, Wupp, Yupp, sSupp, sUupp, Xupp, rupp, wSupp, wUupp, \
        CSDupp, CUDupp, CSIupp, CUIupp, USDupp, UUDupp, USIupp, \
        UUIupp, Iupp, Cupp) 
    lowdata = (KSDlow, KUDlow, BSDlow, BUDlow, Klow, Blow, NSlow, \
        NUlow, Wlow, Ylow, sSlow, sUlow, Xlow, rlow, wSlow, wUlow, \
        CSDlow, CUDlow, CSIlow, CUIlow, USDlow, UUDlow, USIlow, \
        UUIlow, Ilow, Clow) 
    
    return avgdata, uppdata, lowdata


# -----------------------------------------------------------------------------
# MAIN PROGRAM

# LOAD VALUES FROM SS AND LINEARIZATION
    
# load steady state values and parameters
infile = open('UN.pkl', 'rb')
(params, paramsn, bars, barsn, Momdf, IRFserlist, LinCoeffs, LinCoeffsn) \
    = pkl.load(infile)
infile.close()

# unpack
[KSDbar, KUDbar, BSDbar, BUDbar, \
    Kbar, Bbar, NSbar, NUbar, Wbar, GDPbar, sSbar, sUbar, EXbar, rbar, wSbar, \
    wUbar,  CSDbar,  CUDbar, CSIbar, CUIbar, USDbar, UUDbar, USIbar, UUIbar, \
    Ibar, Cbar] = bars
[KSDbarn, KUDbarn, BSDbarn, BUDbarn, \
    Kbarn, Bbarn, NSbarn, NUbarn, Wbarn, GDPbarn, sSbarn, sUbarn, EXbarn, \
    rbarn, wSbarn, wUbarn,  CSDbarn,  CUDbarn, CSIbarn, CUIbarn, USDbarn, \
    UUDbarn, USIbarn, UUIbarn, Ibarn, Cbarn] = barsn
[g, beta, delta, gamma, a, b, c, d, f, HSD, HUD, HSI, HUI, nu, mu, \
        ss, q, rho, sigma, beta, nx, ny, nz] = params
[g, beta, delta, gamma, a, b, c, d, f, HSDn, HUDn, HSIn, HUIn, nu, mu, ss, q, \
    rho, sigma, beta, nx, ny, nz] = paramsn
nx = int(nx)
ny = int(ny)
nz = int(nz)
(PP, QQ, UU, RR, SS, VV) = LinCoeffs
(PPn, QQn, UUn, RRn, SSn, VVn) = LinCoeffsn

# create args lists
XYbar = (KSDbar, KUDbar, BSDbar, BUDbar)
XYbarn = (KSDbarn, KUDbarn, BSDbarn, BUDbarn)
args = (LinCoeffs, XYbar)
argsn = (LinCoeffsn, XYbarn)

# -----------------------------------------------------------------------------
# RUN MONTE CARLOS

# specify the number of observations per simulation
nobs = 160
# specify the period policy shifts
ts = 20
# specify the number of simulations
nsim = 100000
# specify the increment between MC reports
repincr = 100

# specify initial values
initial = (XYbar, 0.)

# get list of arguments for monte carlos simulations 
simargs = (initial, nobs, ts, generateLIN, args, argsn, params, paramsn)

# run the Monte Carlos
mcdata, histdata = runmc(simargs, nsim, nobs, repincr)

#unpack
(KSDmc, KUDmc, BSDmc, BUDmc, Kmc, Bmc, NSmc, \
        NUmc, Wmc, Ymc, sSmc, sUmc, Xmc, rmc, wSmc, wUmc, \
        CSDmc, CUDmc, CSImc, CUImc, USDmc, UUDmc, USImc, \
        UUImc, Imc, Cmc) = mcdata
(KSDhist, KUDhist, BSDhist, BUDhist, Khist, Bhist, NShist, \
        NUhist, Whist, Yhist, sShist, sUhist, Xhist, rhist, wShist, wUhist, \
        CSDhist, CUDhist, CSIhist, CUIhist, USDhist, UUDhist, USIhist, \
        UUIhist, Ihist, Chist) = histdata
# -----------------------------------------------------------------------------
# DO ANALYSIS

avgdata, uppdata, lowdata = mcanalysis(mcdata, bars, histdata, nsim)
# unpack
(KSDavg, KUDavg, BSDavg, BUDavg, Kavg, Bavg, NSavg, \
        NUavg, Wavg, Yavg, sSavg, sUavg, Xavg, ravg, wSavg, wUavg, \
        CSDavg, CUDavg, CSIavg, CUIavg, USDavg, UUDavg, USIavg, \
        UUIavg, Iavg, Cavg) = avgdata
(KSDupp, KUDupp, BSDupp, BUDupp, Kupp, Bupp, NSupp, \
        NUupp, Wupp, Yupp, sSupp, sUupp, Xupp, rupp, wSupp, wUupp, \
        CSDupp, CUDupp, CSIupp, CUIupp, USDupp, UUDupp, USIupp, \
        UUIupp, Iupp, Cupp) = uppdata
(KSDlow, KUDlow, BSDlow, BUDlow, Klow, Blow, NSlow, \
        NUlow, Wlow, Ylow, sSlow, sUlow, Xlow, rlow, wSlow, wUlow, \
        CSDlow, CUDlow, CSIlow, CUIlow, USDlow, UUDlow, USIlow, \
        UUIlow, Ilow, Clow) = lowdata

# -----------------------------------------------------------------------------
# PLOT
 
time = range(0, nobs-1)

plt.figure()
plt.subplot(2,2,1)
plt.plot(time, Yavg[0:nobs-1], 'k-')
plt.plot(time, Yupp[0:nobs-1], 'k:')
plt.plot(time, Ylow[0:nobs-1], 'k:')
plt.title('GDP')
plt.xticks([])
plt.subplot(2,2,2)
plt.plot(time, Cavg[0:nobs-1], 'k-')
plt.plot(time, Cupp[0:nobs-1], 'k:')
plt.plot(time, Clow[0:nobs-1], 'k:')
plt.title('Consumption')
plt.xticks([])
plt.subplot(2,2,3)
plt.plot(time, Iavg[0:nobs-1], 'k-')
plt.plot(time, Iupp[0:nobs-1], 'k:')
plt.plot(time, Ilow[0:nobs-1], 'k:')
plt.title('Investment')
plt.subplot(2,2,4)
plt.plot(time, Bavg[0:nobs-1], 'k-')
plt.plot(time, Bupp[0:nobs-1], 'k:')
plt.plot(time, Blow[0:nobs-1], 'k:')
plt.title('International Savings')

plt.savefig('JapImmAgg_mc.pdf', format='pdf', dpi=2000)
plt.show()

plt.figure()
plt.subplot(2,2,1)
plt.plot(time, wSavg[0:nobs-1], 'k-')
plt.plot(time, wSupp[0:nobs-1], 'k:')
plt.plot(time, wSlow[0:nobs-1], 'k:')
plt.title('Skilled Wage')
plt.xticks([])
plt.subplot(2,2,2)
plt.plot(time, wUavg[0:nobs-1], 'k-')
plt.plot(time, wUupp[0:nobs-1], 'k:')
plt.plot(time, wUlow[0:nobs-1], 'k:')
plt.title('Unskilled Wage')
plt.xticks([])
plt.subplot(2,2,3)
plt.plot(time, sSavg[0:nobs-1], 'k-')
plt.plot(time, sSupp[0:nobs-1], 'k:')
plt.plot(time, sSlow[0:nobs-1], 'k:')
plt.title('Skilled Interest Rate')
plt.subplot(2,2,4)
plt.plot(time, sUavg[0:nobs-1], 'k-')
plt.plot(time, sUupp[0:nobs-1], 'k:')
plt.plot(time, sUlow[0:nobs-1], 'k:')
plt.title('Unskilled Interest Rate')
plt.savefig('JapImmWages_mc.pdf', format='pdf', dpi=2000)
plt.show()
 
plt.figure()
plt.subplot(2,2,1)
plt.plot(time, USDavg[0:nobs-1], 'k-')
plt.plot(time, USDupp[0:nobs-1], 'k:')
plt.plot(time, USDlow[0:nobs-1], 'k:')
plt.title('Skilled Domestic')
plt.xticks([])
plt.subplot(2,2,2)
plt.plot(time, UUDavg[0:nobs-1], 'k-')
plt.plot(time, UUDupp[0:nobs-1], 'k:')
plt.plot(time, UUDlow[0:nobs-1], 'k:')
plt.title('Unskilled Domestic')
plt.xticks([])
plt.subplot(2,2,3)
plt.plot(time, USIavg[0:nobs-1], 'k-')
plt.plot(time, USIupp[0:nobs-1], 'k:')
plt.plot(time, USIlow[0:nobs-1], 'k:')
plt.title('Skilled Immigrant')
plt.subplot(2,2,4)
plt.plot(time, UUIavg[0:nobs-1], 'k-')
plt.plot(time, UUIupp[0:nobs-1], 'k:')
plt.plot(time, UUIlow[0:nobs-1], 'k:')
plt.title('Unskilled Immigrant')
plt.savefig('JapImmUtil_mc.pdf', format='pdf', dpi=2000)
plt.show()

# -----------------------------------------------------------------------------
# SAVE RESULTS

output = open(name + '.pkl', 'wb')

alldata = (mcdata, histdata, avgdata, uppdata, lowdata)

pkl.dump(alldata, output)

output.close()