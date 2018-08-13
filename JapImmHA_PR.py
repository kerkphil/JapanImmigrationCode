# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:41:16 2017

@author: Kerk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle as pkl

from DSGEmoments import calcmom
from LinApp_FindSS import LinApp_FindSS
from LinApp_Deriv import LinApp_Deriv
from LinApp_Solve import LinApp_Solve
from LinApp_Sim import LinApp_Sim

# -----------------------------------------------------------------------------
# DEF AND DYN FUNCTIONS

def mdefs(KSDp, KUDp, BSDp, BUDp, KSD, KUD, BSD, BUD, \
          z, params):
    
    # unpack params
    [g, beta, delta, gamma, a, b, c, d, f, HSD, HUD, HSI, HUI, \
    nu, mu, ss, q, rho, sigma, beta, nx, ny, nz] \
        = params
    
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
    wS = f*(1-a)*(Y/W)**(1/b)*c*(W/NS)**(1/d)*np.exp(z*(1-1/d))
    wU = (1-a)*(Y/W)**(1/b)*(1-c)*(W/NU)**(1/d)*np.exp(z*(1-1/d))
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
     
    g, beta, delta, gamma, a, b, c, d, f, HSD, HUD, HSI, HUI, \
    nu, mu, ss, q, rho, sigma, beta, nx, ny, nz \
        = params
    
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


def LinApp_Policy(nobs, switch, Zhist, X0, Xbar1, Xbar2, LinCoeffs1, \
    LinCoeffs2, nx, nz):
    
    # unpack
    (PP1, QQ1, UU1, RR1, SS1, VV1) = LinCoeffs1
    (PP2, QQ2, UU2, RR2, SS2, VV2) = LinCoeffs2
    
    # initialize
    Xhist = np.zeros((nobs+1, nx))
    
    Xhist[0,:] = X0

    for t in range(0, switch):
        X = Xhist[t, :] - Xbar1
        Xp, Y = LinApp_Sim(X,Zhist[t,:],PP1,QQ1,UU1,RR1,SS1,VV1)
        Xhist[t+1,:] = Xp + Xbar1
    
    for t in range(switch, nobs):
        X = Xhist[t, :] - Xbar2
        Xp, Y = LinApp_Sim(X,Zhist[t,:],PP2,QQ2,UU2,RR2,SS2,VV2)
        Xhist[t+1,:] = Xp + Xbar2
    
    return Xhist


def runsim(nobs, switch, Xbar1, Xbar2, LinCoeffs1, LinCoeffs2, nx, nz, IRF):
    
    # generate history of z's
    zhist = np.zeros(nobs+1)
    if IRF:
        epszhist = np.zeros(nobs+1)
    else:
        epszhist = np.random.normal(0., sigma, nobs+1)
    
    Zhist = np.zeros((nobs+1,nz))
    
    zhist[0] = epszhist[0]
    Zhist[0,0] = zhist[0]
    for t in range(0,nobs):
        zhist[t+1] = rho*zhist[t] + epszhist[t+1]
        Zhist[t+1,0] = zhist[t+1]
    
    Xhist = LinApp_Policy(nobs, switch, Zhist, Xbar1, Xbar1, Xbar2, \
        LinCoeffs1, LinCoeffs2, nx, nz)
    
    # unpack
    KSDhist = Xhist[:,0] 
    KUDhist = Xhist[:,1] 
    BSDhist = Xhist[:,2] 
    BUDhist = Xhist[:,3]
    
    # initialize
    Khist   = np.zeros(nobs)
    Bhist   = np.zeros(nobs)
    NShist  = np.zeros(nobs)
    NUhist  = np.zeros(nobs)
    Whist   = np.zeros(nobs)
    GDPhist = np.zeros(nobs)
    sShist  = np.zeros(nobs)
    sUhist  = np.zeros(nobs)
    EXhist  = np.zeros(nobs)
    rhist   = np.zeros(nobs)
    wShist  = np.zeros(nobs)
    wUhist  = np.zeros(nobs)
    CSDhist = np.zeros(nobs)
    CUDhist = np.zeros(nobs)
    CSIhist = np.zeros(nobs)
    CUIhist = np.zeros(nobs)
    USDhist = np.zeros(nobs)
    UUDhist = np.zeros(nobs)
    USIhist = np.zeros(nobs)
    UUIhist = np.zeros(nobs)
    Ihist   = np.zeros(nobs)
    Chist   = np.zeros(nobs)
    
    for t in range(0, switch):
        Khist[t], Bhist[t], NShist[t], NUhist[t], Whist[t], GDPhist[t], \
            sShist[t], sUhist[t], EXhist[t], rhist[t], wShist[t], wUhist[t], \
            CSDhist[t], CUDhist[t], CSIhist[t], CUIhist[t], USDhist[t], \
            UUDhist[t], USIhist[t], UUIhist[t], Ihist[t], Chist[t] \
            = mdefs(KSDhist[t+1], KUDhist[t+1], BSDhist[t+1], \
            BUDhist[t+1], KSDhist[t], KUDhist[t], BSDhist[t], BUDhist[t], \
            zhist[t], params)
            
    for t in range(switch, nobs):
        Khist[t], Bhist[t], NShist[t], NUhist[t], Whist[t], GDPhist[t], \
            sShist[t], sUhist[t], EXhist[t], rhist[t], wShist[t], wUhist[t], \
            CSDhist[t], CUDhist[t], CSIhist[t], CUIhist[t], USDhist[t], \
            UUDhist[t], USIhist[t], UUIhist[t], Ihist[t], Chist[t] \
            = mdefs(KSDhist[t+1], KUDhist[t+1], BSDhist[t+1], \
            BUDhist[t+1], KSDhist[t], KUDhist[t], BSDhist[t], BUDhist[t], \
            zhist[t], paramsn)
            
    KSDhist = KSDhist[0:nobs] 
    KUDhist = KUDhist[0:nobs] 
    BSDhist = BSDhist[0:nobs] 
    BUDhist = BUDhist[0:nobs]
    
    return zhist, KSDhist, KUDhist, BSDhist, BUDhist, Khist, Bhist, NShist, \
        NUhist, Whist, GDPhist, sShist, sUhist, EXhist, rhist, wShist, wUhist, \
        CSDhist, CUDhist, CSIhist, CUIhist, USDhist, UUDhist, USIhist, \
        UUIhist, Ihist, Chist
        
        
def plots(nobs, serlist, name):
    
    time = range(0, nobs-1)

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(time, Khist[0:nobs-1], label='K')
    plt.title('Domestic Capital')
    plt.xticks([])
    plt.subplot(2,2,2)
    plt.plot(time, GDPhist[0:nobs-1], label='Y')
    plt.title('GDP')
    plt.xticks([])
    plt.subplot(2,1,2)
    plt.plot(time, Bhist[0:nobs-1], label='B')
    plt.plot(time, EXhist[0:nobs-1], label='X')
    plt.legend(loc='center left', bbox_to_anchor=(.8, 0.5))
    plt.title('International Aggregate Variables')
    plt.savefig(name + 'fig1.pdf', format='pdf', dpi=2000)
    plt.show()
    
    plt.figure()
    plt.plot(time, BSDhist[0:nobs-1], label='BSD')
    plt.plot(time, BUDhist[0:nobs-1], label='BUD')
    plt.legend(loc='center left', bbox_to_anchor=(.8, 0.5))
    plt.title('International Savings')
    plt.savefig(name + 'fig2.pdf', format='pdf', dpi=2000)
    plt.show()
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(time, wShist[0:nobs-1], label='wS')
    plt.title('wS')
    plt.xticks([])
    plt.subplot(2,2,2)
    plt.plot(time, wUhist[0:nobs-1], label='wU')
    plt.title('wU')
    plt.xticks([])
    plt.subplot(2,1,2)
    plt.plot(time, rhist[0:nobs-1] - delta, label='r-delta')
    plt.plot(time, sShist[0:nobs-1], label='sS')
    plt.plot(time, sUhist[0:nobs-1], label='sU')
    plt.legend(loc='center left', bbox_to_anchor=(.8, 0.5))
    plt.title('Interest Rates')
    plt.savefig(name + 'fig3.pdf', format='pdf', dpi=2000)
    plt.show()
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(time, CSDhist[0:nobs-1], label='USD')
    plt.title('CSD')
    plt.xticks([])
    plt.subplot(2,2,2)
    plt.plot(time, CUDhist[0:nobs-1], label='UUD')
    plt.title('CUD')
    plt.xticks([])
    plt.subplot(2,2,3)
    plt.plot(time, CSIhist[0:nobs-1], label='USI')
    plt.title('CSI')
    plt.subplot(2,2,4)
    plt.plot(time, CUIhist[0:nobs-1], label='UUI')
    plt.title('CUI')
    plt.savefig(name + 'fig4.pdf', format='pdf', dpi=2000)
    plt.show()
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(time, USDhist[0:nobs-1], label='USD')
    plt.title('USD')
    plt.xticks([])
    plt.subplot(2,2,2)
    plt.plot(time, UUDhist[0:nobs-1], label='UUD')
    plt.title('UUD')
    plt.xticks([])
    plt.subplot(2,2,3)
    plt.plot(time, USIhist[0:nobs-1], label='USI')
    plt.title('USI')
    plt.subplot(2,2,4)
    plt.plot(time, UUIhist[0:nobs-1], label='UUI')
    plt.title('UUI')
    plt.savefig(name + 'fig5.pdf', format='pdf', dpi=2000)
    plt.show()


# -----------------------------------------------------------------------------
# SETUP
name = 'PR'

# declare model parameters
g = .0
delta = .1131/4
gamma = 2.5
a = .38
b = .7 # .7 from Pessoa et al (2005)
c = .4952
d = 2.0 # 2.0 from Behar (2010)
f = 1.8175 # multiplicative factor set wS/wU to 2016 data value of 3.239
HSD = .2296
HUD = .7580
HSI = .0029
HUI = .0095
nu = .81 # .17
mu = .0
q = .799
ss = .006043/4  #.006043 average real return in USA 2003-2016
rho = .95
sigma = .0124
beta =  1/(1+ss)
        
# set program parameters
nx = 4
ny = 0
nz = 1
dotypical = False
domoments = True
doIRFs = True

params = np.array([g, beta, delta, gamma, a, b, c, d, f, HSD, HUD, HSI, HUI, \
    nu, mu, ss, q, rho, sigma, beta, nx, ny, nz])

# -----------------------------------------------------------------------------
# BASELINE STEADY STATE

guessXY = np.ones((1,nx+ny))*.1
Zbar = np.array([0.])
Xbar = LinApp_FindSS(mdyn, params, guessXY, Zbar, nx, ny)

In = np.concatenate((Xbar, Xbar, Xbar, Zbar, Zbar))

check = mdyn(In, params)
print('check: ', check)

[KSDbar, KUDbar, BSDbar, BUDbar] = Xbar

Kbar, Bbar, NSbar, NUbar, Wbar, GDPbar, sSbar, sUbar, EXbar, rbar, wSbar, \
    wUbar, CSDbar, CUDbar, CSIbar, CUIbar, USDbar, UUDbar, USIbar, UUIbar, \
    Ibar, Cbar = mdefs(KSDbar, KUDbar, BSDbar, BUDbar, KSDbar, KUDbar, BSDbar,\
    BUDbar, 0., params)

bars = np.array([KSDbar, KUDbar, BSDbar, BUDbar, \
    Kbar, Bbar, NSbar, NUbar, Wbar, GDPbar, sSbar, sUbar, EXbar, rbar, wSbar, \
    wUbar,  CSDbar,  CUDbar, CSIbar, CUIbar, USDbar, UUDbar, USIbar, UUIbar, \
    Ibar, Cbar])  

# list the variable names in order
varindex = ['KSDbar', 'KUDbar', 'BSDbar', 'BUDbar', \
    'Kbar', 'Bbar', 'NSbar', 'NUbar', 'Wbar', 'GDPbar', 'sSbar', 'sUbar',\
    'EXbar', 'rbar', 'wSbar', 'wUbar', 'CSDbar', 'CUDbar', 'CSIbar', \
    'CUIbar', 'USDbar', 'UUDbar', 'USIbar', 'UUIbar', 'Ibar', 'Cbar'] 

barsdf = pd.DataFrame(bars.T)
barsdf.index = varindex   

# print (barsdf.to_latex())

# -----------------------------------------------------------------------------
# BASELINE LINEARIZATION

# set autocorrelation matrix
NN = np.array([rho])

thetabar = np.array([KSDbar, KUDbar, BSDbar, BUDbar, KSDbar, KUDbar, BSDbar, \
    BUDbar, KSDbar, KUDbar, BSDbar, BUDbar, 0., 0.])

[AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT] = \
    LinApp_Deriv(mdyn, params, thetabar, nx, ny, nz, False)
             
PP, QQ, UU, RR, SS, VV = \
    LinApp_Solve(AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT, NN, \
                 Zbar, False)   

LinCoeffs = (PP, QQ, UU, RR, SS, VV)   

# -----------------------------------------------------------------------------
# NEW STEADY STATE
HSDn = .2296
HUDn = .7580
HSIn = .0119
HUIn = .0381
paramsn = np.array([g, beta, delta, gamma, a, b, c, d, f, HSDn, HUDn, HSIn, \
                    HUIn, nu, mu, ss, q, rho, sigma, beta, nx, ny, nz])

guessXY = Xbar #np.ones((1,nx+ny))*.1

Xbarn = LinApp_FindSS(mdyn, paramsn, guessXY, Zbar, nx, ny)

Inn = np.concatenate((Xbarn, Xbarn, Xbarn, Zbar, Zbar))

checkn = mdyn(Inn, paramsn)
print('check new: ', checkn)

[KSDbarn, KUDbarn, BSDbarn, BUDbarn] = Xbarn

Kbarn, Bbarn, NSbarn, NUbarn, Wbarn, GDPbarn, sSbarn, sUbarn, EXbarn, rbarn, \
    wSbarn, wUbarn, CSDbarn, CUDbarn, CSIbarn, CUIbarn, USDbarn, UUDbarn, \
    USIbarn, UUIbarn, Ibarn, Cbarn = mdefs(KSDbarn, KUDbarn, BSDbarn, BUDbarn,\
    KSDbarn, KUDbarn, BSDbarn, BUDbarn, 0., paramsn)

barsn = np.array([KSDbarn, KUDbarn, BSDbarn, BUDbarn, \
    Kbarn, Bbarn, NSbarn, NUbarn, Wbarn, GDPbarn, sSbarn, sUbarn, EXbarn, \
    rbarn, wSbarn, wUbarn,  CSDbarn,  CUDbarn, CSIbarn, CUIbarn, USDbarn, \
    UUDbarn, USIbarn, UUIbarn, Ibarn, Cbarn])  

barsndf = pd.DataFrame(barsn.T)
barsndf.index = varindex   

# print (barsndf.to_latex())

# -----------------------------------------------------------------------------
# NEW LINEARIZATION

# set autocorrelation matrix
NN = np.array([rho])

thetabarn = np.array([KSDbarn, KUDbarn, BSDbarn, BUDbarn, KSDbarn, KUDbarn, \
    BSDbarn, BUDbarn, KSDbarn, KUDbarn, BSDbarn, BUDbarn, 0., 0.])

[AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT] = \
    LinApp_Deriv(mdyn, paramsn, thetabarn, nx, ny, nz, False)
             
PPn, QQn, UUn, RRn, SSn, VVn = \
    LinApp_Solve(AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT, NN, \
                 Zbar, False)   

LinCoeffsn = (PPn, QQn, UUn, RRn, SSn, VVn)

# -----------------------------------------------------------------------------
# SIMULATE AND PLOT TYPICAL SIMULATION
if dotypical:
    nobs = 250
    switch = 125
    IRF = False
    
    zhist, KSDhist, KUDhist, BSDhist, BUDhist, Khist, Bhist, NShist, \
        NUhist, Whist, GDPhist, sShist, sUhist, EXhist, rhist, wShist, wUhist, \
        CSDhist, CUDhist, CSIhist, CUIhist, USDhist, UUDhist, USIhist, \
        UUIhist, Ihist, Chist = runsim(nobs, switch, Xbar, Xbarn, LinCoeffs, \
        LinCoeffsn, nx, nz, IRF)
        
    serlist = (zhist, KSDhist, KUDhist, BSDhist, BUDhist, Khist, Bhist, NShist, \
        NUhist, Whist, GDPhist, sShist, sUhist, EXhist, rhist, wShist, wUhist, \
        CSDhist, CUDhist, CSIhist, CUIhist, USDhist, UUDhist, USIhist, \
        UUIhist, Ihist, Chist)
    
    filename = name + 'SIM'
    plots(nobs, serlist, filename)



# -----------------------------------------------------------------------------
# GET MOMENTS

if domoments:
    nobs = 1000
    switch = nobs
    IRF = False
    
    zhist, KSDhist, KUDhist, BSDhist, BUDhist, Khist, Bhist, NShist, \
        NUhist, Whist, GDPhist, sShist, sUhist, EXhist, rhist, wShist, wUhist, \
        CSDhist, CUDhist, CSIhist, CUIhist, USDhist, UUDhist, USIhist, \
        UUIhist, Ihist, Chist = runsim(nobs, switch, Xbar, Xbarn, LinCoeffs, \
        LinCoeffsn, nx, nz, IRF)
        
    serlist = (zhist, KSDhist, KUDhist, BSDhist, BUDhist, Khist, Bhist, NShist, \
        NUhist, Whist, GDPhist, sShist, sUhist, EXhist, rhist, wShist, wUhist, \
        CSDhist, CUDhist, CSIhist, CUIhist, USDhist, UUDhist, USIhist, \
        UUIhist, Ihist, Chist)
    
    Data = np.vstack((GDPhist, Chist, Ihist, EXhist/GDPhist, Bhist/GDPhist))
    Data = Data.T
    Data = Data[0:nobs-1,:]
    
    # HP Filter
    Datadev, Datatrd = sm.tsa.filters.hpfilter(Data, 1600)
    
    varindex = ['GDP', 'Cons', 'Inv', 'NX/Y', 'B/Y']
    
    report, momindex = calcmom(Datadev, means = False, stds = True, \
        relstds = True, corrs = True, autos = True, cvars = False)
    
    Momdf = pd.DataFrame(report)
    Momdf.columns = varindex
    Momdf.index = momindex
    Momdf = Momdf.transpose()
    # print (Momdf.to_latex())

# -----------------------------------------------------------------------------
# SIMULATE AND PLOT IMPULSE REPOSNE FUNCTIONS

if doIRFs:
    nobs = 120
    switch = 10
    IRF = True
    
    zhist, KSDhist, KUDhist, BSDhist, BUDhist, Khist, Bhist, NShist, \
        NUhist, Whist, GDPhist, sShist, sUhist, EXhist, rhist, wShist, wUhist, \
        CSDhist, CUDhist, CSIhist, CUIhist, USDhist, UUDhist, USIhist, \
        UUIhist, Ihist, Chist = runsim(nobs, switch, Xbar, Xbarn, LinCoeffs, \
        LinCoeffsn, nx, nz, IRF)
        
    IRFserlist = (zhist, KSDhist, KUDhist, BSDhist, BUDhist, Khist, Bhist, NShist, \
        NUhist, Whist, GDPhist, sShist, sUhist, EXhist, rhist, wShist, wUhist, \
        CSDhist, CUDhist, CSIhist, CUIhist, USDhist, UUDhist, USIhist, \
        UUIhist, Ihist, Chist)
    
    filename = name + 'IRF'
    plots(nobs, IRFserlist, filename)
    
# -----------------------------------------------------------------------------
# SAVE RESULTS TP PKL FILE    

output = open(name + '.pkl', 'wb')
pkl.dump((params, paramsn, bars, barsn, Momdf, IRFserlist, LinCoeffs, \
          LinCoeffsn), output)
output.close()