#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This version Mar 17, 2017
Written by Kerk Phillips

Prints plots for four different scenarios
"""

import matplotlib.pyplot as plt
import pickle

(params, paramsUN, bars, barsUN, Momdf, IRFserlistUN, LinCoeffs, \
          LinCoeffsUN) = \
    pickle.load( open( 'UN.pkl', 'rb' ) )
        
(params, paramsPR, bars, barsPR, Momdf, IRFserlistPR, LinCoeffs, \
          LinCoeffsPR) = \
    pickle.load( open( 'PR.pkl', 'rb' ) )
        
(params, paramsEQ, bars, barsEQ, Momdf, IRFserlistEQ, LinCoeffs, \
          LinCoeffsEQ) = \
    pickle.load( open( 'EQ.pkl', 'rb' ) )
    
(params, paramsSK, bars, barsSK, Momdf, IRFserlistSK, LinCoeffs, \
          LinCoeffsSK) = \
    pickle.load( open( 'SK.pkl', 'rb' ) )
    
(zUN, KSDUN, KUDUN, BSDUN, BUDUN, KUN, BUN, NSUN, NUUN, WUN, GDPUN, sSUN, \
    sUUN, EXUN, rUN, wSUN, wUUN, CSDUN, CUDUN, CSIUN, CUIUN, USDUN, UUDUN, \
    USIUN, UUIUN, IUN, CUN) = IRFserlistUN
 
(zPR, KSDPR, KUDPR, BSDPR, BUDPR, KPR, BPR, NSPR, NUPR, WPR, GDPPR, sSPR, \
    sUPR, EXPR, rPR, wSPR, wUPR, CSDPR, CUDPR, CSIPR, CUIPR, USDPR, UUDPR, \
    USIPR, UUIPR, IPR, CPR) = IRFserlistPR
 
(zEQ, KSDEQ, KUDEQ, BSDEQ, BUDEQ, KEQ, BEQ, NSEQ, NUEQ, WEQ, GDPEQ, sSEQ, \
    sUEQ, EXEQ, rEQ, wSEQ, wUEQ, CSDEQ, CUDEQ, CSIEQ, CUIEQ, USDEQ, UUDEQ, \
    USIEQ, UUIEQ, IEQ, CEQ) = IRFserlistEQ
 
(zSK, KSDSK, KUDSK, BSDSK, BUDSK, KSK, BSK, NSSK, NUSK, WSK, GDPSK, sSSK, \
    sUSK, EXSK, rSK, wSSK, wUSK, CSDSK, CUDSK, CSISK, CUISK, USDSK, UUDSK, \
    USISK, UUISK, ISK, CSK) = IRFserlistSK
 
[KSDbar, KUDbar, BSDbar, BUDbar, \
    Kbar, Bbar, NSbar, NUbar, Wbar, GDPbar, sSbar, sUbar, EXbar, rbar, wSbar, \
    wUbar,  CSDbar,  CUDbar, CSIbar, CUIbar, USDbar, UUDbar, USIbar, UUIbar, \
    Ibar, Cbar] = bars
 
nobs = zUN.size
time = range(0, nobs-1)

plt.figure()
plt.subplot(2,2,1)
plt.plot(time, GDPUN[0:nobs-1]/GDPbar, label='UN')
plt.plot(time, GDPPR[0:nobs-1]/GDPbar, label='PR')
plt.plot(time, GDPEQ[0:nobs-1]/GDPbar, label='EQ')
plt.plot(time, GDPSK[0:nobs-1]/GDPbar, label='SK')
plt.title('GDP')
plt.xticks([])
plt.subplot(2,2,2)
plt.plot(time, CUN[0:nobs-1]/Cbar, label='UN')
plt.plot(time, CPR[0:nobs-1]/Cbar, label='PR')
plt.plot(time, CEQ[0:nobs-1]/Cbar, label='EQ')
plt.plot(time, CSK[0:nobs-1]/Cbar, label='SK')
plt.title('Consumption')
plt.xticks([])
plt.subplot(2,2,3)
plt.plot(time, IUN[0:nobs-1]/Ibar, label='UN')
plt.plot(time, IPR[0:nobs-1]/Ibar, label='PR')
plt.plot(time, IEQ[0:nobs-1]/Ibar, label='EQ')
plt.plot(time, ISK[0:nobs-1]/Ibar, label='SK')
plt.title('Investment')
plt.subplot(2,2,4)
plt.plot(time, BUN[0:nobs-1], label='UN')
plt.plot(time, BPR[0:nobs-1], label='PR')
plt.plot(time, BEQ[0:nobs-1], label='EQ')
plt.plot(time, BSK[0:nobs-1], label='SK')
plt.title('International Savings')

plt.savefig('JapImmAgg.pdf', format='pdf', dpi=2000)
plt.show()

plt.figure()
plt.subplot(2,2,1)
plt.plot(time, wSUN[0:nobs-1]/wSbar, label='UN')
plt.plot(time, wSPR[0:nobs-1]/wSbar, label='PR')
plt.plot(time, wSEQ[0:nobs-1]/wSbar, label='EQ')
plt.plot(time, wSSK[0:nobs-1]/wSbar, label='SK')
plt.title('Skilled Wage')
plt.xticks([])
plt.subplot(2,2,2)
plt.plot(time, wUUN[0:nobs-1]/wUbar, label='UN')
plt.plot(time, wUPR[0:nobs-1]/wUbar, label='PR')
plt.plot(time, wUEQ[0:nobs-1]/wUbar, label='EQ')
plt.plot(time, wUSK[0:nobs-1]/wUbar, label='SK')
plt.title('Unskilled Wage')
plt.xticks([])
plt.subplot(2,2,3)
plt.plot(time, sSUN[0:nobs-1]/sSbar, label='UN')
plt.plot(time, sSPR[0:nobs-1]/sSbar, label='PR')
plt.plot(time, sSEQ[0:nobs-1]/sSbar, label='EQ')
plt.plot(time, sSSK[0:nobs-1]/sSbar, label='SK')
plt.title('Skilled Interest Rate')
plt.subplot(2,2,4)
plt.plot(time, sUUN[0:nobs-1]/sUbar, label='UN')
plt.plot(time, sUPR[0:nobs-1]/sUbar, label='PR')
plt.plot(time, sUEQ[0:nobs-1]/sUbar, label='EQ')
plt.plot(time, sUSK[0:nobs-1]/sUbar, label='SK')
plt.title('Unskilled Interest Rate')
plt.savefig('JapImmWages.pdf', format='pdf', dpi=2000)
plt.show()
 
plt.figure()
plt.subplot(2,2,1)
plt.plot(time, USDUN[0:nobs-1]/USDbar, label='UN')
plt.plot(time, USDPR[0:nobs-1]/USDbar, label='PR')
plt.plot(time, USDEQ[0:nobs-1]/USDbar, label='EQ')
plt.plot(time, USDSK[0:nobs-1]/USDbar, label='SK')
plt.title('Skilled Domestic')
plt.xticks([])
plt.subplot(2,2,2)
plt.plot(time, UUDUN[0:nobs-1]/UUDbar, label='UN')
plt.plot(time, UUDPR[0:nobs-1]/UUDbar, label='PR')
plt.plot(time, UUDEQ[0:nobs-1]/UUDbar, label='EQ')
plt.plot(time, UUDSK[0:nobs-1]/UUDbar, label='SK')
plt.title('Unskilled Domestic')
plt.xticks([])
plt.subplot(2,2,3)
plt.plot(time, USIUN[0:nobs-1]/USIbar, label='UN')
plt.plot(time, USIPR[0:nobs-1]/USIbar, label='PR')
plt.plot(time, USIEQ[0:nobs-1]/USIbar, label='EQ')
plt.plot(time, USISK[0:nobs-1]/USIbar, label='SK')
plt.title('Skilled Immigrant')
plt.subplot(2,2,4)
plt.plot(time, UUIUN[0:nobs-1]/UUIbar, label='UN')
plt.plot(time, UUIPR[0:nobs-1]/UUIbar, label='PR')
plt.plot(time, UUIEQ[0:nobs-1]/UUIbar, label='EQ')
plt.plot(time, UUISK[0:nobs-1]/UUIbar, label='SK')
plt.title('Unskilled Immigrant')
plt.savefig('JapImmUtil.pdf', format='pdf', dpi=2000)
plt.show()