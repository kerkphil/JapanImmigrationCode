import pandas_datareader.data as web
import pandas as pd
import datetime
import numpy as np
import statsmodels.api as sm

from DSGEmoments import calcmom

modname = 'JapImm_moms'

def fdfilter(data):
    (nobs, nvar) = data.shape
    return data[1:nobs,:] - data[0:nobs-1,:]


def ltfilter(data):
    (nobs, nvar) = data.shape
    # regressors are a constant and time trend
    X = np.stack((np.ones(nobs), np.linspace(0, nobs-1, nobs)), axis=1)
    Y = data
    # beta = (X'X)^(-1) X'Y
    beta = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), \
                  np.dot(np.transpose(X), Y))
    # fitted values are X*beta
    return Y - np.dot(X, beta)


# set start and end dates
start = datetime.datetime(1994, 1, 1)
end = datetime.datetime(2013, 12, 1)

# get data from FRED, convert to numpy arrays and take logs.  Resample if needed.

# quarterly data
DataQ = web.DataReader(["JPNRGDPEXP", "JPNRGDPPIN", "JPNRGDPPRI", "JPNRGDPPNI", \
    "JPNPFCEQDSNAQ", "BPCFTT01JPQ636N", "JPNRGDPNGS", "JPNGDPDEFQISMEI"] \
    , "fred", start, end)   
DataQ.columns = ['GDP', 'PubInv', 'PriResInv', 'PriNonInv', 'PriCons', \
    'FinAcct', 'NetExp', 'Defl']


# transform data appropriately logs or ratios to GDP, for example
logY = np.log(DataQ.as_matrix(columns=['GDP']))

logC = np.log(DataQ.as_matrix(columns=['PriCons']))

Inv = np.log(DataQ.as_matrix(columns=['PubInv'])) + \
      np.log(DataQ.as_matrix(columns=['PriResInv'])) + \
      np.log(DataQ.as_matrix(columns=['PriNonInv']))
logI = np.log(Inv)

NX_Y = DataQ.as_matrix(columns=['NetExp']) / DataQ.as_matrix(columns=['GDP'])

# convert B from current yen to billions of 2011 yen 
B = (DataQ.as_matrix(columns=['FinAcct']) / 1000000000) / \
    (.984*DataQ.as_matrix(columns=['Defl']))
B_Y = B / DataQ.as_matrix(columns=['GDP'])

Data = np.hstack((logY, logC, logI, NX_Y, B_Y))

varindex = ['logY', 'logC', 'logI', 'NX_Y', 'B_Y']

# use linear trend filter
DataLT = ltfilter(Data)
reportLT, momindex = calcmom(DataLT)

# use FD filter
DataFD = fdfilter(Data)
reportFD, momindex = calcmom(DataFD)

# use HP filter
DataHP, trendHP = sm.tsa.filters.hpfilter(Data, 1600)
reportHP, momindex = calcmom(DataHP)

# use BK filter
DataCF, trendCF = sm.tsa.filters.cffilter(Data)
reportCF, momindex = calcmom(DataCF)

writer = pd.ExcelWriter(modname + '.xlsx')

LTdf = pd.DataFrame(reportLT)
LTdf.columns = varindex
LTdf.index = momindex
LTdf = LTdf.transpose()
print (LTdf.to_latex())
LTdf.to_excel(writer,'LT')

FDdf = pd.DataFrame(reportFD)
FDdf.columns = varindex
FDdf.index = momindex
FDdf = FDdf.transpose()
print (FDdf.to_latex())
FDdf.to_excel(writer,'FD')

HPdf = pd.DataFrame(reportHP)
HPdf.columns = varindex
HPdf.index = momindex
HPdf = HPdf.transpose()
print (HPdf.to_latex())
HPdf.to_excel(writer,'HP')

CFdf = pd.DataFrame(reportCF)
CFdf.columns = varindex
CFdf.index = momindex
CFdf = CFdf.transpose()
print (CFdf.to_latex())
CFdf.to_excel(writer,'CF')

writer.save()