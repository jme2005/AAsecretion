# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 13:31:29 2014

@author: jme2005
"""

import os 
import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

import matplotlib.pyplot as plt


os.chdir("/Users/johanedvinsson/projects/Arun")
from scipy.stats import cumfreq

def ecdfSyt(df,Group,Conc,threshold):
    tmp=df[(df['Conc']==Conc) & (df['Group']==Group)]
    tmp['time']=np.round(tmp['time'],1)    
    tmp=tmp.sort('time')
    nbins=np.unique(tmp['time']).size    
    tmp1=cumfreq(tmp['time'].values,numbins=nbins)[0]
    time=np.unique(tmp['time'])    
    tmparray=np.zeros([nbins,4])    
    DF=pd.DataFrame(tmparray,columns=["Group","Conc","time","cumfreq"])
    DF['Group']=[Group]*nbins; DF['Conc']=[Conc]*nbins; DF['time']=time;DF['cumfreq']=tmp1    
    DF['cumfreq']=DF['cumfreq']/DF['cumfreq'].max()
    DF=DF[DF['cumfreq']>threshold]
    DF['cumfreq']=(DF['cumfreq']/(DF['cumfreq'].max()-DF['cumfreq'].min()))-DF['cumfreq'].min()
    
    DF['time']=DF['time']-DF['time'].min()    
    return DF

def createDF(DF,Group,conc,threshold):
    DFtmp=pd.DataFrame    
    first=True    
    for i in conc:
        tmp=ecdfSyt(DF,Group,i,threshold)
        if first:
            DFtmp=tmp
            first=False
        else:
            DFtmp=DFtmp.append(tmp)
    return DFtmp
        

def initialConditions(params,Cai):
    k_1 = params[1] 
    k2 = params[2]
    k_2 = params[3]
    ksr = params[4]
    krr = params[5]
    Ca = Cai
    Kd = params[6]
    rmax = params[7]
    k1 = (rmax*Ca)/(Ca+Kd)
    Caa = params[8]
    CaS = params[9]
    CaF = params[0]
    CaFa = params[10]
    CaFS = params[11]
    CaFF = params[12]

    M=np.matrix([-(k1*CaFa*Ca**Caa),k1*CaFa*Ca**Caa,0,1,k_1,-(k_1+ksr*CaFS*Ca**CaS+k2),k2,1,0,k_2,-(krr*CaFF*Ca**CaF+k_2),1]).reshape(3,4)
    dS=np.array([0,0,0,1])
    try:    
        Minv=np.linalg.pinv(M)
    except:
        return [1.0, 1.5318878112344034e-17, 1.2786737572473748e-16]
    yinitial=dS*Minv 
    yinitial = yinitial.tolist()
    return yinitial[0]

def solveequation(params,times,Cai):
    timepoints = np.where(times == 0)
    yinitial = initialConditions(params,Cai)
    yinitial.append(0)
    yinitial.append(0)
    Ca = 5e-6
    k_1 = params[1] 
    k2 = params[2]
    k_2 = params[3]
    ksr = params[4]
    krr=params[5]
    Kd=params[6]
    rmax=params[7]
    k1=(rmax*Ca)/(Ca+Kd)
    Caa=params[8]
    CaS=params[9]
    CaF=params[0]
    CaFa = params[10]
    CaFS = params[11]
    CaFF = params[12]
    
    def f(yinitial,times):
        A = yinitial[0]
        B = yinitial[1]
        C = yinitial[2]
        D = yinitial[3]
        E = yinitial[4]
        
        dA = -k1*CaFa*Ca**Caa*A + k_1*B
        dB = -(k_1+CaFS*Ca**CaS*ksr+k2)*B+A*k1*CaFa*Ca**Caa+C*k_2
        dC = -(krr*CaFa*Ca**CaF+k_2)*C + B*k2
        dD = B*ksr*CaFS*Ca**CaS
        dE = C*krr*CaFF*Ca**CaF
        return [dA,dB,dC,dD,dE]
        
    out = np.array(odeint(f,yinitial,times[timepoints[0][0]:timepoints[0][1]]))
    output = out[:,3]+out[:,4]
    
    Ca = 10e-6;
    k1 =(rmax*Ca)/(Ca+Kd)
    
    out = np.array(odeint(f,yinitial,times[timepoints[0][1]:timepoints[0][2]]))
    output = np.append(output,(out[:,3]+out[:,4]))
    
    Ca = 30e-6;
    k1 =(rmax*Ca)/(Ca+Kd)
    
    out = np.array(odeint(f,yinitial,times[timepoints[0][2]:timepoints[0][3]]))
    output = np.append(output,(out[:,3]+out[:,4]))
    
    Ca = 100e-6;
    k1 =(rmax*Ca)/(Ca+Kd)
    
    out = np.array(odeint(f,yinitial,times[timepoints[0][3]:]))
    output = np.append(output,(out[:,3]+out[:,4]))
    
    return output

def CostFunction(params):
    if any( x < 0 for x in params):
        return 100
    target=digitsyt7['target'].values
    model = solveequation(params,digitsyt7['time'],1e-9)
    RMSE = np.sqrt((target-model)**2)
    return RMSE.mean()

def sampleInit(n):
    seed=[]
    fitvalue=[]
    for i in n:
        np.random.seed(i)
        x = np.random.uniform(-3,3,13)
        params = 10**x
        
        fit = minimize(CostFunction,params)
        print fit.fun,i        
        fitvalue.append(fit.fun)
        seed.append(i)
        output = zip(seed,fitvalue)

    return output

# seed 2 work well

def main():
    digit = pd.read_csv('latencies.csv')
    digit100 = pd.read_csv('100uMLat.csv')
    newName = digit.columns.values
    newName[2] = 'time'
    digit.columns = newName
    digit=digit.append(digit100)
    digitsyt1 = digit[digit['Group']=='Syt-1']
    digitsyt7 = digit[digit['Group']=='Syt-7']
    
    digitsyt1 = createDF(digitsyt1,'Syt-1',[5,10,30,100],0.08)
    digitsyt7 = createDF(digitsyt7,'Syt-7',[5,10,30,100],0.08)
    DR = pd.read_csv('DigitoninDR.csv')
    tmp=DR[['Conc','Syt1']]
    tmp.columns=['Conc','Dose']  
    digitsyt1 = pd.merge(digitsyt1,tmp, how='inner')
    tmp=DR[['Conc','Syt7']]
    tmp.columns=['Conc','Dose']
    digitsyt7 = pd.merge(digitsyt7,tmp,how='inner')
    digitsyt1['target']=digitsyt1['cumfreq']*digitsyt1['Dose']
    digitsyt7['target']=digitsyt7['cumfreq']*digitsyt7['Dose']
        
    return(digitsyt1,digitsyt7)



