# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:58:32 2019

@author: jpristovnik
"""

import numpy as np
import pandas as pd
import math

class ImportSID_PD:
    def __init__(self, path):
        self.masterscale = gen_masterscale()
        self.lamdas = gen_lamdas_y0(path)[0]
        self.y0 = gen_lamdas_y0(path)[1]
        self.lm = gen_lamdas_y0(path)[2]
        self.isum = gen_lamdas_y0(path)[3]

def logit (score):

    return math.exp(score)/(1+math.exp(score))

        

def gen_masterscale():
    #masterscale
    masterscale = pd.DataFrame({'rating': ['AAA','AA+','AA','AA-','A+','A','A-','BBB+','BBB','BBB-','BB+','BB','BB-','B+','B','B-','C+','C'],'grade':range(1,19), \
    
                                'RW' : [0.2,0.2,0.2,0.2,0.5,0.5,0.5,1,1,1,1,1,1,1.5,1.5,1.5,1.5,1.5], 'cor': [0.2,0.2,0.2,0.2,0.5,0.5,0.5,1,1,1,1,1,1,1.5,1.5,1.5,1.5,1.5]})
    
    #masterscale = pd.DataFrame({'rating': ['AAA','AA+','AA','AA-','A+','A','A-','BBB+','BBB','BBB-','BB+','BB','BB-','B+','B','B-','CCC+','CCC','CCC-','CC','C','D'],'grade':range(1,23)})
    
    masterscale['PD_low']= (masterscale['grade']-0.5)*0.6113-12.247
    
    masterscale['PD_high']= (masterscale['grade']+0.5)*0.6113-12.247
    
    masterscale['PD_low']=masterscale['PD_low'].apply(logit)
    
    masterscale['PD_high']=masterscale['PD_high'].apply(logit)
    
    masterscale.loc[masterscale['rating']=='C','PD_high']=1
    
    masterscale.index = pd.IntervalIndex.from_arrays(masterscale['PD_low'],masterscale['PD_high'],closed='right')
    
    def coc_SID(grade):
    
        sid = 0.2
    
        i = masterscale.index[masterscale['grade']==grade].tolist()[0]
    
        rw = masterscale['RW'][i]
    
        coc = 0.606*sid*0.08*0.1 + (1-0.606)*rw*0.08*0.1

        return coc

 
    masterscale['coc'] = masterscale['grade'].apply(coc_SID)
      
    def coc_brez(grade):
    
        i = masterscale.index[masterscale['grade']==grade].tolist()[0]
    
        rw = masterscale['RW'][i]
    
        coc = rw*0.08*0.1
    
        return coc
  
    masterscale['coc_brez'] = masterscale['grade'].apply(coc_brez)

    return masterscale


def gen_lamdas_y0(path):
    SIDk= pd.read_excel(path + "PDji_SID_kalibracija.xlsx",sheet_name="Long") #Tabela verjetnosti propada. stolpci bonitete, vrstice leta
    SID2=(1-SIDk).cumprod() #tabela verjetnosti prezivetja n let. S iz worda
    
    SID3 = SID2.shift(1) #S(i+1)
    SID3.iloc[0]=1
    
    lm=SID2.copy()
    y0=SID2.copy()
    integ=SID2.copy()
    
    ints = 1.0*(SID2.index.values-1.0) #array float indeksev 0 - 19 zadnji je neskoncno
    ints=np.append(ints,math.inf)
    
    for s in SIDk.columns:
        ys=SID2[s].apply(math.log) # logaritmirani stolpci prezivetji
        lm[s]=-SID2[s].apply(math.log)+SID3[s].apply(math.log) #dobimo lambde za 20 let za vsako boniteto
        y0[s]=SID2[s]/(SID2.index.values*(-lm[s])).apply(math.exp) #dobimo y0 iz worda le da je v wordu napaka. v stevcu je S(i,b)
        #lines[s], = ax.plot(SIDk.index, SID2[s], linewidth=2)
    
    #v tem bloku naredimo leta intervalska 
    lm.index = pd.IntervalIndex.from_breaks(ints,closed='left')
    y0.index = pd.IntervalIndex.from_breaks(ints,closed='left')
    integ.index = pd.IntervalIndex.from_breaks(ints,closed='left')
    
    
    for s in SIDk.columns:
        integ[s]=-y0[s]*((-lm[s]*lm.index.right.values).apply(math.exp)-(-lm[s]*lm.index.left.values).apply(math.exp))/lm[s] #integral
    
    
    #pogledamo kaksen delez ploscine predstavlja ploscina DO itega leta. intinv je zamaknjen. 1. vrstica 0 21. torej 1. indeksi spet tockovni in ne intervalski
    isum=integ.sum()
    intinv=(integ/integ.sum()).cumsum()
    intinv=intinv.reset_index(drop=True)
    intinv = intinv.append(intinv.head(1)).shift(1)
    intinv=intinv.reset_index(drop=True)
    intinv[0:1]=0
    #kreiramo indekse
    lamdas = {}
    y0dict = {}
    
    #lambde in y0 damo v slovar. kljuc boniteta, value seznam lambd. potem indekse naredimo intevalske in normirane, meje so delezi ploscin iz intinv
    for s in SIDk.columns:
        lamdas[s]=lm[s].copy()
        lamdas[s].index=pd.IntervalIndex.from_breaks(intinv[s],closed='left')
        y0dict[s]=y0[s].copy()
        y0dict[s].index=pd.IntervalIndex.from_breaks(intinv[s],closed='left')
    
    return (lamdas, y0dict, lm, isum)
            