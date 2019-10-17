
import numpy as np
import pandas as pd
import math

path="C:\\Users\\lhrobat\\Documents\\PROJECTS\\SID\\"

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