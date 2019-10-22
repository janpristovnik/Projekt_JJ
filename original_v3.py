# -*- coding: utf-8 -*-

"""

Created on Fri Oct 12 11:06:48 2018

 

@author: cesibasslovenia

"""

 

# -*- coding: utf-8 -*-

"""

Created on Mon Oct  8 11:23:19 2018

 

@author: cesibasinfo

"""

 

# -*- coding: utf-8 -*-

"""

Created on Tue Mar 13 06:23:35 2018

 

@author: jkodre

"""

import time

import datetime as dt

import math

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

import pylab as pyl

# import multiprocessing as mp

 

def logit (score):

    return math.exp(score)/(1+math.exp(score))

 

def yeartomonth(pd):

    return 1-math.pow(1-pd,1/12)

 

def survtime(sss,a):

    #print(sss,a,sep=' ')

    idx=lamdas[sss].index.get_loc(a)

    ld=lamdas[sss].iloc[idx]

    yy0=y0[sss].iloc[idx]

    s1=lamdas[sss].index[idx].left

    x1=lm[sss].index[idx].left

    x= -math.log(-(a-s1)*ld*isum[sss]/yy0+math.exp(-ld*x1))/ld

    return x

 

def coc_SID(grade):

    sid = 0.2

    i = masterscale.index[masterscale['grade']==grade].tolist()[0]

    rw = masterscale['RW'][i]

    coc = 0.606*sid*0.08*0.1 + (1-0.606)*rw*0.08*0.1

    return coc

 

def coc_brez(grade):

    i = masterscale.index[masterscale['grade']==grade].tolist()[0]

    rw = masterscale['RW'][i]

    coc = rw*0.08*0.1

    return coc

 

 

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

masterscale['coc'] = masterscale['grade'].apply(coc_SID)

masterscale['coc_brez'] = masterscale['grade'].apply(coc_brez)

 

#SID kalibracija

 

path="C:\\Users\\jrems\\Documents\\SID\\"

 

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

 

 

 

amt_max = 10000000

amt_min = 30000

porlen = math.log(amt_max)*(1-amt_min/amt_max)*amt_max/amt_min

 

#low_mora=0

Nscen=100

slength=1600 #Število izdanih kreditov

 

total_months=44

LGD = 0.5

amt_placed_0_t= 98600000

amt_placed_0=0

#first_loss=0.3*amt_placed_0

credit_loss_ratio = 0.625

#portfolio_loss_ratio = 0.3 #po potrebi se ga doloci drugace. 0.5 je provizoricno

N_FPs=5

eff_s=0.65

effic=1-math.pow(1-eff_s,1/12)

vzvod=1

loss_limit = 61625000 # amt_placed_0_t * 62,5%

 

# porazdelitev rocnosti

months_l=36   # Se ne uporablja. glej spodaj porazdelitev

months_h=120  # Se ne uporablja. glej spodaj porazdelitev

 

rocnost_p = [0.0051,0.0308,0.0564,0.0821,0.1077,0.1077,0.1077,0.1077,0.1077,0.0906,0.0735,0.0564,0.0393,0.0222,0.0051]

rocnost_r = [36,42,48,54,60,66,72,78,84,90,96,102,108,114,120]

 

 

#tranches=[0.379,0.207,0.207,0.207,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

effs={0:1,12:1}

tranches={0:0.521,12:0.154,36:0.154,48:0.171} #nacin crpanja sredstev s strani FP

#targets={0:2993750,       3:7050000,   6:6406250,   9:4912500,          12:1287500, 15:1287500, 18:1287500, 21:1287500, 24:3568750,          27:3568750, 30:3568750, 33:1287500, 36:2493750, 39:2493750,          42:2493750, 45:1318750, 48:2118750, 51:2118750, 54:1200000,          57:1200000, 60:431250}

c_target=0

 

#lm = 0.25

 

PD_M=0.30/12

 

#LGD beta distribution

LGD_mu=0.60

LGD_max=1

LGD_sig=0.2

#LGD_a = (LGD_mu/LGD_max)* (LGD_mu*(LGD_max-LGD_mu)/(LGD_max*LGD_sig*LGD_sig)-1)

#LGD_b = LGD_a* ((LGD_max/LGD_mu)-1)

LGD_a = LGD_mu*LGD_mu *(1-LGD_mu)/(LGD_sig*LGD_sig) - LGD_mu

LGD_b = LGD_a *(1/LGD_mu - 1)

for bb in reversed(masterscale['rating'].index[-1:]):

    boni = masterscale['rating'][bb]

    print('Calculating: ',boni)

    lpercs= []

    lffp = []

    counter=0

    total_losses = []

    total_places = []

    lpercs_bank = []

    casi_total =[]

    zlom = 1

    cor = []

    cort = []

    #casi_percents =[]

    for ii in range (0,Nscen):

        amt_placed_0 = 0

        amt_placed=amt_placed_0

        total_placed=0

        tt1 = time.time()

   

        df = pd.DataFrame({'nm': pd.Series(np.random.random_sample(size=slength)),\

                           # 'rocnost': pd.Series(np.random.randint(low = months_l, high=months_h, size=slength)),\

                           'rocnost': pd.Series(np.random.choice(rocnost_r, size=slength, replace=True, p=rocnost_p)),\

                              

                           #-3.2, 0.7 28%

                           #-5.2, 1.0 obicajna

                           'PD': pd.Series(-3.2+(1.0)*np.random.randn(slength)).apply(logit),\

                            #'PD': pd.Series(-2.0+(1.5)*np.random.randn(slength)).apply(logit),\

                            #'PD': pd.Series(-1.5+(0.7)*np.random.randn(slength)).apply(logit),\

                           #'PD': pd.Series(-5.2+(1.0)*np.random.randn(slength)).apply(logit),\

                            #'LGD': pd.Series(0.7+(0.15)*np.random.randn(slength))

                            's_uni': pd.Series(np.random.random_sample(size=slength)),\

                            'mora': pd.Series(np.random.random_sample(size=slength)),\

                           'LGD': pd.Series(np.random.beta(a=LGD_a, b=LGD_b, size=slength))

                           })

   

        #df['rating'] = df['PD'].apply(lambda x : masterscale.iloc[masterscale.index.get_loc(x)]['rating'])

        df['grade'] = df['PD'].apply(lambda x : masterscale.iloc[masterscale.index.get_loc(x)]['grade'])

        df['EKP_available']=0

        df['rating'] = df['PD'].apply(lambda x : masterscale.iloc[masterscale.index.get_loc(x)]['rating'])

        #df['rating'] = boni

       

        

        df['vzvod']=vzvod

        df['mxnomin']=amt_max

        df['nm']=df['nm']*(porlen-1)+1

        df['nomin']=df.apply(lambda por: por['mxnomin']*math.log(por['mxnomin'])/(por['nm']+math.log(por['mxnomin'])),axis=1)

        df['SIDnomin']=df['nomin']/df['vzvod']

        #df['survtime']= df.apply(lambda row:  math.log(1-row['s_uni'])/(math.log(-row['PD']+1)),axis=1)

        df['survtime']= df.apply(lambda row:  survtime(sss=row['rating'],a=row['s_uni']),axis=1)

       

        

        #df['PDy'] = math.pow(df['PD'],12)

        df.loc[df['LGD']<0,'LGD']=0.05

        df.loc[df['PD']<0,'PD']=0.003

        df.loc[df['LGD']>1,'LGD']=1

        df.loc[df['PD']>1,'PD']=1

       

        df['min_mora'] = round(df['rocnost']/3) #dodano

        df['max_mora'] = round(df['rocnost']/2)

        #slength = len(df['nomin'])

        df = df.assign(mora=pd.Series(np.random.random_sample(slength)).values)

        df['mora']=round(df['mora']*(df['max_mora']-df['min_mora'])+df['min_mora']) #dodan min mora v pravi obliki

        df['annuity']=df['nomin']/(df['rocnost']-df['mora'])

        df['cs_nomin']=np.cumsum(df['nomin'])

        #df['LGD']=LGD

       

        df['start_month']=-1

        df['ziv']=0

        #df['moratorium_used']=0

        df['loss']=0

        df['default_month']=-1

        df['not_settled']=0

        df['repay']=0

        df['repay_sid']=0

        #df['repay_am']=0

        df['recovery']=0

        df['loss_month']=-1

        df['wo_duration']=36

        df['loss_bank']= 0 # izguba ki je SID ne krije

        df['loss_SID'] = 0 # izguba ki jo ima SID

        df['accum_loss_bank'] = 0

        df['accum_loss_SID'] = 0

        df['enke'] = 1

       #df['covered_ratio'] = 1 # binaren, 0 ce izguba ki jo krije SID vkljucno do konkretnega kredita presega x% celotne vrednosti portfelja do zdaj

        df['covered'] = 1

        df['accum_credit'] = 0

        df['repay_bank'] = 0

        df['coc'] = 0

        df['obd']=0

        # binaren, 0 ce izguba ki jo krije SID vkljucno do konkretnega kredita presega razpisan loss_limit

        total_loss=0

        cnt_zivih=1

        i=0

        amt_placed_0=0

        #df = df.assign(izid=pd.Series(np.random.random_sample(slength)).values)

       

        while cnt_zivih>0:

            #T0: tranches

            #if i in tranches:

            #    amt_placed_0 += tranches[i]*amt_placed_0_t

            #    amt_placed += tranches[i]*amt_placed_0_t

 

# =============================================================================

#             if i in targets:

#                 c_target = targets[i]

#

#             amt_placed_0 += c_target/3.0

#             amt_placed += c_target/3.0

# =============================================================================

            amt_placed_0 += amt_placed_0_t/total_months

            amt_placed += amt_placed_0_t/total_months

 

            if i in effs:

                eff_s=effs[i]

                effic=1-math.pow(1-eff_s,1/12)

               

           #T1: place the credits  

            if i<total_months:

                #df.loc[(df['start_month']==-1) & (df['cs_nomin']<amt_placed*df['vzvod']*effic),'ziv']=1

                #df.loc[(df['start_month']==-1) & (df['cs_nomin']<amt_placed*df['vzvod']*effic),'not_settled']=1

                #df.loc[(df['start_month']==-1) & (df['cs_nomin']<amt_placed*df['vzvod']*effic),'start_month']=i

                df.loc[(df['start_month']==-1) & (df['cs_nomin']<total_placed+(amt_placed*df['vzvod']-total_placed)*effic),'ziv']=1

                df.loc[(df['start_month']==-1) & (df['cs_nomin']<total_placed+(amt_placed*df['vzvod']-total_placed)*effic),'not_settled']=1

                df.loc[(df['start_month']==-1) & (df['cs_nomin']<total_placed+(amt_placed*df['vzvod']-total_placed)*effic),'EKP_available']=amt_placed_0

                df.loc[(df['start_month']==-1) & (df['cs_nomin']<total_placed+(amt_placed*df['vzvod']-total_placed)*effic),'start_month']=i

           

            #T2: we check if the loan is to be repaid and can be repaid

            #df.loc[(df['mora']+df['start_month']<i) & (df['rocnost']+df['start_month']>=i) & (df['izid']>=df['PD']) & (df['ziv']==1) & (df['default_month']==-1),'repay']+=df['annuity']

            #df.loc[(df['mora']+df['start_month']<i) & (df['rocnost']+df['start_month']>=i) & (df['izid']<df['PD']) & (df['ziv']==1) & (df['default_month']==-1),'default_month']=i;

            df.loc[(df['mora']+df['start_month']<i) & (df['rocnost']+df['start_month']>=i) & (df['survtime']*12.0+df['start_month']>=i) & (df['ziv']==1) & (df['default_month']==-1),'repay']+=df['annuity']

            df.loc[(df['mora']+df['start_month']<i) & (df['rocnost']+df['start_month']>=i) & (df['survtime']*12.0+df['start_month']>=i) & (df['ziv']==1) & (df['default_month']==-1),'repay_sid']+=df['annuity']/df['vzvod']

            df.loc[(df['mora']+df['start_month']<i) & (df['rocnost']+df['start_month']>=i) & (df['survtime']*12.0+df['start_month']<i)  & (df['ziv']==1) & (df['default_month']==-1),'default_month']=i

            df.loc[(df['mora']+df['start_month']<i) & (df['rocnost']+df['start_month']>=i) & (df['ziv']==1) & (df['default_month']>-1),'loss']+=df['annuity']

           

            #T3: on maturity we check the balance

            df.loc[(df['rocnost']+df['start_month']==i) & (df['ziv']==1) &(df['default_month']==-1),'not_settled']=0

            #df.loc[(df['rocnost']+df['start_month']==i) & (df['ziv']==1) &(df['default_month']==-1),'cas_poravnave']=i

            #df.loc[(df['rocnost']+df['start_month']==i) & (df['ziv']==1),'repay_sid']=df['repay_sid']+df['repay']/df['vzvod']

            df.loc[(df['rocnost']+df['start_month']==i) & (df['ziv']==1),'ziv']=0

           

            #T4: on writeoff we get partially repaid loss

            df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1),'not_settled']= 0

            df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1),'accum_credit']= total_placed

            df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1),'obd']=df['repay_sid']

            df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1),'repay_sid']+=df['loss']*(1-df['LGD'])

            df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1),'loss']=df['loss']*df['LGD']

            df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1),'loss_month']=i

            df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1) & (df['loss'] >= (df['nomin']-df['obd'])*credit_loss_ratio) ,'loss_bank']=df['loss'] - (df['nomin']-df['obd'])*credit_loss_ratio

            #df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1) & (df['accum_loss_SID'] >= amt_placed_0_t*portfolio_loss_ratio) ,'loss_bank']=df['loss']

           # df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1) & (df['loss'] >= df['nomin']*credit_loss_ratio) & ((df['accum_loss_SID'] > (amt_placed_0_t *vzvod* portfolio_loss_ratio)) | (df['accum_loss_SID'] > loss_limit)) ,'accum_loss_bank'] += df['loss']

            df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1) ,'loss_SID'] =df['loss']

            df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1) & (df['loss'] >= (df['nomin']-df['obd'])*credit_loss_ratio) ,'loss_SID']= (df['nomin']-df['obd'])*credit_loss_ratio

            #df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1) & (df['loss'] >=  amt_placed_0_t*portfolio_loss_ratio) ,'loss_SID']= 0

            df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1),'repay_bank'] =df[['loss_bank','repay_sid']].min(axis=1)

            df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1),'loss_bank'] -= df['repay_bank']

            df.loc[(df['rocnost']+df['start_month']+df['wo_duration']==i) & (df['default_month']>-1),'loss_SID'] -=  df['loss']*(1-df['LGD']) - df['repay_bank']

 

           

            amt_placed= amt_placed_0#+df['repay_sid'].sum()

            i=i+1

            cnt_zivih = df['ziv'].sum() + df['not_settled'].sum()

   

            total_placed = df[df['start_month']>-1]['cs_nomin'].max()

            df = df.sort_values(by = ['loss_month'])

            df['accum_loss_SID'] = np.cumsum(df['loss_SID'])

            df = df.sort_index()

           

        

        df = df[df['start_month']>-1]

        t_months=df['rocnost'].sum()

 

        print(i,' slices ',len(df['ziv']))

    #    losses=df[df['loss']>0][['nomin','rocnost','max_mora','loss','loss_month']].sort_values('loss_month')

    #    losses['cs_loss']=np.cumsum(losses['loss'])

    #    losses.loc[losses['cs_loss']>first_loss,'FL_breach']=1;

# =============================================================================

#         df.loc[df['nomin']*credit_loss_ratio <= df['loss'],'loss_per_credit'] = df['loss'] - df['nomin']*credit_loss_ratio

#         df = df.sort_values(by = ['loss_month'])

#         df['accum_loss'] = np.cumsum(df['loss'])

#         df['accum_loss_per_credit'] = np.cumsum(df['loss_per_credit'])

#         df['accum_loss_SID'] = df['accum_loss'] - df['accum_loss_per_credit']

#         df.loc[df['accum_loss_SID'] >= (df['cs_nomin'] * portfolio_loss_ratio), 'covered_ratio'] = 0

#         df.loc[df['accum_loss_SID'] >= loss_limit, 'covered_total'] = 0

#         df['covered'] = df['covered_ratio'] * df['covered_total'] #logicni, covered_ratio & covered_total

#         total_loss = df['loss'].sum()

#         df = df.sort_index()

# =============================================================================

       

        df = df.sort_values(by = ['loss_month'])

        df['accum_loss_SID'] = np.cumsum(df['loss_SID'])

        df.loc[df['accum_loss_SID']>loss_limit,'covered'] = 0

        index_total = np.inner(df['covered'], df['enke'])

        if index_total == len(df['enke']):

            if df.shape[2] == 1:

                cas_total = df.iloc[[0]]

            else:

                cas_total = df.iloc[[-1]]

        else:

            cas_total = df.iloc[[index_total]]

       

        cas_total = cas_total.iloc[0]['loss_month']

        casi_total.append(cas_total)

        #df.loc[df['accum_loss_SID']>portfolio_loss_ratio*df['accum_credit'],'covered'] = 0

        #index_percent = np.inner(df['covered'], df['enke'])

        #if index_percent == index_total:

            #cas_total = df.iloc[[len(df['enke']) -1]]

            #cas_total = cas_total.iloc[0]['loss_month']

        #else:

            #cas_total = df.iloc[[index_percent]]

           #cas_total = cas_total.iloc[0]['loss_month']

        #casi_percents.append(cas_total)

        df['loss_bank_popravljen'] = df['loss_bank'] +(df['enke'] - df['covered'])*df['loss_SID']

        df['loss_SID'] = df['loss_SID']*df['covered']

        df['accum_loss_SID'] = np.cumsum(df['loss_SID'])

        df['accum_loss_bank'] = np.cumsum(df['loss_bank'])

        df['accum_loss_bank_pop'] = np.cumsum(df['loss_bank_popravljen'])

        df['cs_nomin_loss'] = np.cumsum(df['nomin'])

        #imax = df['accum_loss_SID'].idxmax()

        #df['accum_loss_SID'][imax:] = df['accum_loss_SID'].max()

        df = df.sort_index()

        #meja 50% na celotni portfel se poracuna glede na vse izdane kredite po koncu simulacije.

       

        total_loss = df['loss_SID'].sum()

        total_losses.append(total_loss)

        #fl_month = losses[losses['FL_breach']==1]['loss_month'].min()

        balance = amt_placed_0-total_loss

       

        l_FP = df['accum_loss_bank_pop'].max()

        lffp.append(l_FP)

 

       

        total_placed = df[df['start_month']>-1]['cs_nomin'].max()

        total_places.append(total_placed)

        loss_perc=(100.0*(total_loss+l_FP)/total_placed)

        lpercs.append(loss_perc)

        loss_perc_bank = (100.0*(l_FP)/total_placed)

        lpercs_bank.append(loss_perc_bank)

        #poroc = t_months/(len(df['ziv'])*12)

        if len(df['ziv'])==0:

            poroc = 0

        else:

            poroc = t_months/(len(df['ziv'])*12)

        lFPperc = l_FP/total_placed

        cor1 = lFPperc/poroc

        cor.append(cor1) 

        

        cor_total = loss_perc/(poroc*100.0)

        cort.append(cor_total)

       

        

 

       

# =============================================================================

#         if loss_perc>30:

#             l_FP = 0.01*(loss_perc-30.0)*amt_placed_0/(total_placed*poroc)

#         else:

#             l_FP=0

# =============================================================================

       

        

        print('Time for 1 simulation ',time.time()-tt1, ' seconds')

        if ii % 10==0:

            print(ii)

        zlom = zlom * np.prod(df['covered'])

       

        

        #dodatek

        for b in reversed(masterscale['rating'].index):

            j=0 #stevilo bonitet

            sum=0

            tmp = df.loc[df['rating']==b ]

            tmp_cor = tmp['accum_loss_bank_pop'].max()

            tmp_cor = tmp_cor/tmp[tmp['start_month']>-1]['cs_nomin'].max()

            if len(tmp['ziv'])==0:

                poroc_t = 0

            else:

                poroc_t = t_months/(len(tmp['ziv'])*12)

                tmp_cor= tmp_cor/poroc_t # LH_20191014

           

            tmp_sid = np.cumsum(tmp['loss_SID'])

            tmp_sid = tmp_sid/tmp[tmp['start_month']>-1]['cs_nomin'].max()

            tmp_sid = tmp_sid/poroc_t

            tmp_sid = tmp_sid + tmp_cor

            if ii ==0:

                masterscale.loc[masterscale['rating']==b, 'cor']= tmp_cor

                masterscale.loc[masterscale['rating']==b, 'cort']= tmp_sid

            else:

                masterscale.loc[masterscale['rating']==b, 'cor']= (tmp_cor + masterscale['cor'])/2

                masterscale.loc[masterscale['rating']==b, 'cort']= (tmp_sid + masterscale['cort'])/2

 

       

    #End simul

   

    

    lpercs.sort()

    obrat = df['SIDnomin'].sum()

    lp99=lpercs[int(0.99*Nscen)]

    lp2=pd.DataFrame({'lp2':lpercs, \

                      'total_placed': total_places, \

                      'total_loss': total_losses, \

                      'bank_loss': lffp, \

                      'lp2_bank':lpercs_bank,\

                      'cas_izteka_sredstev': casi_total, \

                    #'cas_prekoracitve_odstotka': casi_percents})

                      })

    atenor = t_months/(len(df['ziv'])*12.0)

   

    

    #povprecja

    cor_pop = np.mean(cor)

    cort_pop = np.mean(cort)

    #masterscale.loc[masterscale['rating']==boni,'cort'] =  cort_pop

    #masterscale.loc[masterscale['rating']==boni,'cor'] =  cor_pop

    total_placed = np.mean(total_places)

    total_loss = np.mean(total_losses)

    l_FP = np.mean(lffp)

    loss_perc_bank = np.mean(lpercs_bank)

   

    profil = df[["loss_month","accum_loss_bank",'accum_loss_SID',"cs_nomin",'accum_loss_bank_pop','cs_nomin_loss']].groupby("loss_month").max()

   

    bench = pd.DataFrame(data={'sm':[0,15,30,45,60], 'plas':[0,0.25,0.5,0.75,1]})

    sns.set_style("whitegrid")

    fig,ax = plt.subplots(figsize=(16,9))   

    ax.plot(profil.index, (profil["accum_loss_bank"]/profil['cs_nomin_loss']), 'g--')

    #ax.plot(prej.index, prej["cs_nomin"]/vzvod, 'b')

    ax.plot(profil.index, (profil['accum_loss_SID']/profil['cs_nomin_loss']), 'r', label = 'SID')

    ax.plot(profil.index, ((profil['accum_loss_bank_pop'] + profil['accum_loss_SID'])/profil['cs_nomin_loss']), 'b', label= 'Skupaj')

    ax.plot(profil.index, (profil['accum_loss_bank_pop']/profil['cs_nomin_loss']), 'g', label='Financni posrednik')

    plt.legend(bbox_to_anchor=(0.4, 1), loc="best", borderaxespad=0.,fontsize = 'large')

    vals=ax.get_yticks()

    plt.axes.labelsize=8

    plt.title('Odstotek izgube glede na trenutno velikost portfelja')

    plt.xlabel("mesec")

    plt.ylabel("odstotek izgube")

    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])

    pyl.savefig('C:\\Users\\lhrobat\\Documents\\PROJECTS\\SID\\Outputs\\' + boni + "_fiksne_garancije"+ '_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+ '.pdf')

    plt.show()

   

    kr = np.percentile(a = lffp, q=99)-np.mean(lffp)

    coc = 0.08*kr

    cocr = np.mean(lffp)

 

    bal = pd.DataFrame({'znesek': ['Obrat','Izguba','Izguba glede na skupni znesek','Izguba na skupni znesek na leto'], \

                        'EKP':[obrat,total_loss,total_loss/total_placed, total_loss/(total_placed*atenor)], \

                        'FP':[0,l_FP,l_FP/total_placed,l_FP/(total_placed*atenor)], \

                        'Skupaj':[total_placed,total_loss+l_FP, (total_loss+l_FP)/total_placed,(total_loss+l_FP)/(total_placed*atenor)]})

#    summary = pd.Series({'Zacetni vložek': amt_placed_0,'Skupen obrat EKP': obrat, 'Izguba EKP': total_loss, 'Skupna izguba - odstotek' : total_loss/amt_placed_0, 'Koncna bilanca': balance,  'Skupaj dodeljeno': total_placed, 'Izguba na strani financnega posrednika':l_FP,'Izguba glede na skupna dodeljena sredstva': total_loss/total_placed, 'Število podeljenih kreditov': len(df),'Efektivnost plasmaja':eff_s, 'Skupaj mesecev':t_months, 'Cost of capital':coc, 'CoR2':cocr })

    summary = pd.Series({'Zacetni vložek': amt_placed_0, 'Število podeljenih kreditov': len(df),'Efektivnost plasmaja':eff_s,'Cost of capital':coc})

    masterscale.index = pd.IntervalIndex.from_arrays(masterscale['PD_low'],masterscale['PD_high'],closed='right')

    #df['rating'] = df['PD'].apply(lambda x : masterscale.iloc[masterscale.index.get_loc(x)]['rating'])

    #df['grade'] = df['PD'].apply(lambda x : masterscale.iloc[masterscale.index.get_loc(x)]['grade'])

    #df2=df.groupby(['rating','grade']).size()

    #df3=df2.reset_index().sort_values('grade')

   

    #histogram PD

    #df['rating'] = masterscale.loc[masterscale.index.get_indexer(df.PD),'rating']

   

    print (100.0*total_loss/amt_placed_0)

     

    writer = pd.ExcelWriter('C:\\Users\\lhrobat\\Documents\\PROJECTS\\SID\\Outputs\\output_FP_garancije_'+boni+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'.xlsx')

    summary.to_excel(writer,'Summary')

    df.to_excel(writer,'Simulation')

    lp2.to_excel(writer,'LPercs')

    bal.to_excel(writer,'Bals')

    #losses.to_excel(writer,'Loss_list')

    writer.save()

   

writer = pd.ExcelWriter('C:\\Users\\lhrobat\\Documents\\PROJECTS\\SID\\Outputs\\masterscale_' +dt.datetime.now().strftime('%Y%m%d_%H%M%S')+ '.xlsx')

masterscale.to_excel(writer,'Stroski')

#losses.to_excel(writer,'Loss_list')

writer.save()