
import numpy as np
import pandas as pd
import math

###############################################################################

# Najmanjša in največja velikost kredita
AMT_MIN = 30000
AMT_MAX = 10000000

# Porazdelitev ročnosti
ROCNOST_P = [0.0051,0.0308,0.0564,0.0821,0.1077,0.1077,0.1077,0.1077,0.1077,0.0906,0.0735,0.0564,0.0393,0.0222,0.0051]
ROCNOST_R = [36    ,42    ,48    ,54    ,60    ,66    ,72    , 78   ,84    ,90    ,96    ,102   ,108   ,114   ,120   ]

# Beta porazdelitev LGD
LGD_mu  = 0.60
LGD_sig = 0.2
LGD_a   = LGD_mu * LGD_mu * (1 - LGD_mu) / (LGD_sig * LGD_sig) - LGD_mu
LGD_b   = LGD_a * (1 / LGD_mu - 1)

# Porazdelitev PD
PD_mu = -3.2
PD_si = 1.0

###############################################################################

class Loan:
    def __init__(self, month_i):
        self.notional   = rand_notional()
        self.start_mnth = month_i
        self.maturity   = rand_maturity()
        self.moratorium = rand_moratorium(self.maturity/3, self.maturity/2)
        self.annuity    = self.notional / (self.maturity - self.moratorium)
        self.LGD        = rand_LGD()
        self.PD         = rand_PD()

    def age(self):
        if (self.maturity > 0):
            self.maturity -= 1
            
def rand_notional():
    porlen = math.log(AMT_MAX) * (1 - AMT_MIN / AMT_MAX) * AMT_MAX / AMT_MIN
    nm = np.random.random_sample() * (porlen - 1) + 1
    return AMT_MAX * math.log(AMT_MAX) / (nm + math.log(AMT_MAX))

def rand_maturity():
    return np.random.choice(ROCNOST_R, replace = True, p = ROCNOST_P)

def rand_moratorium(low, high):
    return round(np.random.random_sample() * (high - low) + low)

def rand_LGD():
    return np.random.beta(a=LGD_a, b=LGD_b)

def rand_PD():
    return logit(PD_mu + PD_si * np.random.randn())

def logit (score):
    return math.exp(score) / (1 + math.exp(score))

def yeartomonth(pd):
    return 1-math.pow(1-pd,1/12)

def survtime(sss,a):
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

   

masterscale = pd.DataFrame({'rating': ['AAA','AA+','AA','AA-','A+','A','A-',
                                       'BBB+','BBB','BBB-','BB+','BB','BB-','B+','B','B-',
                                       'C+','C'], \
                            'grade' : range(1,19), \
                            'RW'    : [0.2,0.2,0.2,0.2,0.5,0.5,0.5,1,1,1,1,1,1,1.5,1.5,1.5,1.5,1.5], \
                            'cor'   : [0.2,0.2,0.2,0.2,0.5,0.5,0.5,1,1,1,1,1,1,1.5,1.5,1.5,1.5,1.5]})

masterscale['PD_low']= (masterscale['grade']-0.5)*0.6113-12.247

masterscale['PD_high']= (masterscale['grade']+0.5)*0.6113-12.247

masterscale['PD_low']=masterscale['PD_low'].apply(logit)
masterscale['PD_high']=masterscale['PD_high'].apply(logit)
masterscale.loc[masterscale['rating']=='C','PD_high']=1
masterscale.index = pd.IntervalIndex.from_arrays(masterscale['PD_low'],masterscale['PD_high'],closed='right')
masterscale['coc'] = masterscale['grade'].apply(coc_SID)
masterscale['coc_brez'] = masterscale['grade'].apply(coc_brez)