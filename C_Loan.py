
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
    def __init__(self, month_i, data):
        self.notional   = rand_notional()
        self.start_mnth = month_i
        self.maturity   = rand_maturity()
        self.moratorium = rand_moratorium(self.maturity/3, self.maturity/2)
        self.annuity    = self.notional / (self.maturity - self.moratorium)
        self.LGD        = rand_LGD()
        self.PD         = rand_PD()
        self.grade      = grade(data,self.PD)
        self.rating     = rating(data,self.PD)
        self.survtime   = survtime(data, self.rating)
        

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

def yeartomonth(PD):
    return 1-math.pow(1-PD,1/12)

def survtime(data,sss):
    lamdas = data.lamdas
    y0 = data.y0
    lm = data.lm
    isum = data.isum
    a = np.random.random_sample()
    idx=lamdas[sss].index.get_loc(a)
    ld=lamdas[sss].iloc[idx]
    yy0=y0[sss].iloc[idx]
    s1=lamdas[sss].index[idx].left
    x1=lm[sss].index[idx].left
    x= -math.log(-(a-s1)*ld*isum[sss]/yy0+math.exp(-ld*x1))/ld
    return x

def grade(data,PD):
    masterscale = data.masterscale
    return masterscale.iloc[masterscale.index.get_loc(PD)]["grade"]

#meji ga lohk zbrišemo ker je isti k grade 
def rating(data,PD):
    masterscale = data.masterscale
    return masterscale.iloc[masterscale.index.get_loc(PD)]["rating"]


 

