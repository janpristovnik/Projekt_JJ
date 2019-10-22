import numpy as np
import pandas as pd
import os

###############################################################################

#ASSUMPTIONS
n_scenario = 10

duration = 44

total_placed = 98600000

monthly_placed = total_placed/duration

SID_share = 0.625







path = 'C:\\Users\\jrems\\Documents\\SID\\'

os.chdir("C:\\Users\\jrems\\Documents\\GitHub\\Projekt_JJ\\")

from C_Loan import Loan

from C_ImportSID_PD import ImportSID_PD

from C_SID_Bank import Bank

#nalozimo podatke od SID (masterscale, PDs)
data = ImportSID_PD(path)

for i in range(n_scenario):
    bank = Bank()
    alive = []
    defaulted = []
    i = 0
    while (i == 0) | ((alive != []) & (defaulted != [])):
        for 
        





















if __name__ == "__main__":

    a = Loan()
    if (1 == 1):
        print(1)
        

