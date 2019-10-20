import numpy as np
import pandas as pd
import os

path = 'C:\\Users\\jpristovnik\\Documents\\SID\\'

os.chdir("C:\\Users\\jpristovnik\\Documents\\GitHub\\Projekt_JJ\\")

from C_Loan import Loan

from C_ImportSID_PD import ImportSID_PD

#nalozimo podatke od SID (masterscale, PDs)
data = ImportSID_PD(path)

kredit = Loan(2,data)

if __name__ == "__main__":

    a = Loan()
    if (1 == 1):
        print(1)
        

