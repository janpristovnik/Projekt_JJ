import numpy as np
import pandas as pd
import os

path = 'C:\\Users\\jrems\\Documents\\SID\\'

from C_Loan import Loan

from C_ImportSID_PD import ImportSID_PD

data = ImportSID_PD(path)

if __name__ == "__main__":

    a = Loan()
    if (1 == 1):
        print(1)