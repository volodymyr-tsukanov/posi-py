# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:24:35 2024

@author: vt
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#zad1.2
data_csv = pd.read_csv("1/practice_lab_1.csv", sep=";")
data = data_csv.values

t1 = data[1::2,:]   #parzyste
t2 = data[0::2,:]   #nieparzyste
t3 = t1 - t2    #róźnica

avg = t3.sum() / len(t3)    #średnia

print(avg)
