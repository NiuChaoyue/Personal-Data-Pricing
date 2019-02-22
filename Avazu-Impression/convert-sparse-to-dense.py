"""
@Topic: Pricing of Impression Under Logistic Market Value Model
@Usage: Convert sparse feature vectors to dense feature vectors (according to the ideal weight vector)
@Author: Chaoyue Niu
@Email: rvince@sjtu.edu.cn
"""

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
import numpy as np
from sklearn.preprocessing import normalize

'''
Save training data for pricing usage
Pay Attention to sparsity
'''
D = 2**7
train_size = 404289670
fw = open("./new-input/theta_n%d"%D)
thetaStar = np.zeros(D, float)
flag_nonzero = np.zeros(D,int)
densethetaStar = []
for i in range(D):
    line = fw.readline()
    if not line:
        break
    thetaStar[i] = float(line)
    if(thetaStar[i] != 0.0):
        flag_nonzero[i] = 1
        densethetaStar.append(thetaStar[i])
fw.close()

'''
Save Dense w
'''
wline = ""
for aw in densethetaStar:
    wline += str(aw) + '\n'
fdw = open("./dense-input/theta_n%d"%D, 'w')
fdw.write(wline)
fdw.close()
print("Now save dense w file with size: %d"%len(densethetaStar))

"""
Save Dense X
"""
print("Now save dense train file with size: %d"%train_size)
fX = open("./new-input/train_X_n%d" % D)
for t in range(train_size):
    line = fX.readline()
    if not line:
        break
    linesp = line.split()
    dxt = ""
    for i in range(D):
        if(flag_nonzero[i] == 1):
            dxt += linesp[i] + ' '
    dxt += '\n'
    fdX = open('./dense-input/train_X_n%d' % D, 'a+')
    fdX.write(dxt)
    fdX.close()
fX.close()