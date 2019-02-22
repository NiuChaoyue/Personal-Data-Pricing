"""
@Topic: Pricing of Noisy Linear Query Under Linear Market Value Model
@Usage: Simulate/Generate Online Noisy Linear Queries (Multi-Dimensional)
@Author: Chaoyue Niu
@Email: rvince@sjtu.edu.cn
"""

import numpy as np
import math
from sklearn.preprocessing import normalize

"""
Simulate online linear query 
"""

def Query_feature_vector(n):
    #final feature vctor
    xt = np.zeros((n,1),float)
    #N: the original number of data owners
    N = 138493

    #generate "weight vectors" in linear queries
    #multivariate Gaussian distribution (0, I)
    mg_mean = np.zeros(N,float)
    mg_cov = np.identity(N)
    mulGau = np.random.multivariate_normal(mg_mean,mg_cov,1)
    mulGau = normalize(mulGau)
    #uniform distribution[-1, 1]
    uniFor = np.random.uniform(-1.0,1.0, (1,N))
    uniFor = normalize(uniFor)

    tmpwChoice = np.random.random_integers(0,1)
    if(tmpwChoice == 0):
        wVec = mulGau
    else:
        wVec = uniFor
    variance = math.pow(10.0,np.random.random_integers(-4,4))

    #privacy compensation based on tanh
    pcVec = np.zeros(N, float)
    for i in range(N):
        pcVec[i] = math.tanh(abs(wVec[0,i]) * 1.0 /math.sqrt(variance))
    maxPc = max(pcVec)
    minPc = min(pcVec)
    intervalPc = (maxPc - minPc) * 1.0 / n
    for i in range(N):
        index = int((pcVec[i] - minPc) * 1.0 /intervalPc) - 1
        index = max(0, index)
        index = min(n - 1, index)
        xt[index, 0] = xt[index, 0] + pcVec[i]
    #must assign values
    xt = normalize(xt,'l2',0)
    return xt

def self_normalize(theta,n,R2):
    tmp2sum = 0.0
    for i in range(n):
        tmp2sum += theta[i,0]**2
    for i in range(n):
        theta[i,0] = math.sqrt(theta[i,0]**2 * 1.0 /tmp2sum * R2)
    return theta

def Query_market_value_thetastar(n, R2):
    #thetaStar = np.zeros((n,1), float)

    # generate "weight vectors" in linear queries
    # multivariate Gaussian distribution (0, I)
    mg_mean = np.zeros(n, float)
    mg_cov = np.identity(n)
    mulGau = np.random.multivariate_normal(mg_mean, mg_cov, 1)
    mulGau = normalize(mulGau)
    # uniform distribution[-1, 1]
    uniFor = np.random.uniform(-1.0, 1.0, (1, n))
    uniFor = normalize(uniFor)

    tmpChoice = np.random.random_integers(0,1)
    if (tmpChoice == 0):
        thetaStar = abs(mulGau.transpose())
    else:
        thetaStar = abs(uniFor.transpose())
    thetaStar = self_normalize(thetaStar, n, R2)
    return thetaStar

if __name__=="__main__":
    """
    Some global/inital variables
    """
    #number of rounds
    T = 100000
    #number of features
    n = 100
    #2-norm domain of feature vector
    R2 = 4 * n
    #2-norm domain of weight vector
    S = 1

    #the true weight vector in market value
    thetaStar = Query_market_value_thetastar(n, R2/2.0)

    """
    Save feature vectors and theta* to file to keep the same
    """
    X = np.zeros((T, n), float)

    for t in range(T):
        #Query Q_t
        #the feature vector xt
        xt = Query_feature_vector(n)
        xt_T = xt.transpose()
        X[t] = xt_T

    np.savetxt("./new-query/X_T_%d_n_%d"%(T,n), X, fmt='%.10f')
    np.savetxt("./new-query/theta_T_%d_n_%d"%(T,n), thetaStar.transpose(), fmt='%.10f')