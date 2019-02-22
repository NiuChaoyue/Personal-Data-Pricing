"""
@Topic: Pricing of Noisy Linear Query Under Linear Market Value Model
@Version: Pure version (Multi-Dimensional)
@Author: Chaoyue Niu
@Email: rvince@sjtu.edu.cn
"""

import numpy as np
import math
from sklearn.preprocessing import normalize

"""
Simulate online linear query 
"""

def Read_Query_feature_vector(T, n):
    X = np.zeros((T, n), float)
    fX = open("./query/X_T_%d_n_%d"%(T,n))
    for t in range(T):
        line = fX.readline()
        if not line:
            break
        linesp = line.split()
        for i in range(n):
            X[t, i] = float(linesp[i])
    fX.close()
    return X

def Query_reserve_price(xt, n):
    #In reserve price, each feature vector's weight is 1
    return sum(xt)

def Read_Query_market_value_thetastar(T, n):
    thetaStar = np.zeros((n, 1), float)
    fTheta = open("./query/theta_T_%d_n_%d"%(T,n))
    line = fTheta.readline()
    linesp = line.split()
    for i in range(n):
        thetaStar[i,0] = float(linesp[i])
    fTheta.close()
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
    #shape matrix and center of the current ellipsoid
    A = np.identity(n)
    for i in range(n):
        A[i, i] = R2
    c = np.zeros((n,1),float)
    #uncertainty paramter
    delta = 0.0
    #threshold
    epsilon = max((n**2) * 1.0 /T,4*n*delta)

    print("Threshold: %f\n"%epsilon)

    """
    Read the true weight vector in market value from file
    """
    thetaStar = Read_Query_market_value_thetastar(T, n)
    
    """
    Some counting/recording variables
    """
    regretVec = np.zeros(T, float)
    totalregretVec = np.zeros(T, float)
    totalMarketValue = np.zeros(T, float)
    regretRatio = np.zeros(T, float)

    """
    Read Feature Vectors from File
    """
    X = Read_Query_feature_vector(T, n)

    for t in range(T):
        # Judge whether theta* is in the current ellipsoid
        if (np.dot(np.dot((thetaStar - c).transpose(), np.linalg.inv(A)), (thetaStar - c)) <= 1):
            print("Round %d: Yes! Theta* is within the current ellipsoid."%t)
        else:
            print("Round %d: No! Theta* is outside the current ellipsoid."%t)
        #Query Q_t
        #the feature vector xt
        xt = np.zeros((n,1), float)
        xt[:,0] = X[t]
        xt_T = xt.transpose()
        #the reserve price
        #qt = Query_reserve_price(xt, n)
        qt = 0.0
        #the market value
        vt = np.dot(xt_T, thetaStar)

        #intermediate vector
        bt = np.dot(A, xt)/(1.0 * math.sqrt(np.dot(np.dot(xt_T, A), xt)))
        bt_T = bt.transpose()

        #lower bound and upper bound on estimating the market value
        pt = 0.0
        pt_lower = np.dot(xt_T, c - bt)
        pt_upper = np.dot(xt_T, c + bt)

        if(qt >= (pt_upper + delta)):
            if (t == 0):
                totalregretVec[t] = regretVec[t]
                totalMarketValue[t] = vt
            else:
                totalregretVec[t] = totalregretVec[t - 1] + regretVec[t]
                totalMarketValue[t] = totalMarketValue[t - 1] + vt
            continue
        else:
            #exploratory posted price
            if((pt_upper - pt_lower) > epsilon):
                #pt = max(qt, (pt_lower + pt_upper)/2)
                pt = (pt_lower + pt_upper) / 2.0
            #conservative posted price
            else:
                pt = pt_lower
                #pt = max(qt, pt_lower - delta)
            #handle feedback from the data consumer
            #posted price is rejected
            if(pt > vt):
                #position parameter
                alphat = ((pt_lower + pt_upper)/2.0 - (pt + delta))/(math.sqrt(np.dot(np.dot(xt_T,A),xt)))
                #update the shape matrix and center of ellipsoid
                if((alphat >= (-1.0/n)) and (alphat <= 1)):
                    nextA =(n**2 * (1 - alphat**2) * 1.0 /(n**2 - 1)) * (A - (2.0 * (1 + n * alphat)/((n + 1) * (1 + alphat))) * np.dot(bt,bt_T))
                    nextc = c - ((1 + n * alphat) * 1.0 /(n + 1)) * bt
                    A = nextA
                    c = nextc
                #record regret
                #pt - vt) > 10**(-1)
                if ((vt > qt) and (pt - vt > 0.01)):
                    regretVec[t] = vt
            # posted price is accepted
            else:
                # position parameter
                alphat = ((pt_lower + pt_upper) / 2.0 - (pt - delta)) / (math.sqrt(np.dot(np.dot(xt_T, A), xt)))
                alphat_neg = alphat * -1
                if((alphat_neg >= (-1.0 /n)) and (alphat_neg <= 1)):
                    nextA = (n**2 * (1 - alphat**2) * 1.0 /(n**2 - 1)) * (A - (2.0 * (1 - n * alphat)/((n+1) * (1 - alphat))) * (np.dot(bt, bt_T)))
                    nextc = c + ((1 - n * alphat) * 1.0 /(n + 1)) * bt
                    A = nextA
                    c = nextc
                #record regret
                regretVec[t] = vt - pt
        if (t == 0):
            totalregretVec[t] = regretVec[t]
            totalMarketValue[t] = vt
        else:
            totalregretVec[t] = totalregretVec[t - 1] + regretVec[t]
            totalMarketValue[t] = totalMarketValue[t - 1] + vt
        print("Market value: %f; Pt_low: %f; Pt_up: %f" % (vt, pt_lower, pt_upper))
        print("Posted price: %f"%pt)
        print("Regret: %f\n"%(regretVec[t]))
    #save regret vector to file
    #np.savetxt("./regret/RegretsPure_n%d_T_%d"%(n,T), regretVec, fmt='%.10f')
    #np.savetxt("./regret/TotalRegretsPure_n%d_T_%d" % (n, T), totalregretVec, fmt='%.10f')
    #save regret ratio to file
    for t in range(T):
        if(totalMarketValue[t] != 0):
            regretRatio[t] = totalregretVec[t] * 1.0 / totalMarketValue[t]
    np.savetxt("./regret-ratio/RegretRatio_Pure_n%d_T_%d" % (n, T), regretRatio, fmt='%.10f')