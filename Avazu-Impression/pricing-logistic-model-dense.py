"""
@Topic: Pricing of Impression Under Logistic Market Value Model
@Version: The pure version (Dense Case)
@Author: Chaoyue Niu
@Email: rvince@sjtu.edu.cn
"""

import numpy as np
import math
from sklearn.preprocessing import normalize

"""
Load online impressions 
"""

def Read_CTR_market_values(T, on):
    fy = open("../dense-input/train_Y_n%d"%on)
    y = np.zeros(T, float)

    for t in range(T):
        line = fy.readline()
        if not line:
            break
        y[t] = float(line)
    fy.close()
    return y

def Read_CTR_market_value_thetastar(on, n):
    fw = open("../dense-input/theta_n%d"%on)
    thetaStar = np.zeros((n, 1), float)

    for i in range(n):
        line = fw.readline()
        if not line:
            break
        thetaStar[i,0] = float(line)
    fw.close()
    return thetaStar

def wx_prob(wTx):
    # bounded sigmoid function, this is the probability estimation
    return 1. / (1. + math.exp(-max(min(wTx, 35.), -35.)))

if __name__=="__main__":
    """
    Some global/inital variables
    """
    on = 2**7
    #logloss = 0.420, 0.406, 0.390
    if(on == 2**7):
        T = 100000#404289670
        # 2-norm domain of feature vector
        R2 = 0.5
        n = 21
    elif(on == 2**10):
        T = 100000 #404289670
        R2 = 0.8
        #only considering non-zero weights in weight vector
        n = 23
    elif(on == 2**23):
        T = 808579340
        R2 = 1.0
        n = 23
    #2-norm domain of weight vector, unnormalized
    #S = 1
    #shape matrix and center of the current ellipsoid
    A = np.identity(n)
    for i in range(n):
        A[i, i] = R2
    c = np.zeros((n,1),float)
    #uncertainty paramter
    delta = 0.0 #math.sqrt(R2) * n * 1.0 / T
    #threshold
    epsT = 1000000
    epsilon = max((n**2) * 1.0 /epsT,4*n*delta)

    print("Threshold: %f\n"%epsilon)

    """
    Read Market Values from File
    """
    y = Read_CTR_market_values(T, on)

    """
    Read trained (not true) weight vector in market value from file
    """
    thetaStar = Read_CTR_market_value_thetastar(on, n)

    """
    ratio between market value and reserve price
    """
    rp_ratio = 0.0

    tmpsum = 0.0
    for i in range(n):
        tmpsum += thetaStar[i,0] ** 2
    print("l2 norm of thetaStar: %f\n"%tmpsum)
    
    """
    Some counting/recording variables
    """
    regretVec = np.zeros(T, float)
    totalregretVec = np.zeros(T, float)
    totalMarketValue = np.zeros(T, float)
    regretRatio = np.zeros(T, float)

    """
    Read feature vectors one by one
    """
    fX = open("../dense-input/train_X_n%d" % on)

    for t in range(T):
        # CTR in round t
        # the feature vector xt
        xt = np.zeros((n, 1), float)
        line = fX.readline()
        if not line:
            break
        linesp = line.split()
        for i in range(n):
            xt[i, 0] = float(linesp[i])
        xt_T = xt.transpose()

        # Judge whether theta* is in the current ellipsoid
        if (np.dot(np.dot((thetaStar - c).transpose(), np.linalg.inv(A)), (thetaStar - c)) <= 1):
            print("Round %d: Yes! trained W is within the current ellipsoid."%t)
        else:
            print("Round %d: No! trained W is outside the current ellipsoid."%t)

        #the market value
        #ln
        #vt = y[t]
        vt = np.dot(xt_T,thetaStar)

        # the reserve price
        # qt = Query_reserve_price(xt, n)
        if(rp_ratio == 0):
            qt = -1000000000
        else:
            if(vt < 0):
                qt = vt * (1.0 / rp_ratio)
            else:
                qt = vt * rp_ratio

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
                totalMarketValue[t] = wx_prob(vt)
            else:
                totalregretVec[t] = totalregretVec[t - 1] + regretVec[t]
                totalMarketValue[t] = totalMarketValue[t - 1] + wx_prob(vt)
            continue
        else:
            #exploratory posted price
            if((pt_upper - pt_lower) > epsilon):
                pt = max(qt, (pt_lower + pt_upper)/2.0)
                #pt = (pt_lower + pt_upper) / 2.0
            #conservative posted price
            else:
                pt = max(qt, pt_lower - delta)
                #pt = pt_lower
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
                if ((vt >= qt) and (pt - vt > 0.01)):
                    regretVec[t] = wx_prob(vt)
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
                regretVec[t] = wx_prob(vt) - wx_prob(pt)
        if (t == 0):
            totalregretVec[t] = regretVec[t]
            totalMarketValue[t] = wx_prob(vt)
        else:
            totalregretVec[t] = totalregretVec[t - 1] + regretVec[t]
            totalMarketValue[t] = totalMarketValue[t - 1] + wx_prob(vt)
        print("Market value: %f; Pt_low: %f; Pt_up: %f" % (vt, pt_lower, pt_upper))
        print("Reserve price: %f"%qt)
        print("Posted price: %f"%pt)
        print("Regret: %f\n"%(regretVec[t]))
    fX.close()
    np.savetxt("./regret-total/Dense_RegretTotal_RP_rpratio%f_n%d_T_%d" % (rp_ratio, on, T), totalregretVec, fmt='%.10f')
    #save regret vector to file
    #np.savetxt("./regret/RegretsPure_n%d_T_%d"%(n,T), regretVec, fmt='%.10f')
    #np.savetxt("./regret/TotalRegretsPure_n%d_T_%d" % (n, T), totalregretVec, fmt='%.10f')
    #save regret ratio to file
    for t in range(T):
        if(totalMarketValue[t] != 0):
            regretRatio[t] = totalregretVec[t] * 1.0 / totalMarketValue[t]
    np.savetxt("./regret-ratio/Dense_RegretRatio_RP_rpratio%f_n%d_T_%d" % (rp_ratio, on, T), regretRatio, fmt='%.10f')
