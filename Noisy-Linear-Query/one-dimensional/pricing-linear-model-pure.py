"""
@Topic: Pricing of Noisy Linear Query Under Linear Market Value Model
@Version: Pure version (One-Dimensional)
@Author: Chaoyue Niu
@Email: rvince@sjtu.edu.cn
"""

import numpy as np
import math
from sklearn.preprocessing import normalize

"""
Simulate online linear query 
"""

def Query_market_value_thetastar(R2):
    # uniform distribution[1.0, sqrt(R2)]
    uniFor = np.random.uniform(1.0, math.sqrt(R2))
    return uniFor

def Query_market_value_uncertainty(vt_certain, delta, T):
    #standard deviation \sigma
    #C = 2
    sigma = delta * 1.0 /(math.sqrt(2 * math.log(2)) * math.log(T))
    noise = np.random.normal(0.0, sigma)
    print("Noise: %f"%noise)
    vt = vt_certain + noise
    return vt

if __name__=="__main__":
    """
    Some global/inital variables
    """
    #number of rounds
    T = 100
    #number of features
    n = 1
    #2-norm domain of feature vector
    R2 = 4 * n
    #2-norm domain of weight vector
    S = 1
    #interval of theta*
    theta_lower = 0.0
    theta_upper = math.sqrt(R2)

    #uncertainty paramter
    #delta = n * 1.0 /T
    delta = 0.0

    #threshold
    #pay attention to python 2.7: float/
    epsilon = max(math.log(2,T)/T, 2.0 * delta)

    print("Threshold: %f\n"%epsilon)

    """
    Generate the true weight
    """
    thetaStar = math.sqrt(R2/2.0)
    # thetaStar = 1.19933496843
    #thetaStar = Query_market_value_thetastar(R2/2.0)
    print(thetaStar)
    
    """
    Some counting/recording variables
    """
    regretVec = np.zeros(T, float)
    totalregretVec = np.zeros(T, float)

    for t in range(T):
        # Judge whether theta* is in the current interval
        if (thetaStar >= theta_lower and thetaStar <= theta_upper):
            print("Round %d: Yes! Theta* is within the current interval."%t)
        else:
            print("Round %d: No! Theta* is outside the current interval."%t)
        #Query Q_t
        # the feature vector xt
        xt = 1.0
        qt = 0.0
        #the market value with uncertainty
        vt = xt * thetaStar
        #vt = Query_market_value_uncertainty(vt_certain,delta,T)

        #lower bound and upper bound on estimating the market value
        pt = 0.0
        pt_lower = xt * theta_lower
        pt_upper = xt * theta_upper

        if(qt >= pt_upper + delta):
            if (t == 0):
                totalregretVec[t] = regretVec[t]
            else:
                totalregretVec[t] = totalregretVec[t - 1] + regretVec[t]
            continue
        else:
            #exploratory posted price
            if((pt_upper - pt_lower) > epsilon):
                pt = max(qt, (pt_lower + pt_upper)/2.0)
            #conservative posted price
            else:
                pt = max(qt, pt_lower - delta)
            #handle feedback from the data consumer
            #posted price is rejected
            if(pt > vt):
                theta_upper = (pt + delta)/xt
                #record regret
                #pt - vt > 10**(-4) considering the precision
                if((vt > qt) and (pt - vt > 0.01)):
                    regretVec[t] = vt
            # posted price is accepted
            else:
                theta_lower = (pt - delta)/xt
                #record regret
                regretVec[t] = vt - pt
        if (t == 0):
            totalregretVec[t] = regretVec[t]
        else:
            totalregretVec[t] = totalregretVec[t - 1] + regretVec[t]
        print("Market value: %f; Pt_low: %f; Pt_up: %f" % (vt, pt_lower, pt_upper))
        print("Reserve price: %f" % qt)
        print("Posted price: %f" % pt)
        print("Regret: %f\n"%(regretVec[t]))
    #save regret vector to file
    #np.savetxt("./regret/RegretsPure_n%d_T_%d"%(n,T), regretVec, fmt='%.10f')
    np.savetxt("./regret-total/TotalRegretsPure_n%d_T_%d" % (n, T), totalregretVec, fmt='%.10f')