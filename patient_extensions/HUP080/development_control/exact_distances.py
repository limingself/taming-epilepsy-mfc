import numpy as np
from scipy.stats import wasserstein_distance

def distance(pool, target):
    return np.array([wasserstein_distance(pool[...,j].ravel(),target[...,j].ravel()) for j in range(96)])

def time_distance(traces,ref):
    # Compare particles at each aligned sample using exact quantile integration.
    n,m=len(traces),len(ref); L=np.lcm(n,m)
    knots=np.union1d(np.arange(n+1)*(L//n),np.arange(m+1)*(L//m))
    a,b=np.sort(traces,axis=0),np.sort(ref,axis=0)
    return sum((y-x)/L*np.abs(a[x*n//L]-b[x*m//L]).mean(0) for x,y in zip(knots[:-1],knots[1:]))
