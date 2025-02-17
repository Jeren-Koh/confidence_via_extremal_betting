import numpy as np
from math import lgamma
from scipy.special import lambertw
import betting_util

def log_minmax_prior(T):
    kvals = np.arange(T + 1)
    vterms = np.array([vterm(k, T) for k in kvals])
    lV = -np.log(np.sum(vterms))
    P = np.array([pk(k, lV, T) for k in kvals])
    return P

def vterm(k, T):
    if k == 0 or k == T:
        return 1
    return np.exp(
        lgamma(T + 1) +
        k * np.log(k) +
        (T - k) * np.log(T - k) -
        lgamma(k + 1) -
        lgamma(T - k + 1) -
        T * np.log(T)
    )

def pk(k, lV, T):
    if k != 0 and k != T:
        return lV + k * np.log(k / T) + (T - k) * np.log((T - k) / T)
    return lV

def beta_lower(d, R, T):
    """
    Compute the lower bound of beta*
    """
    log_d = np.log(d)
    sqrt_term = np.sqrt(T * (R - log_d) * (16 * R**2 + 9 * T**2 + 2 * log_d * (-16 * R + 9 * T + 8 * log_d)))
    num = 4 * R * T - 4 * T * log_d + np.sqrt(2) * sqrt_term
    exponent = 36 / (T * (9 + num**2 / (T**2 * (-2 * R + T + 2 * log_d)**2)))
    
    return 1 + 1/lambertw(-(d**exponent) / np.e, k=-1).real

def log_regret(T):
    #First term
    lV = betting_util.lnbinomial(T,0)
    for k in range(1,T):
        lV = np.logaddexp(lV, betting_util.lnbinomial(T,k)  + k*np.log(k/T) + (T-k)*(np.log(T-k) - np.log(T)))
    #Last term
    lV = np.logaddexp(lV, betting_util.lnbinomial(T,T))
    return lV

def truncated_prior(T,d):
    p=np.zeros(T+1)
    beta_bound = beta_lower(d,log_regret(T),T)
    lim1 = (1-beta_bound)/2
    lim2 = (1+beta_bound)/2
    #First term
    b = 0
    if b>lim1 and b<lim2:
        b=lim1
    p[0] = np.log(1-b)*T
    w = p[0] + betting_util.lnbinomial(T,0)

    for i in range(1,T):
        b = i/T
        if b>lim1 and b<lim2:
            if b>0.5:
                b=lim2
            else:
                b=lim1
        p[i] = i*np.log(b) + (T-i)*np.log(1-b)
        w = np.logaddexp(w,p[i] + betting_util.lnbinomial(T,i))

    #Last term
    b = 1
    if b>lim1 and b<lim2:
        b=lim2
    p[T] = np.log(b)*T
    w = np.logaddexp(w,p[T])
    p -= w
    return p
