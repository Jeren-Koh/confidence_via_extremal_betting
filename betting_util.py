import numpy as np
from math import lgamma
def lnbinomial(n, k):
    """
    Compute the natural logarithm of the binomial coefficient.
    """
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

def marginal(t, pr):
    T = len(pr) - 1
    M = np.ones(t + 1)
    
    for j in range(t + 1):
        #Because we're in logspace, this sum being 1 means the total value is e, which isn't a probability. 
        #So we'll use that as a flag for the sum being 0 in normal space.
        sm = 1  
        for k in range(j, T - t + j + 1):
            term = pr[k] + lnbinomial(T - t, k - j)
            if sm == 1:
                sm = term
            else:
                sm = np.logaddexp(sm, term)
        M[j] = sm
    
    return M

def marginal_priors(pr):
    """
    Compute marginal priors as a list of numpy arrays.
    
    Parameters:
        pr: np.ndarray, array of prior values.
        
    Returns:
        marginals: list of np.ndarray, marginal priors for each t.
    """
    T = len(pr) - 1
    marginals = [None] * T
    marginals[-1] = pr

    for t in range(1,T):
        marginals[t-1] = marginal(t, pr)

    return marginals

def sum_marginals(pr):
    T = len(pr) - 1
    S = np.zeros(T)

    for t in range(1, T + 1):
        sm = sum(
            np.exp(lnbinomial(t, k) + pr[t - 1][k]) 
            for k in range(t + 1)
        )
        S[t - 1] = sm

    return S
def seq2Returns(s,m):
    x = (s - m)
    return np.vstack((1+x,1-x))