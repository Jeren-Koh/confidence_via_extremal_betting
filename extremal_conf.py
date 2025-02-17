import numpy as np
import betting_util as butil
def portfolio_algo(T,X,pr):
    """
    Compute the wealth of the extremal strategy defined by log-prior pr on sequence of returns X.
    
    Parameters:
        T: length of the sequence of returns.
        X: Sequence of returns.
        pr: log-prior allocation vector.
        
    Returns:
        w: wealth of the extremal strategy on sequence of returns X.
    """
    P = np.zeros((T + 1, T + 1), dtype=X.dtype)
    P[0, 0] = 1

    # First column
    for j in range(1, T + 1):
        P[j, 0] = X[1, j - 1] * P[j - 1, 0]

    # Remaining entries
    for t in range(1, T + 1):
        for k in range(1, t):
            P[t, k] = X[0, t - 1] * P[t - 1, k - 1] + X[1, t - 1] * P[t - 1, k]
        P[t, t] = X[0, t - 1] * P[t - 1, t - 1]
    w = 0
    for i in range(T + 1):
        if P[T,i]!=0: 
            w += np.exp(pr[i] + np.log(P[T, i]))
    return w

def conf(S, prior, d):
    """
    Compute the confidence sequence for the extremal strategy defined by log-prior prior on sequence S with confidence level 1-d.
    
    Parameters:
        S: Data sequence in [0,1].
        X: log-prior allocation vector.
        d: confidence level parameter, the sequence will have confidence level 1-d.
        
    Returns:
        lcblist, ucblist: Lower and upper confidence sequences.
    """
    T = len(S)
    lcblist = np.zeros(T)
    ucblist = np.zeros(T)
    invdelta = 1/d
    m_lb_old = np.finfo(float).eps
    m_ub_old = 1 - np.finfo(float).eps
    
    for i in range(T):
        St = S[:i+1]
        mean_c = np.mean(St)
        
        # Upper confidence interval
        m_ub = m_ub_old
        m_lb = max(m_lb_old, mean_c)
        m = m_ub

        X = butil.seq2Returns(St, m)
        w = portfolio_algo(i+1, X, prior[i])
        
        if w >= invdelta:
            while (m_ub - m_lb) > 0.0001:
                m = (m_ub + m_lb)/2
                X = butil.seq2Returns(St, m)
                w = portfolio_algo(i+1, X, prior[i])
                if w >= invdelta:
                    m_ub = m
                else:
                    m_lb = m
                    
        ucblist[i] = m_ub
        m_ub_old = m_ub
        
        # Lower confidence interval
        m_ub = min(m_ub, mean_c)
        m_lb = m_lb_old
        m = m_lb
        
        X = butil.seq2Returns(St, m)
        w = portfolio_algo(i+1, X, prior[i])
        
        if w >= invdelta:
            while (m_ub - m_lb) > 0.0001:
                m = (m_ub + m_lb)/2
                X = butil.seq2Returns(St, m)
                w = portfolio_algo(i+1, X, prior[i])
                if w >= invdelta:
                    m_lb = m
                else:
                    m_ub = m
                    
        lcblist[i] = m_lb
        m_lb_old = m_lb
        
    return lcblist, ucblist