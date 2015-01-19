from numpy import dot, zeros
from numpy.linalg import norm
from numpy import log, exp
import numpy as np

def _d_minimizing_function_j(Theta, X, j, durations, T, coef_penalizer, smoothing_penalizer):
    return coef_penalizer * _d_coef_norm(Theta, j)\
        + smoothing_penalizer * _d_smoothing_norm(Theta, j)\
        - _d_log_likelihood_j(Theta, X, j, durations, T)

def _d_coef_norm(Theta, j):
    return Theta[j,:]

def _d_smoothing_norm(Theta, j):
    m, d = Theta.shape

    if j == 0:
        return -(Theta[j+1,:] - Theta[j,:])
    elif j == m - 1:
        return Theta[j,:] - Theta[j-1,:]
    else:
        return (Theta[j,:] - Theta[j-1,:]) - (Theta[j+1,:] - Theta[j,:])

def _d_log_likelihood_j(Theta, X, j, durations, T):
    m, d = Theta.shape
    n, d = X.shape

    t = T[j]
    survival_booleans = _survival_status_at_time_t(t, durations)
    logistic_likelihood = zeros(d)

    for i in xrange(n):

        is_dead = survival_booleans[i]

        x_i = X[i,:]
        logistic_likelihood += (is_dead - _sum_exp_f(Theta, x_i, j) / _sum_exp_f(Theta, x_i, m))*x_i

        if np.any(np.isnan(logistic_likelihood)):
            import pdb
            pdb.set_trace()

    return logistic_likelihood

def _survival_status_at_time_t(t, T):
    """
    1 if dead at time t, else 0
    """
    return (t >= T).astype(float)

def _first_part_log_likelihood_j(Theta, X, durations, T, j):
    n, d = X.shape
    m, d = Theta.shape
    first_sum = 0
    t = T[j]
    survival_booleans = _survival_status_at_time_t(t, durations)
    theta_j = Theta[j,:]

    #return (X.dot(theta_j)*survival_booleans).sum()

    for i in xrange(n):

        is_dead = survival_booleans[i]
        x_i = X[i,:]

        if is_dead:
            first_sum += theta_j.dot(x_i)
    return first_sum

def _second_part_log_likelihood(Theta, X):
    n, d = X.shape
    m, d = Theta.shape
    second_sum = 0

    for i in xrange(n):
        x_i = X[i,:]
        second_sum += log(_sum_exp_f(Theta, x_i, m))

    return second_sum

def _log_likelihood(Theta, X, durations, T):
    m, d = Theta.shape
    log_likelihood = 0
    for j in range(m):
        log_likelihood += _first_part_log_likelihood_j(Theta, X, durations, T, j)
    return log_likelihood - _second_part_log_likelihood(Theta, X)

def _minimizing_function(Theta, X, durations, T, coef_penalizer, smoothing_penalizer):
    return 0.5 * coef_penalizer * _coef_norm(Theta) + \
        0.5 * smoothing_penalizer * _smoothing_norm(Theta) - \
        _log_likelihood(Theta, X, durations, T)

def _smoothing_norm(Theta):
    t, d = Theta.shape
    s = 0
    for i in range(t - 1):
        s += norm(Theta[i+1,:] - Theta[i,:])**2
    return s


def _coef_norm(Theta):
    return norm(Theta) ** 2


def _sum_exp_f(Theta, x, upper_bound):
    m, d = Theta.shape
    v = Theta.dot(x)
    return exp(v[::-1].cumsum()[-(upper_bound+1):]).sum() + float(upper_bound == m)

