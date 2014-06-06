#cox regression
#http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes3.pdf

import pandas as pd
import numpy as np
from numpy import dot, exp


data = pd.DataFrame( [  
         [6,31.4], 
         [98, 21.5],  
         [189, 27.1],  
         [374, 22.7],  
         [1002, 35.7],  
         [1205, 30.7],  
         [2065, 26.5],  
         [2201, 28.3],  
         [2421, 27.9] ],
      columns = ['t', 'x'])
data['E'] = True


def theta(x, beta):
    return exp(dot(x,beta))


def score(X, beta, T, E):
    """
    X: (n,d)
    beta: (d,1)
    T: (n,1)
    """

    assert X.shape[1] == beta.shape[0]
    assert X.shape[0] == T.shape[0]

    n,d = X.shape
    second_term = np.zeros(X.shape[1])
    for t in T[E]:
        R = risk_set(T,t)
        X_j = X[R]
        num = np.zeros((1,d))
        for i in range(R.shape[0]):
            x_j =  X_j[i,:]
            num += x_j*theta(x_j, beta)

        denom = sum_exp_over_risk(X, beta, R)
        second_term = second_term + num/denom

    return X[E].sum(0) - second_term

def sum_exp_over_risk(X, beta, R):
    return exp(dot(X[R], beta)).sum()


def hessian(X, beta, T, E):
    """
    X: (n,d)
    beta: (d,1)
    T: (n,1)
    """

    assert X.shape[1] == beta.shape[0]
    assert X.shape[0] == T.shape[0]

    d = X.shape[1]
    M = np.zeros((d,d))
    for t in T[E]:
        R = risk_set(T,t)
        sum_thetas = sum_exp_over_risk(X, beta, R)
        #compute first part of hessian
        a1 = np.zeros((d,d))
        a2 = np.zeros((1,d))

        for i in R:
            x_j = X[[i],:]
            assert x_j.shape == (1,d)
            a1 += dot(x_j.T, x_j)*theta(x_j, beta)
            a2 += x_j*theta(x_j, beta)
            assert a2.shape == (1,d)

        assert a1.shape == (d,d)
        A1 = a1 / sum_thetas
        A2 = dot(a2.T, a2) / sum_thetas**2
        assert A2.shape == (d,d)

        M += A1 - A2

    return M


def risk_set(T, t):
    return np.where(T >= t)[0]


if __name__=="__main__":

    X = data['x'][:,None]
    T = data['t']
    E = data['E']

    #tests from http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes3.pdf
    beta = 0

    l = hessian(X, beta, T, E)
    u = score(X, beta, T, E)
    assert np.abs(l[0][0] - 77.13) < 0.05
    assert np.abs(u[0] - -2.51) < 0.05
    beta = beta + u/l
    assert np.abs(beta - -0.0326) < 0.05


    l = hessian(X, beta, T, E)
    u = score(X, beta, T, E)
    assert np.abs(l[0][0] - 72.83) < 0.05
    assert np.abs(u[0] - -0.069) < 0.05
    beta = beta + u/l
    assert np.abs(beta - -0.0325) < 0.01


    l = hessian(X, beta, T, E)
    u = score(X, beta, T, E)
    assert np.abs(l[0][0] - 72.70) < 0.01
    assert np.abs(u[0] - -0.000061) < 0.01
    beta = beta + u/l
    assert np.abs(beta - -0.0335) < 0.01


    X = np.random.randn(9,3)
    beta = np.zeros((3,1))
    score(X, beta, T, E)
    hessian(X, beta, T, E)




