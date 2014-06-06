#cox regression
#http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes3.pdf

import pandas as pd
import numpy as np
from numpy import dot, exp
from numpy.linalg import solve, norm

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

def theta_x(X,beta):
    return dot(exp(np.dot(X,beta)).T,X)


def score_efron(X, beta, T, E):
    """
    X: (n,d)
    beta: (d,1)
    T: (n,1)
    """
    assert X.shape[1] == beta.shape[0]
    assert X.shape[0] == T.shape[0]

    n,d = X.shape
    partial_score = np.zeros((1,d))
    for t in np.unique(T[E]):

        ix = np.where((T == t) & E)[0]
        m = ix.shape[0]
        R = risk_set(T,t)
        X_j = X[R]
        X_tie = X[ix,:]
        tied_theta_x = theta_x(X_tie,beta)
        assert tied_theta_x.shape == (1,d)

        tied_theta = theta(X_tie,beta).sum()
        all_theta_x = theta_x(X_j,beta)
        assert all_theta_x.shape == (1,d)

        risk_theta = sum_exp_over_risk(X, beta, R)
        partial_sum = np.zeros((1,d))

        for l in range(m): 
            c = 1.0*l/m
            partial_sum += (all_theta_x - c*tied_theta_x)/(risk_theta - c*tied_theta)
            assert partial_sum.shape==(1,d)


        partial_score = partial_score + (X_tie.sum(0) - partial_sum)
        assert partial_score.shape == (1,d)

    return partial_score

def hessian_efron(X,beta, T, E):
    """
    X: (n,d)
    beta: (d,1)
    T: (n,1)
    """
    assert X.shape[1] == beta.shape[0]
    assert X.shape[0] == T.shape[0]

    d = X.shape[1]
    M = np.zeros((d,d))
    for t in np.unique(T[E]):

        #compute the risk factor
        R = risk_set(T,t)
        theta_x_x = np.zeros((d,d))
        Z_risk = np.zeros((1,d))
        sum_thetas = sum_exp_over_risk(X, beta, R)
        for i in R:
            x_j = X[[i], :]
            assert x_j.shape == (1,d)
            theta_x_x += np.dot( x_j.T, x_j)*theta(x_j, beta)
            assert theta_x_x.shape == (d,d)
            Z_risk += theta_x(x_j, beta)
            assert Z_risk.shape == (1,d)

        #compute the tied factor
        ix = np.where((T == t) & E)[0]
        X_tie = X[ix,:]
        m = ix.shape[0]
        tied_theta_x_x = np.zeros((d,d))
        Z_tied = np.zeros((1,d))
        tied_theta = 0
        for i in range(m):
            x_j = X_tie[[i],:]
            assert x_j.shape == (1,d)
            tied_theta_x_x += theta(x_j, beta)*dot(x_j.T, x_j)
            assert tied_theta_x_x.shape == (d,d)
            tied_theta += theta(x_j, beta)
            Z_tied += theta_x(x_j, beta)
            assert Z_tied.shape == (1,d)

        M_t = np.zeros((d,d))
        for l in range(m):
            c = 1.0*l/m
            phi = (sum_thetas - c*tied_theta)
            a1 = (theta_x_x - c*tied_theta_x_x)/phi
            assert a1.shape==(d,d)
            Z = Z_risk - c*Z_tied 
            a2 = dot(Z.T, Z)/phi**2
            assert a2.shape == (d,d)
            M_t += a1 - a2
        M += M_t

    return M


def sum_exp_over_risk(X, beta, R):
    return exp(dot(X[R], beta)).sum()


def risk_set(T, t):
    return np.where(T >= t)[0]


def newton_rhapdson(X, T, E, initial_beta = None, step_size = 1., epsilon = 0.001,
                   score=score_efron, hessian=hessian_efron):

    assert epsilon <= 1., "epsilon must be less than or equal to 1."
    n,d = X.shape
    if initial_beta is not None:
        assert beta.shape == (d, 1)
        beta = initial_beta
    else:
        beta = np.zeros((d,1))


    converging = True
    while converging:

        delta = solve(hessian(X,beta,T,E), step_size*score(X,beta,T,E).T)

        beta_new = delta + beta


        beta = beta_new
        if norm(delta) < epsilon:
            converging = False


    return beta


if __name__=="__main__":

    X = data['x'][:,None]
    T = data['t']
    E = data['E']

    #tests from http://courses.nus.edu.sg/course/stacar/internet/st3242/handouts/notes3.pdf
    beta = np.array([[0]])

    l = hessian_efron(X, beta, T, E)
    u = score_efron(X, beta, T, E)
    assert np.abs(l[0][0] - 77.13) < 0.05
    assert np.abs(u[0] - -2.51) < 0.05
    beta = beta + u/l
    assert np.abs(beta - -0.0326) < 0.05


    l = hessian_efron(X, beta, T, E)
    u = score_efron(X, beta, T, E)
    assert np.abs(l[0][0] - 72.83) < 0.05
    assert np.abs(u[0] - -0.069) < 0.05
    beta = beta + u/l
    assert np.abs(beta - -0.0325) < 0.01


    l = hessian_efron(X, beta, T, E)
    u = score_efron(X, beta, T, E)
    assert np.abs(l[0][0] - 72.70) < 0.01
    assert np.abs(u[0] - -0.000061) < 0.01
    beta = beta + u/l
    assert np.abs(beta - -0.0335) < 0.01

    print newton_rhapdson(X, T, E)



    X = np.random.randn(9,3)
    beta = np.zeros((3,1))
    score_efron(X, beta, T, E)
    hessian_efron(X, beta, T, E)






