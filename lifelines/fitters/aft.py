from autograd import numpy as np
from autograd import value_and_grad, elementwise_grad as egrad, jacobian, hessian
from scipy.optimize import minimize
from lifelines.datasets import load_regression_dataset

df = load_regression_dataset()


def _cumulative_hazard(beta, X, T):
    rho_ =  2.9800
    lambda_ = np.exp(np.dot(X, beta))
    return (lambda_ * T) ** rho_


_hazard = egrad(_cumulative_hazard, argnum=2) # diff w.r.t. time

def _negative_log_likelihood(beta, X, T, E):
    n = T.shape[0]

    hz = _hazard(beta, X, T)
    hz = np.clip(hz, 1e-18, np.inf)

    ll = (
        (E * np.log(hz)).sum()
        - _cumulative_hazard(beta, X, T).sum()
    )
    return -ll / n


df['intercept'] = 1
T = df.pop('T').values
E = df.pop('E').values
X = df.values

_SLICE = {
    ''

}

init_values = np.zeros(3 + 1)  # one for intercept, one for shape

results = minimize(
    value_and_grad(_negative_log_likelihood),
    init_values,
    jac=True,
    args=(X, T, E),
    options={"disp": True},
)

print(results)

hessian_ = hessian(_negative_log_likelihood)(results.x, X, T, E)


def predict_survival_function(beta, x, times=np.linspace(0, 100)):
    return np.exp(-_cumulative_hazard(beta, x, times))
