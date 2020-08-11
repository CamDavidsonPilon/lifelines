# -*- coding: utf-8 -*-
import warnings
from scipy.stats import weibull_min
from sklearn import linear_model
from sklearn.exceptions import ConvergenceWarning


def generate_weibull_dataset(lambda_, rho, censoring_ratio, N=15000):
    max_ = 10

    while True:
        RVs = weibull_min(rho, scale=lambda_).rvs(N)
        T = max_ * np.random.uniform(size=N)
        if abs((T < RVs).mean() - censoring_ratio) < 0.01:
            return np.minimum(RVs, T), RVs < T

        max_ *= (T < RVs).mean() / censoring_ratio


def generated_features(T, E):
    return np.array(
        [
            T.mean(),
            T[E].mean(),
            E.mean(),
            np.log(T.mean()),
            np.log(T[E].mean()),
            np.log(E.mean()) * np.log(T.mean()),
            np.log(E.mean()),
            np.log(T).std(),
            (T ** 2).mean(),
            E.mean() * np.log(T).mean(),
            np.log(T[E]).mean(),
            (E.mean() * T * np.log(T)).mean(),
            (T[E] * np.log(T[E])).mean(),
            (T ** 2 * np.log(T)).mean(),
        ]
    )


lambdas = []
rhos = []
features = []


for lambda_ in np.logspace(-2, 4, 20):
    for rho in np.logspace(-1, 1.0, 21):  # I want to hit rho_ = 1.
        for c in np.linspace(0.00, 0.95, 11):
            print(lambda_, rho, c)
            T, E = generate_weibull_dataset(lambda_, rho, censoring_ratio=c)
            X = generated_features(T, E)
            lambdas.append(np.log(lambda_))
            rhos.append(np.log(rho))
            features.append(X)

cols = [
    "T.mean()",
    "T[E].mean()",
    "E.mean()",
    "np.log(T.mean())",
    "np.log(T[E].mean())",
    "np.log(E.mean()) * np.log(T.mean())",
    "np.log(E.mean())",
    "np.log(T).std()",
    "(T**2).mean()",
    "E.mean() * np.log(T).mean()",
    "np.log(T[E]).mean()",
    "(E.mean() * T * np.log(T)).mean()",
    "(T[E]*np.log(T[E])).mean()",
    "(T ** 2 * np.log(T)).mean()",
]


X = pd.DataFrame(features, columns=cols)

mean_ = X.mean()
std_ = X.std()
X_ = (X - mean_) / std_


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    lr = linear_model.LassoCV(cv=5, fit_intercept=True, normalize=False, alphas=np.logspace(-1.0, 1, 100)).fit(X_, lambdas)

print("np.exp(%.4E" % lr.intercept_)
for col, coef, mean__, std__ in zip(cols, lr.coef_, mean_, std_):
    if np.abs(coef) < 1e-5:
        continue

    print("+ %.4E * (%s - %.4E)/%.4E " % (coef, col, mean__, std__))
print(")")


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    lr = linear_model.LassoCV(cv=5, fit_intercept=True, normalize=False, alphas=np.logspace(-1.0, 1, 100)).fit(X_, rhos)

print(",")

print("np.exp(%.4E" % lr.intercept_)
for col, coef, mean__, std__ in zip(cols, lr.coef_, mean_, std_):
    if np.abs(coef) < 1e-5:
        continue

    print("+ %.4E * (%s - %.4E)/%.4E " % (coef, col, mean__, std__))
print(")")
