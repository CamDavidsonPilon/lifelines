import numpy as np


def _negative_log_likelihood(lambda_rho, T, E):
    if np.any(lambda_rho < 0):
        return np.inf
    lambda_, rho = lambda_rho
    return - np.log(rho * lambda_) * E.sum() - (rho - 1) * (E * np.log(lambda_ * T)).sum() + ((lambda_ * T) ** rho).sum()


def _lambda_gradient(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return - rho * (E / lambda_ - (lambda_ * T) ** rho / lambda_).sum()


def _rho_gradient(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return - E.sum() / rho - (np.log(lambda_ * T) * E).sum() + (np.log(lambda_ * T) * (lambda_ * T) ** rho).sum()
    # - D/p - D Log[m t] + (m t)^p Log[m t]


def _d_rho_d_rho(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return (1. / rho ** 2 * E + (np.log(lambda_ * T) ** 2 * (lambda_ * T) ** rho)).sum()
    # (D/p^2) + (m t)^p Log[m t]^2


def _d_lambda_d_lambda_(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return (rho / lambda_ ** 2) * (E + (rho - 1) * (lambda_ * T) ** rho).sum()


def _d_rho_d_lambda_(lambda_rho, T, E):
    lambda_, rho = lambda_rho
    return (-1. / lambda_) * (E - (lambda_ * T) ** rho - rho * (lambda_ * T) ** rho * np.log(lambda_ * T)).sum()
