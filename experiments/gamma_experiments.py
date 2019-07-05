# -*- coding: utf-8 -*-
from autograd import numpy as np
from lifelines.fitters import KnownModelParametericUnivariateFitter, ParametericUnivariateFitter
from lifelines.utils.gamma import lower_inc_gamma
from autograd.scipy.special import gamma

"""
class GG(ParametericUnivariateFitter):

    _fitted_parameter_names = ["alpha_", "lambda_", "rho_"]

    def _cumulative_hazard(self, params, times):
        alpha_, lambda_, rho_ = params
        return -log_regularized_upper_inc_gamma(alpha_ / rho_, (times / lambda_) ** rho_)



gg = GG()
gg.fit(df['T'].astype(np.float64)/70., df['E'])

gg.print_summary()

"""

from lifelines.datasets import load_waltons

df = load_waltons()


class LogGammaFitter(ParametericUnivariateFitter):

    _fitted_parameter_names = ["lambda_", "delta_", "kappa_"]

    _bounds = [(0.0, None), (0.0, None), (0.0, None)]

    def _cumulative_hazard(self, params, times):

        lambda_, delta_, kappa_ = params
        z = np.exp((times - delta_) / lambda_)
        v = -log_regularized_upper_inc_gamma(kappa_, z)

        return v


lg = LogGammaFitter()
lg.fit(df["T"] / 70.0, df["E"])

lg.print_summary()
