# -*- coding: utf-8 -*-
from autograd import numpy as np
from lifelines.fitters import KnownModelParametericUnivariateFitter, ParametericUnivariateFitter
from lifelines.utils.gamma import gammainc, gammaincc


class GG(ParametericUnivariateFitter):

    _fitted_parameter_names = ["alpha_", "lambda_", "rho_"]
    _bounds = [(0.0, None), (0.0, None), (0.0, None)]

    def _cumulative_hazard(self, params, times):
        alpha_, lambda_, rho_ = params
        lg = np.clip(gammaincc(alpha_ / rho_, (times / lambda_) ** rho_), 1e-20, 1 - 1e-20)
        return -np.log(lg)


from lifelines.datasets import load_waltons

df = load_waltons()

gg = GG()
gg.fit(df["T"], df["E"])

gg.print_summary(3)


class LogGammaFitter(ParametericUnivariateFitter):

    _fitted_parameter_names = ["lambda_", "delta_", "kappa_"]

    _bounds = [(0.0, None), (0.0, None), (0.0, None)]
    _initial_values = np.array([1.0, 1.0, 1.0])

    def _cumulative_hazard(self, params, times):
        lambda_, delta_, kappa_ = params
        z = np.exp((times - delta_) / lambda_)
        lg = np.clip(gammaincc(kappa_, z), 1e-20, 1 - 1e-20)
        v = -np.log(lg)
        return v


lg = LogGammaFitter()
lg.fit(df["T"], df["E"])

lg.print_summary(3)
