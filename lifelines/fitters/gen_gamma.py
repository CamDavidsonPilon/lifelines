# -*- coding: utf-8 -*-
from __future__ import print_function, division
import autograd.numpy as np
from autograd.scipy.special import gamma
from lifelines.utils.gamma import gammainc
from autograd import jacobian, hessian

from lifelines.fitters import ParametericUnivariateFitter


class GeneralizedGammaFitter(ParametericUnivariateFitter):

    _fitted_parameter_names = ["lambda_", "p_", "alpha_"]

    def _cumulative_hazard(self, params, times):
        lambda_, p_, alpha_ = params
        return np.log(gamma(alpha_ / p_)) + np.log(gammainc(alpha_ / p_, (lambda_ * times) ** p_)) + 10


T = np.sort(np.random.exponential(5, 500))

fitter = GeneralizedGammaFitter()

print(jacobian(fitter._cumulative_hazard)(fitter._initial_values, T))

GeneralizedGammaFitter().fit(T)
