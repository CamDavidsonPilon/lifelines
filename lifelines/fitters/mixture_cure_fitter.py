# -*- coding: utf-8 -*-
import autograd.numpy as anp
from autograd.scipy.special import expit
from lifelines.fitters import ParametricUnivariateFitter

# What should the name of the fitter actually be?
class MixtureCureFitter(ParametricUnivariateFitter):
    def __init__(self, base_fitter, *args, **kwargs):
        self._base_fitter = base_fitter

        # What should the variable that we use be called?
        # Should raise an exception if that variable is in the base fitted parameter names

        self._fitted_parameter_names = ["cured_"] + base_fitter._fitted_parameter_names
        self._bounds = [(None, None)] + base_fitter._bounds
        super().__init__(*args, **kwargs)

    def _fit(self, *args, **kwargs):
        self._base_fitter._censoring_type = self._censoring_type
        result = super()._fit(*args, **kwargs)
        for param_name, fitted_value in zip(self._fitted_parameter_names[1:], self._fitted_parameters_[1:]):
            setattr(self._base_fitter, param_name, fitted_value)
        return result

    def _cumulative_hazard(self, params, times):
        c = expit(params[0])
        sf = self._base_fitter._survival_function(params[1:], times)
        return -anp.log(c + (1 - c) * sf)

    def _create_initial_point(self, Ts, E, *args):
        base_point = self._base_fitter._create_initial_point(Ts, E, *args)
        return anp.array([0] + list(base_point))

    def percentile(self, p):
        c = expit(self.cured_)

        if p <= c:
            raise ValueError("Percentile must be larger than the cure fraction")

        non_cure_p = (p - c) / (1 - c)
        return self._base_fitter.percentile(non_cure_p)
