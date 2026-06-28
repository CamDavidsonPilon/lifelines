# -*- coding: utf-8 -*-
"""
Royston-Parmar Flexible Parametric Proportional Hazards Model.

Reference:
    Royston, P., & Parmar, M. K. B. (2002). Flexible parametric proportional-hazards
    and proportional-odds models for censored survival data, with application to
    prognostic modelling and estimation of treatment effects. Statistics in Medicine,
    21(15), 2175-2197. doi:10.1002/sim.1203
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq

from lifelines.fitters import BaseFitter
from lifelines.utils import CensoringType


__all__ = ["FlexibleParametricPHFitter"]


# ---------------------------------------------------------------------------
# Restricted Natural Cubic Spline helpers
# ---------------------------------------------------------------------------

def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Compute the restricted natural cubic spline design matrix.

    Given K knots (boundary + interior), returns a matrix with columns
    [1, x, v_1(x), ..., v_{K-3}(x)]  (i.e. K-1 columns, no intercept counted
    separately — intercept is the first column).

    Parameters
    ----------
    x : array-like, shape (n,)
        Values at which to evaluate the basis (typically log(t)).
    knots : array-like, shape (K,) sorted ascending
        Knot locations in the same scale as x.

    Returns
    -------
    B : ndarray, shape (n, K-1)
        Design matrix.  The first column is 1 (intercept), second is x,
        remaining K-3 columns are RCS cubic terms.
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    K = len(knots)
    n = x.shape[0]

    # For K knots the RCS has K-1 free parameters:
    #   K=2 -> 1 col (intercept only, i.e. Exponential baseline)
    #   K=3 -> 2 cols (intercept + linear, i.e. Weibull baseline)
    #   K>=4 -> K-1 cols (intercept + linear + K-3 cubic restriction terms)
    n_cols = K - 1
    B = np.empty((n, n_cols), dtype=float)
    B[:, 0] = 1.0
    if n_cols >= 2:
        B[:, 1] = x

    if n_cols >= 3:
        t_max = knots[-1]
        interior = knots[1:-1]  # length K-2 >= 2 when K>=4

        for j, t_j in enumerate(interior[:-1]):  # gives K-3 columns (indices 2 .. K-2)
            lam = (t_max - t_j) / (t_max - knots[-2])
            def _relu3(a, c):
                return np.maximum(a - c, 0.0) ** 3
            v = _relu3(x, t_j) - lam * _relu3(x, knots[-2]) + (1 - lam) * _relu3(x, t_max)
            B[:, 2 + j] = v

    return B


def _rcs_basis_deriv(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Derivative d/dx of the RCS basis (same columns as _rcs_basis).

    Returns
    -------
    dB : ndarray, shape (n, K-1)
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    K = len(knots)
    n = x.shape[0]

    n_cols = K - 1
    dB = np.empty((n, n_cols), dtype=float)
    dB[:, 0] = 0.0   # d(1)/dx = 0
    if n_cols >= 2:
        dB[:, 1] = 1.0   # d(x)/dx = 1

    if n_cols >= 3:
        interior = knots[1:-1]

        for j, t_j in enumerate(interior[:-1]):
            lam = (knots[-1] - t_j) / (knots[-1] - knots[-2])
            def _relu2(a, c):
                return np.where(a > c, 3.0 * (a - c) ** 2, 0.0)
            dv = _relu2(x, t_j) - lam * _relu2(x, knots[-2]) + (1 - lam) * _relu2(x, knots[-1])
            dB[:, 2 + j] = dv

    return dB


# ---------------------------------------------------------------------------
# Main fitter class
# ---------------------------------------------------------------------------

class FlexibleParametricPHFitter(BaseFitter):
    r"""
    Royston-Parmar Flexible Parametric Proportional Hazards Model.

    Models the log cumulative hazard as a restricted natural cubic spline
    (RCS) in log(t), with optional linear covariate effects:

    .. math::

        \log H(t \mid x) = s(\log t;\, \boldsymbol\gamma) + \mathbf{x}^\top \boldsymbol\beta

    where :math:`s` is a restricted natural cubic spline with `n_baseline_knots`
    knots placed at quantiles of the log event times, and
    :math:`\boldsymbol\gamma` (the spline coefficients) together with
    :math:`\boldsymbol\beta` (covariate effects) are estimated by maximum
    likelihood.

    Parameters
    ----------
    n_baseline_knots : int, optional (default 4)
        Total number of knots (boundary + interior).  Must be >= 2.
        The model has (n_baseline_knots - 1) spline parameters.
    knot_locations : array-like or None, optional
        If provided, overrides automatic knot placement.  Must be in the
        *original time* scale (not log-scale); the fitter converts internally.
        Length must equal n_baseline_knots when provided.
    penalizer : float, optional (default 0.0)
        L2 penalty on the spline coefficients gamma (not on beta).

    Attributes
    ----------
    knots_ : ndarray
        Knot locations in log-time scale.
    params_ : Series
        Estimated parameters (gamma_0 .. gamma_k, then covariate betas).
    log_likelihood_ : float
    AIC_ : float
    baseline_cumulative_hazard_ : DataFrame
        H_0(t) on a fine grid (columns: ['baseline cumulative hazard']).
    baseline_survival_ : DataFrame

    References
    ----------
    Royston, P., & Parmar, M. K. B. (2002). Statistics in Medicine, 21(15), 2175-2197.
    """

    def __init__(
        self,
        n_baseline_knots: int = 4,
        knot_locations: Optional[np.ndarray] = None,
        penalizer: float = 0.0,
    ):
        if knot_locations is None and n_baseline_knots < 2:
            raise ValueError("n_baseline_knots must be >= 2.")
        self.n_baseline_knots = n_baseline_knots
        self.knot_locations = knot_locations
        self.penalizer = penalizer
        super().__init__()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @CensoringType.right_censoring
    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        covariates: Optional[List[str]] = None,
    ) -> "FlexibleParametricPHFitter":
        """
        Fit the Royston-Parmar model.

        Parameters
        ----------
        df : pd.DataFrame
        duration_col : str
            Column name for observed times (must be > 0).
        event_col : str
            Column name for event indicator (1 = event, 0 = censored).
        covariates : list of str or None
            Columns to include as linear proportional-hazards covariates.
            If None, fits the baseline model only.

        Returns
        -------
        self
        """
        self.duration_col = duration_col
        self.event_col = event_col
        self.covariates = list(covariates) if covariates is not None else []

        T = df[duration_col].values.astype(float)
        E = df[event_col].values.astype(bool)

        if np.any(T <= 0):
            raise ValueError("All durations must be strictly positive.")

        # Keep weights and event_observed for BaseFitter __repr__
        self.weights = np.ones(len(T))
        self.event_observed = E.astype(float)
        self.durations = T

        # ---- Covariate matrix ----------------------------------------
        if self.covariates:
            X = df[self.covariates].values.astype(float)
        else:
            X = np.zeros((len(T), 0))

        # ---- Knot placement ------------------------------------------
        log_event_times = np.log(T[E])
        if self.knot_locations is not None:
            knots = np.log(np.asarray(self.knot_locations, dtype=float))
            self.n_baseline_knots = len(knots)
        else:
            percentiles = np.linspace(0, 100, self.n_baseline_knots)
            knots = np.percentile(log_event_times, percentiles)

        self.knots_ = knots
        K = len(knots)         # total knots
        n_spline = K - 1       # spline params: intercept + linear + (K-3) cubic

        # ---- Initial parameters -------------------------------------
        # gamma: start with intercept ~ log(-log(0.5)) scaled by median T,
        #        slope ~ 1 (Weibull-like), rest 0
        gamma0 = np.zeros(n_spline)
        gamma0[0] = np.log(-np.log(0.5))  # rough intercept for median ~ 1
        if n_spline > 1:
            gamma0[1] = 1.0
        beta0 = np.zeros(X.shape[1])
        params0 = np.concatenate([gamma0, beta0])

        # ---- Objective -----------------------------------------------
        log_t = np.log(T)

        def neg_log_likelihood(params):
            gamma = params[:n_spline]
            beta = params[n_spline:]
            return _neg_ll(gamma, beta, log_t, E, X, knots, self.penalizer)

        def neg_log_likelihood_grad(params):
            gamma = params[:n_spline]
            beta = params[n_spline:]
            return _neg_ll_grad(gamma, beta, log_t, E, X, knots, self.penalizer)

        # ---- Optimise ------------------------------------------------
        result = minimize(
            neg_log_likelihood,
            params0,
            jac=neg_log_likelihood_grad,
            method="L-BFGS-B",
            options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-7},
        )

        if not result.success:
            warnings.warn(
                "Optimization did not converge: %s" % result.message,
                RuntimeWarning,
            )

        params_hat = result.x
        gamma_hat = params_hat[:n_spline]
        beta_hat = params_hat[n_spline:]

        # ---- Store results ------------------------------------------
        gamma_names = ["gamma%d_" % i for i in range(n_spline)]
        beta_names = list(self.covariates)
        all_names = gamma_names + beta_names
        self.params_ = pd.Series(params_hat, index=all_names)
        self._gamma = gamma_hat
        self._beta = beta_hat

        self.log_likelihood_ = -result.fun
        n_params = len(params_hat)
        self.AIC_ = 2 * n_params - 2 * self.log_likelihood_

        # ---- Baseline functions on fine grid -------------------------
        t_min = np.exp(knots[0])
        t_max = np.exp(knots[-1])
        fine_times = np.linspace(t_min, t_max, 500)
        # Pass a single dummy row for X (baseline, no covariates)
        H0 = self._cumulative_hazard_internal(gamma_hat, np.zeros((1, 0)), fine_times)
        H0_vec = H0[:, 0]  # (500,)
        self.baseline_cumulative_hazard_ = pd.DataFrame(
            H0_vec, index=fine_times, columns=["baseline cumulative hazard"]
        )
        self.baseline_survival_ = pd.DataFrame(
            np.exp(-H0_vec), index=fine_times, columns=["baseline survival"]
        )

        return self

    def predict_cumulative_hazard(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Return cumulative hazard H(t | x) for each subject in X at each time.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_subjects, n_covariates)
            Covariate matrix.  If DataFrame, columns must match covariates
            used during fit.  For baseline (no covariates) pass an empty
            DataFrame or zeros array.
        times : array-like or None
            Times at which to evaluate.  Defaults to 200 quantile-spaced
            points covering the training event time range.

        Returns
        -------
        DataFrame, shape (n_times, n_subjects)
        """
        X_arr, subject_labels = self._prepare_X(X)
        times = self._default_times(times)

        H = self._cumulative_hazard_internal(self._gamma, X_arr, times)
        return pd.DataFrame(H, index=times, columns=subject_labels)

    def predict_survival_function(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Return survival probability S(t | x) = exp(-H(t | x)).

        Returns
        -------
        DataFrame, shape (n_times, n_subjects)
        """
        H = self.predict_cumulative_hazard(X, times)
        return np.exp(-H)

    def predict_median(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.Series:
        """
        Return the predicted median survival time for each subject (the time
        where S(t | x) = 0.5, i.e. H(t | x) = log(2)).

        Returns
        -------
        Series, indexed by subject labels
        """
        X_arr, subject_labels = self._prepare_X(X)
        t_min = np.exp(self.knots_[0])
        t_max = np.exp(self.knots_[-1])

        medians = []
        target = np.log(2.0)  # H = log(2) <=> S = 0.5

        # RCS extrapolates linearly outside boundary knots; that linear slope
        # can be negative, so we must not search arbitrarily far outside the
        # knot range.  Instead scan a grid inside [t_min/100, t_max * 200]
        # to find a valid (lo, hi) bracket then refine with brentq.
        scan_lo = t_min * 1e-4
        scan_hi = t_max * 200.0
        scan_grid = np.linspace(np.log(scan_lo), np.log(scan_hi), 2000)
        scan_times = np.exp(scan_grid)

        for i in range(X_arr.shape[0]):
            xi = X_arr[i : i + 1, :]
            H_scan = self._cumulative_hazard_internal(
                self._gamma, xi, scan_times
            )[:, 0]  # (n_scan,)

            def h_minus_target(t):
                H = self._cumulative_hazard_internal(self._gamma, xi, np.array([t]))
                return H[0, 0] - target

            try:
                # Find first crossing where H goes from < target to >= target
                below = H_scan < target
                above = ~below
                cross_idx = np.where(below[:-1] & above[1:])[0]

                if len(cross_idx) == 0:
                    # H never reaches target in scan range
                    if H_scan[0] >= target:
                        medians.append(scan_times[0])
                    else:
                        medians.append(np.inf)
                    continue

                idx = cross_idx[0]
                lo_t = scan_times[idx]
                hi_t = scan_times[idx + 1]
                med = brentq(h_minus_target, lo_t, hi_t, xtol=1e-6, maxiter=200)
                medians.append(med)
            except Exception:
                medians.append(np.nan)

        return pd.Series(medians, index=subject_labels, name="median")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cumulative_hazard_internal(
        self,
        gamma: np.ndarray,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """
        Compute H(t | x) for all times and subjects.

        Parameters
        ----------
        gamma : (n_spline,)
        X : (n_subjects, n_covariates) — may be (n, 0) for baseline only
        times : (n_times,)

        Returns
        -------
        H : (n_times, n_subjects)
        """
        n_subjects = X.shape[0]
        log_t = np.log(np.clip(times, 1e-300, np.inf))
        B = _rcs_basis(log_t, self.knots_)            # (n_times, n_spline)
        log_H0 = B @ gamma                            # (n_times,)

        if X.shape[1] > 0:
            Xbeta = X @ self._beta                    # (n_subjects,)
            log_H = log_H0[:, np.newaxis] + Xbeta[np.newaxis, :]
        else:
            # Baseline-only: replicate the baseline curve for each subject
            log_H = np.tile(log_H0[:, np.newaxis], (1, max(n_subjects, 1)))

        return np.exp(log_H)

    def _prepare_X(self, X):
        """Convert X to ndarray and extract subject labels."""
        if isinstance(X, pd.DataFrame):
            subject_labels = X.index
            if self.covariates:
                X_arr = X[self.covariates].values.astype(float)
            else:
                X_arr = np.zeros((len(X), 0))
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            subject_labels = list(range(X.shape[0]))
            X_arr = X.astype(float)
            if len(self.covariates) == 0:
                X_arr = np.zeros((X.shape[0], 0))
        else:
            raise TypeError("X must be a DataFrame or ndarray.")
        return X_arr, subject_labels

    def _default_times(self, times):
        if times is None:
            t_min = np.exp(self.knots_[0])
            t_max = np.exp(self.knots_[-1])
            times = np.linspace(t_min, t_max, 200)
        return np.asarray(times, dtype=float)


# ---------------------------------------------------------------------------
# Log-likelihood and gradient (pure numpy, no autograd)
# ---------------------------------------------------------------------------

def _neg_ll(
    gamma: np.ndarray,
    beta: np.ndarray,
    log_t: np.ndarray,
    E: np.ndarray,
    X: np.ndarray,
    knots: np.ndarray,
    penalizer: float,
) -> float:
    """
    Negative log-likelihood for the Royston-Parmar PH model.

    log L = sum_i { d_i * [log(dH/dt)(t_i|x_i)] - H(t_i|x_i) }

    where log H(t|x) = B(log t) @ gamma + x @ beta
    and dH/dt = H(t|x) * (dB/d(log t) @ gamma) / t

    so log(dH/dt) = log H + log(dB/d(log t) @ gamma) - log t
    """
    B = _rcs_basis(log_t, knots)          # (n, n_spline)
    dB = _rcs_basis_deriv(log_t, knots)   # (n, n_spline)

    log_H0 = B @ gamma                    # (n,)

    if X.shape[1] > 0:
        Xbeta = X @ beta
    else:
        Xbeta = 0.0

    log_H = log_H0 + Xbeta               # (n,)
    H = np.exp(log_H)                     # (n,)

    # derivative of spline w.r.t. log_t
    s_prime = dB @ gamma                  # (n,)
    # h(t) = H(t) * s'(log t) / t  => log h = log H + log(s') - log t
    # guard against s_prime <= 0 (non-monotone region) with a small floor
    s_prime_safe = np.where(s_prime > 0, s_prime, 1e-300)
    log_h = log_H + np.log(s_prime_safe) - log_t

    ll = np.sum(E * log_h - H)

    # L2 penalty on gamma (not beta)
    if penalizer > 0:
        ll -= penalizer * np.sum(gamma ** 2)

    return -ll


def _neg_ll_grad(
    gamma: np.ndarray,
    beta: np.ndarray,
    log_t: np.ndarray,
    E: np.ndarray,
    X: np.ndarray,
    knots: np.ndarray,
    penalizer: float,
) -> np.ndarray:
    """
    Analytical gradient of the negative log-likelihood.
    """
    B = _rcs_basis(log_t, knots)
    dB = _rcs_basis_deriv(log_t, knots)

    log_H0 = B @ gamma

    if X.shape[1] > 0:
        Xbeta = X @ beta
    else:
        Xbeta = 0.0

    log_H = log_H0 + Xbeta
    H = np.exp(log_H)

    s_prime = dB @ gamma
    s_prime_safe = np.where(s_prime > 0, s_prime, 1e-300)

    # --- gradient w.r.t. gamma ----------------------------------------
    # ll = sum_i { d_i * (B_i @ g  + log(dB_i @ g) - log t_i)  - exp(B_i@g + Xb) }
    # dll/dg_k = sum_i { d_i * (B_ik + dB_ik / s'_i)  - H_i * B_ik }
    d_gamma = np.sum(
        E[:, None] * (B + dB / s_prime_safe[:, None]) - H[:, None] * B,
        axis=0,
    )

    # penalty contribution
    if penalizer > 0:
        d_gamma -= 2 * penalizer * gamma

    # --- gradient w.r.t. beta -----------------------------------------
    if X.shape[1] > 0:
        # dll/db = sum_i { d_i * x_i  - H_i * x_i }
        d_beta = np.sum((E - H)[:, None] * X, axis=0)
    else:
        d_beta = np.zeros(0)

    grad_ll = np.concatenate([d_gamma, d_beta])
    return -grad_ll
