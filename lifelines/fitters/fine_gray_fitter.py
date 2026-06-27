# -*- coding: utf-8 -*-
"""
Fine-Gray proportional subdistribution hazard model for competing risks.

References
----------
Fine, J. P. and Gray, R. J. (1999). A proportional hazards model for the
subdistribution of a competing risk. Journal of the American Statistical
Association, 94(446):496-509.
"""
from __future__ import annotations

import warnings
from textwrap import dedent
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import LinAlgError, solve as spsolve

from lifelines.exceptions import ConvergenceWarning, ConvergenceError
from lifelines.fitters import SemiParametricRegressionFitter
from lifelines.utils import CensoringType, check_nans_or_infs
from lifelines.utils.printer import Printer


__all__ = ["FineGrayFitter"]

_CONVERGENCE_DOCS = dedent(
    """\
    Hints for fixing convergence issues:
      - Scale your covariates to zero mean and unit variance.
      - Try a smaller `step_size` (e.g., 0.50).
      - Check for highly correlated covariates and remove redundancies.
    """
)


def _km_censoring_survival(T: np.ndarray, E: np.ndarray) -> tuple:
    """
    Estimate G(t) = P(censoring > t) via Kaplan-Meier.

    Censoring is the "event" here; actual observed events are treated as
    censored in this KM.  Returns (times, G_values) as 1-D arrays, where
    G_values[i] = P(C > times[i]).
    """
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(E == 0))
    sf = kmf.survival_function_
    return sf.index.values.astype(float), sf.iloc[:, 0].values.astype(float)


def _eval_step(times_sorted: np.ndarray, values: np.ndarray, t: Union[float, np.ndarray]) -> np.ndarray:
    """
    Evaluate a right-continuous step function defined at (times_sorted, values)
    at one or many query points.  Returns the value just after the last step
    that is <= t, or 1.0 if t < times_sorted[0].
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    idx = np.searchsorted(times_sorted, t, side="right") - 1
    result = np.where(idx < 0, 1.0, values[np.clip(idx, 0, len(values) - 1)])
    return float(result[0]) if scalar else result


def _compute_fine_gray_derivatives(
    X: np.ndarray,
    T: np.ndarray,
    event_k: np.ndarray,
    competing: np.ndarray,
    G_times: np.ndarray,
    G_vals: np.ndarray,
    beta: np.ndarray,
    event_times_k: np.ndarray,
) -> tuple:
    """
    Return (gradient, hessian, log_partial_likelihood) for Fine-Gray model.

    The modified risk set R̃(t) at each event time t includes:
      - All subjects with T_i >= t (standard risk set), weight = 1
      - Subjects with a competing event at T_i < t, IPCW weight = G(T_i)/G(t)
    Censored subjects and subjects with event-of-interest at T_i < t are excluded.
    """
    n, d = X.shape
    scores = np.exp(X @ beta)  # (n,)

    log_lik = 0.0
    gradient = np.zeros(d)
    hessian = np.zeros((d, d))

    for tj in event_times_k:
        G_tj = _eval_step(G_times, G_vals, tj)
        if G_tj < 1e-12:
            continue

        # --- modified risk set weights ---
        w = np.zeros(n)
        w[T >= tj] = 1.0

        comp_before = competing & (T < tj)
        if comp_before.any():
            G_Ti = _eval_step(G_times, G_vals, T[comp_before])
            w[comp_before] = np.where(G_Ti > 0, G_Ti / G_tj, 0.0)

        ws = w * scores  # (n,) weighted scores
        S0 = ws.sum()
        if S0 < 1e-12:
            continue

        S1 = X.T @ ws  # (d,)
        mean_x = S1 / S0

        # S2 = X' diag(ws) X  =  (d, d)
        Xw = X * ws[:, None]  # (n, d)
        S2 = Xw.T @ X  # (d, d)

        mask_j = event_k & (T == tj)
        n_j = int(mask_j.sum())
        if n_j == 0:
            continue

        sum_x_j = X[mask_j].sum(0)  # (d,)

        log_lik += float(sum_x_j @ beta) - n_j * np.log(S0)
        gradient += sum_x_j - n_j * mean_x
        hessian -= n_j * (S2 / S0 - np.outer(mean_x, mean_x))

    return gradient, hessian, log_lik


def _breslow_baseline(
    X: np.ndarray,
    T: np.ndarray,
    event_k: np.ndarray,
    competing: np.ndarray,
    G_times: np.ndarray,
    G_vals: np.ndarray,
    beta: np.ndarray,
    event_times_k: np.ndarray,
) -> pd.Series:
    """
    Breslow estimator for the baseline cumulative subdistribution hazard Λ̃₀(t).
    """
    n = len(T)
    scores = np.exp(X @ beta)

    dH = []
    for tj in event_times_k:
        G_tj = _eval_step(G_times, G_vals, tj)
        if G_tj < 1e-12:
            dH.append(0.0)
            continue

        w = np.zeros(n)
        w[T >= tj] = 1.0
        comp_before = competing & (T < tj)
        if comp_before.any():
            G_Ti = _eval_step(G_times, G_vals, T[comp_before])
            w[comp_before] = np.where(G_Ti > 0, G_Ti / G_tj, 0.0)

        S0 = (w * scores).sum()
        mask_j = event_k & (T == tj)
        n_j = int(mask_j.sum())
        dH.append(n_j / S0 if S0 > 1e-12 else 0.0)

    cumh = np.cumsum(dH)
    return pd.Series(cumh, index=event_times_k, name="baseline_cumulative_subdistribution_hazard_")


class FineGrayFitter(SemiParametricRegressionFitter):
    r"""
    Fine-Gray proportional subdistribution hazard model for competing risks.

    Fits the model

    .. math::

        \tilde{h}_k(t \mid x) = \tilde{h}_{k0}(t) \exp(x^\top \beta)

    where :math:`\tilde{h}_k` is the subdistribution hazard for the event of
    interest :math:`k`, related to the cumulative incidence function (CIF) by

    .. math::

        F_k(t \mid x) = 1 - \exp\!\left(-\int_0^t \tilde{h}_{k0}(s)\,ds \cdot e^{x^\top \beta}\right).

    Estimation uses the two-weight IPCW partial likelihood of Fine & Gray (1999).
    Censored subjects are excluded from the modified risk set; subjects who
    experienced a competing event before time :math:`t` are re-entered with
    an inverse-probability-of-censoring weight :math:`G(T_i^-)/G(t^-)`, where
    :math:`G(\cdot)` is the Kaplan-Meier estimate of the censoring survival
    function.

    Parameters
    ----------
    alpha : float, optional (default 0.05)
        Significance level for confidence intervals.

    Attributes
    ----------
    params_ : Series
        Estimated log subdistribution hazard ratios, shape (p,).
    variance_matrix_ : DataFrame
        Estimated covariance matrix of params_, shape (p, p).
    confidence_intervals_ : DataFrame
        (p, 2) confidence intervals for each parameter.
    baseline_cumulative_subdistribution_hazard_ : Series
        Breslow estimate of the baseline cumulative subdistribution hazard,
        indexed by unique event times.
    event_of_interest : int
        The value of *event_col* that was treated as the event of interest.
    log_likelihood_ : float
        Value of the partial log-likelihood at the fitted parameters.
    AIC_partial_ : float
        Partial AIC: :math:`-2\ell + 2p`.
    BIC_partial_ : float
        Partial BIC: :math:`-2\ell + p\ln(d)`, where *d* = number of events of
        interest.

    Examples
    --------
    .. code:: python

        import pandas as pd
        from lifelines import FineGrayFitter, AalenJohansenFitter

        df = pd.DataFrame({
            'T':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'E':  [1, 2, 1, 0, 1, 2, 0, 1, 2,  1],
            'x1': [0, 1, 0, 1, 0, 1, 0, 1, 0,  1],
        })

        fgf = FineGrayFitter()
        fgf.fit(df, duration_col='T', event_col='E', event_of_interest=1)
        fgf.print_summary()
        fgf.predict_cumulative_incidence(df, times=[3, 6, 9])

    References
    ----------
    Fine, J. P. and Gray, R. J. (1999). A proportional hazards model for the
    subdistribution of a competing risk. *Journal of the American Statistical
    Association*, 94(446):496-509.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__(alpha=alpha)

    @CensoringType.right_censoring
    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        event_of_interest: int,
        weights_col: Optional[str] = None,
        show_progress: bool = False,
        initial_point: Optional[np.ndarray] = None,
        step_size: float = 0.95,
        max_steps: int = 200,
        precision: float = 1e-7,
    ) -> "FineGrayFitter":
        """
        Fit the Fine-Gray model.

        Parameters
        ----------
        df : DataFrame
            Must contain columns for *duration_col* and *event_col*, plus any
            covariates.  Non-numeric or unused columns should be dropped before
            calling.
        duration_col : str
            Column name for observed times (must be non-negative).
        event_col : str
            Column name for event indicator.  0 = right-censored;
            *event_of_interest* = event of interest; any other positive integer
            = competing event.
        event_of_interest : int
            The integer value in *event_col* that denotes the event to model.
        weights_col : str, optional
            Column of case-weights.  Default: all weights are 1.
        show_progress : bool, optional
            Print iteration details.
        initial_point : array of shape (p,), optional
            Starting values for beta (default: zero vector).
        step_size : float, optional (default 0.95)
            Newton-Raphson damping factor.  Reduce toward 0.5 if convergence
            is slow.
        max_steps : int, optional (default 200)
            Maximum Newton-Raphson iterations.
        precision : float, optional (default 1e-7)
            Convergence threshold on the gradient norm.

        Returns
        -------
        self
        """
        df = df.copy()

        if duration_col not in df.columns:
            raise ValueError("'%s' not found in DataFrame columns." % duration_col)
        if event_col not in df.columns:
            raise ValueError("'%s' not found in DataFrame columns." % event_col)

        T = df[duration_col].values.astype(float)
        E = df[event_col].values.astype(int)

        if (T < 0).any():
            raise ValueError("All durations must be >= 0.")
        check_nans_or_infs(pd.Series(T))
        check_nans_or_infs(pd.Series(E.astype(float)))

        if event_of_interest not in E:
            raise ValueError(
                "event_of_interest=%d not found in event_col '%s'." % (event_of_interest, event_col)
            )

        # weights
        if weights_col is not None:
            weights = df[weights_col].values.astype(float)
        else:
            weights = np.ones(len(T))

        # drop non-covariate columns
        drop_cols = [duration_col, event_col]
        if weights_col is not None:
            drop_cols.append(weights_col)
        X_df = df.drop(columns=drop_cols)

        # require numeric covariates
        non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise TypeError(
                "Covariates must be numeric. Non-numeric columns: %s. "
                "Encode categoricals before fitting." % non_numeric
            )

        covariate_names = list(X_df.columns)
        X = X_df.values.astype(float)  # (n, d)
        n, d = X.shape

        event_k = E == event_of_interest
        competing = (E != 0) & (E != event_of_interest)

        n_events_k = int(event_k.sum())
        if n_events_k == 0:
            raise ValueError("No events of type %d found." % event_of_interest)

        # --- estimate censoring survival G(t) ---
        G_times, G_vals = _km_censoring_survival(T, E)

        event_times_k = np.sort(np.unique(T[event_k]))

        # --- Newton-Raphson optimisation ---
        beta = np.zeros(d) if initial_point is None else np.asarray(initial_point, dtype=float).copy()

        converged = False
        for step_i in range(max_steps):
            gradient, hessian, log_lik = _compute_fine_gray_derivatives(
                X, T, event_k, competing, G_times, G_vals, beta, event_times_k
            )
            grad_norm = np.linalg.norm(gradient)

            if show_progress:
                print("Iteration %4d | log-lik = %12.6f | ||grad|| = %.4e" % (step_i, log_lik, grad_norm))

            if grad_norm < precision:
                converged = True
                break

            try:
                delta = spsolve(-hessian, gradient)
            except (LinAlgError, ValueError):
                warnings.warn(
                    "Hessian was singular at iteration %d. Fitting may be unreliable.\n%s"
                    % (step_i, _CONVERGENCE_DOCS),
                    ConvergenceWarning,
                    stacklevel=2,
                )
                break

            # damping
            beta = beta + step_size * delta

        if not converged:
            warnings.warn(
                "FineGrayFitter did not converge. Try increasing max_steps or scaling covariates.\n%s"
                % _CONVERGENCE_DOCS,
                ConvergenceWarning,
                stacklevel=2,
            )

        # --- final quantities at beta_hat ---
        gradient, hessian, log_lik = _compute_fine_gray_derivatives(
            X, T, event_k, competing, G_times, G_vals, beta, event_times_k
        )

        try:
            variance_matrix = np.linalg.inv(-hessian)
        except LinAlgError:
            warnings.warn(
                "Could not invert the Hessian. Variance estimates may be unreliable.",
                ConvergenceWarning,
                stacklevel=2,
            )
            variance_matrix = np.full((d, d), np.nan)

        se = np.sqrt(np.diag(variance_matrix).clip(0))
        z = stats.norm.ppf(1 - self.alpha / 2)

        self.params_ = pd.Series(beta, index=covariate_names, name="coef")
        self.variance_matrix_ = pd.DataFrame(variance_matrix, index=covariate_names, columns=covariate_names)
        self.standard_errors_ = pd.Series(se, index=covariate_names, name="se(coef)")
        self.confidence_intervals_ = pd.DataFrame(
            {
                "%.2f lower-bound" % self.alpha: beta - z * se,
                "%.2f upper-bound" % self.alpha: beta + z * se,
            },
            index=covariate_names,
        )

        self.baseline_cumulative_subdistribution_hazard_ = _breslow_baseline(
            X, T, event_k, competing, G_times, G_vals, beta, event_times_k
        )
        self.baseline_cumulative_incidence_ = pd.Series(
            1.0 - np.exp(-self.baseline_cumulative_subdistribution_hazard_.values),
            index=self.baseline_cumulative_subdistribution_hazard_.index,
            name="baseline_cumulative_incidence_",
        )

        self.log_likelihood_ = log_lik
        self.event_col = event_col
        self.duration_col = duration_col
        self.event_of_interest = event_of_interest
        self._n_training_rows = n
        self._n_events_k = n_events_k
        self._covariate_names = covariate_names
        self._G_times = G_times
        self._G_vals = G_vals
        self._training_df = df  # retained for plot_partial_effects_on_outcome
        self._converged = converged
        return self

    @property
    def AIC_partial_(self) -> float:
        """Partial AIC: -2 * log_lik + 2 * n_params."""
        return -2 * self.log_likelihood_ + 2 * len(self.params_)

    @property
    def BIC_partial_(self) -> float:
        """Partial BIC: -2 * log_lik + n_params * log(n_events_k)."""
        return -2 * self.log_likelihood_ + len(self.params_) * np.log(self._n_events_k)

    def predict_cumulative_incidence(
        self,
        X: pd.DataFrame,
        times: Optional[Union[Iterable[float], float]] = None,
    ) -> pd.DataFrame:
        r"""
        Predict the cumulative incidence function (CIF) for each subject.

        .. math::

            F_k(t \mid x) = 1 - \exp\!\bigl(-e^{x^\top \hat{\beta}} \cdot \hat{\Lambda}_0(t)\bigr)

        Parameters
        ----------
        X : DataFrame
            Must contain the same covariate columns used during ``fit``.
            Each row is one subject.
        times : float or iterable of floats, optional
            Query times.  If omitted, the fitted event times are used.

        Returns
        -------
        DataFrame
            Shape (len(times), len(X)).  Entry [t, i] is the predicted CIF
            for subject i at time t.
        """
        self._check_is_fitted()

        x_vals = X[self._covariate_names].values.astype(float)
        lp = x_vals @ self.params_.values  # (n_subjects,) linear predictor

        baseline = self.baseline_cumulative_subdistribution_hazard_
        baseline_times = baseline.index.values
        baseline_cumh = baseline.values

        if times is None:
            query_times = baseline_times
        else:
            query_times = np.atleast_1d(np.asarray(times, dtype=float))

        # interpolate baseline cumulative hazard at query times
        cumh_at_t = np.interp(query_times, baseline_times, baseline_cumh, left=0.0, right=baseline_cumh[-1])

        # CIF = 1 - exp(-exp(lp) * Λ₀(t))
        # shape: (len(query_times), n_subjects)
        cif = 1.0 - np.exp(-np.outer(cumh_at_t, np.exp(lp)))

        subject_labels = X.index if isinstance(X, pd.DataFrame) else np.arange(x_vals.shape[0])
        return pd.DataFrame(cif, index=query_times, columns=subject_labels)

    def predict_log_partial_hazard(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the log subdistribution partial hazard x'β for each subject.

        Parameters
        ----------
        X : DataFrame
            Covariate matrix.

        Returns
        -------
        Series
        """
        self._check_is_fitted()
        x_vals = X[self._covariate_names].values.astype(float)
        lp = x_vals @ self.params_.values
        return pd.Series(lp, index=X.index, name="log_partial_hazard")

    def predict_partial_hazard(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the subdistribution partial hazard exp(x'β) for each subject.

        Parameters
        ----------
        X : DataFrame
            Covariate matrix.

        Returns
        -------
        Series
        """
        return np.exp(self.predict_log_partial_hazard(X))

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "params_"):
            raise RuntimeError("Model must be fit before predicting. Call .fit() first.")

    def print_summary(
        self,
        decimals: int = 2,
        style: Optional[str] = None,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Print a formatted summary table of the fitted Fine-Gray model.

        Parameters
        ----------
        decimals : int
            Number of decimal places.
        style : str, optional
            ``"ascii"`` for plain-text, ``"html"`` for HTML output.
        columns : list of str, optional
            Subset of summary columns to display.
        """
        self._check_is_fitted()

        z = stats.norm.ppf(1 - self.alpha / 2)
        p_vals = 2 * stats.norm.sf(np.abs(self.params_.values / self.standard_errors_.values))

        summary = pd.DataFrame(
            {
                "coef": self.params_,
                "exp(coef)": np.exp(self.params_),
                "se(coef)": self.standard_errors_,
                "z": self.params_ / self.standard_errors_,
                "p": p_vals,
                "%.2f lower-bound" % self.alpha: self.confidence_intervals_.iloc[:, 0],
                "%.2f upper-bound" % self.alpha: self.confidence_intervals_.iloc[:, 1],
            }
        )

        headers = {
            "model": "FineGrayFitter",
            "event_of_interest": str(self.event_of_interest),
            "duration_col": "'%s'" % self.duration_col,
            "event_col": "'%s'" % self.event_col,
            "number_of_subjects": self._n_training_rows,
            "number_of_events_of_interest": self._n_events_k,
            "log-likelihood": round(self.log_likelihood_, decimals),
            "partial_AIC": round(self.AIC_partial_, decimals),
            "partial_BIC": round(self.BIC_partial_, decimals),
            "converged": self._converged,
        }

        Printer(
            self,
            summary,
            headers=headers,
            footers={},
            justify="left",
            decimals=decimals,
            columns=columns,
        ).print(style=style)

    def plot_partial_effects_on_outcome(
        self,
        covariates: Union[str, List[str]],
        values: Iterable,
        plot_baseline: bool = True,
        times: Optional[Iterable[float]] = None,
        y: str = "cumulative_incidence",
        **kwargs,
    ):
        """
        Plot the CIF for different covariate values, holding others at their
        median/mode.

        Parameters
        ----------
        covariates : str or list of str
            Covariate(s) to vary.
        values : list
            Values to assign to *covariates* for each curve.
        plot_baseline : bool, optional (default True)
            Also plot the CIF at the median covariate values.
        times : iterable of float, optional
            Times at which to evaluate.  Defaults to fitted event times.
        y : str
            Must be ``"cumulative_incidence"`` (only option).
        **kwargs
            Passed to ``matplotlib.axes.Axes.plot``.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        self._check_is_fitted()

        covariates = [covariates] if isinstance(covariates, str) else list(covariates)
        baseline_row = self._compute_central_values_of_raw_training_data(
            self._training_df.drop(columns=[self.duration_col, self.event_col])
        )

        ax = kwargs.pop("ax", None)
        if ax is None:
            _, ax = plt.subplots()

        if plot_baseline:
            cif_df = self.predict_cumulative_incidence(baseline_row, times=times)
            ax.step(
                cif_df.index,
                cif_df.iloc[:, 0].values,
                where="post",
                label="baseline",
                linestyle="--",
                **kwargs,
            )

        for val in values:
            row = baseline_row.copy()
            row[covariates] = val
            cif_df = self.predict_cumulative_incidence(row, times=times)
            ax.step(
                cif_df.index,
                cif_df.iloc[:, 0].values,
                where="post",
                label="%s=%s" % (covariates, val),
                **kwargs,
            )

        ax.set_xlabel("time")
        ax.set_ylabel("CIF")
        ax.set_title("Partial effects on cumulative incidence (cause %d)" % self.event_of_interest)
        ax.legend()
        return ax

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        if hasattr(self, "_n_training_rows"):
            return (
                "<lifelines.%s: fitted with %d observations, %d events of interest (cause %d)>"
                % (classname, self._n_training_rows, self._n_events_k, self.event_of_interest)
            )
        return "<lifelines.%s>" % classname
