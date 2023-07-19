# -*- coding: utf-8 -*-

import warnings
from typing import Union, Optional, Iterable
from scipy import stats as stats

import pandas as pd
import numpy as np

from lifelines.utils import coalesce, CensoringType, _group_event_table_by_intervals


__all__ = [
    "add_at_risk_counts",
    "plot_lifetimes",
    "plot_interval_censored_lifetimes",
    "qq_plot",
    "cdf_plot",
    "rmst_plot",
    "loglogs_plot",
]


def _iloc(x, i):
    """
    Returns the item at index i or items at indices i from x,
    where x is a numpy array or pd.Series.
    """
    try:
        return x.iloc[i]
    except AttributeError:
        return x[i]


def get_distribution_name_of_lifelines_model(model):
    return model._class_name.replace("Fitter", "").replace("AFT", "").lower()


def create_scipy_stats_model_from_lifelines_model(model):
    from lifelines.fitters import KnownModelParametricUnivariateFitter

    is_univariate_model = isinstance(model, KnownModelParametricUnivariateFitter)
    dist = get_distribution_name_of_lifelines_model(model)

    if not (is_univariate_model):
        raise TypeError(
            "Cannot use qq-plot with this model. See notes here: https://lifelines.readthedocs.io/en/latest/Examples.html?highlight=qq_plot#selecting-a-parametric-model-using-qq-plots"
        )
    if dist == "weibull":
        scipy_dist = "weibull_min"
        sparams = (model.rho_, 0, model.lambda_)
    elif dist == "lognormal":
        scipy_dist = "lognorm"
        sparams = (model.sigma_, 0, np.exp(model.mu_))
    elif dist == "loglogistic":
        scipy_dist = "fisk"
        sparams = (model.beta_, 0, model.alpha_)
    elif dist == "exponential":
        scipy_dist = "expon"
        sparams = (0, model.lambda_)
    else:
        raise NotImplementedError("Distribution not implemented in SciPy")
    return getattr(stats, scipy_dist)(*sparams)


def cdf_plot(model, timeline=None, ax=None, **plot_kwargs):
    """
    This plot compares the empirical CDF (derived by KaplanMeier) vs the model CDF.

    Parameters
    ------------
    model: lifelines univariate model
    timeline: iterable
    ax: matplotlib axis

    """
    from lifelines import KaplanMeierFitter
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    if timeline is None:
        timeline = model.timeline
    COL_EMP = "empirical CDF"

    if CensoringType.is_left_censoring(model):
        empirical_kmf = KaplanMeierFitter().fit_left_censoring(
            model.durations,
            model.event_observed,
            label=COL_EMP,
            timeline=timeline,
            weights=model.weights,
            entry=model.entry,
        )
    elif CensoringType.is_right_censoring(model):
        empirical_kmf = KaplanMeierFitter().fit_right_censoring(
            model.durations,
            model.event_observed,
            label=COL_EMP,
            timeline=timeline,
            weights=model.weights,
            entry=model.entry,
        )
    elif CensoringType.is_interval_censoring(model):
        empirical_kmf = KaplanMeierFitter().fit_interval_censoring(
            model.lower_bound,
            model.upper_bound,
            label=COL_EMP,
            timeline=timeline,
            weights=model.weights,
            entry=model.entry,
        )
    empirical_kmf.plot_cumulative_density(ax=ax, **plot_kwargs)

    dist = get_distribution_name_of_lifelines_model(model)
    dist_object = create_scipy_stats_model_from_lifelines_model(model)
    ax.plot(
        timeline, dist_object.cdf(timeline), label="fitted %s" % dist, **plot_kwargs
    )
    ax.legend()
    return ax


def rmst_plot(model, model2=None, t=np.inf, ax=None, text_position=None, **plot_kwargs):
    """
    This functions plots the survival function of the model plus it's area-under-the-curve (AUC) up
    until the point ``t``. The AUC is known as the restricted mean survival time (RMST).

    To compare the difference between two models' survival curves, you can supply an
    additional model in ``model2``.

    Parameters
    -----------
    model: lifelines.UnivariateFitter
    model2: lifelines.UnivariateFitter, optional
        used to compute the delta RMST of two models
    t: float
        the upper bound of the expectation
    ax: axis
    text_position: tuple
        move the text position of the RMST.


    Examples
    ---------
    .. code:: python

        from lifelines.utils import restricted_mean_survival_time
        from lifelines.datasets import load_waltons
        from lifelines.plotting import rmst_plot

        df = load_waltons()
        ix = df['group'] == 'miR-137'
        T, E = df['T'], df['E']
        time_limit = 50

        kmf_exp = KaplanMeierFitter().fit(T[ix], E[ix], label='exp')
        kmf_con = KaplanMeierFitter().fit(T[~ix], E[~ix], label='control')

        ax = plt.subplot(311)
        rmst_plot(kmf_exp, t=time_limit, ax=ax)

        ax = plt.subplot(312)
        rmst_plot(kmf_con, t=time_limit, ax=ax)

        ax = plt.subplot(313)
        rmst_plot(kmf_exp, model2=kmf_con, t=time_limit, ax=ax)



    """
    from lifelines.utils import restricted_mean_survival_time
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    rmst = restricted_mean_survival_time(model, t=t)
    c = ax._get_lines.get_next_color()
    model.plot_survival_function(ax=ax, color=c, ci_show=False, **plot_kwargs)

    if text_position is None:
        text_position = (np.percentile(model.timeline, 10), 0.15)
    if model2 is not None:
        c2 = ax._get_lines.get_next_color()
        rmst2 = restricted_mean_survival_time(model2, t=t)
        model2.plot_survival_function(ax=ax, color=c2, ci_show=False, **plot_kwargs)
        timeline = np.unique(model.timeline.tolist() + model2.timeline.tolist() + [t])
        predict1 = model.predict(timeline).loc[:t]
        predict2 = model2.predict(timeline).loc[:t]
        # positive
        ax.fill_between(
            timeline[timeline <= t],
            predict1,
            predict2,
            where=predict1 > predict2,
            step="post",
            facecolor="w",
            hatch="|",
            edgecolor="grey",
        )

        # negative
        ax.fill_between(
            timeline[timeline <= t],
            predict1,
            predict2,
            where=predict1 < predict2,
            step="post",
            hatch="-",
            facecolor="w",
            edgecolor="grey",
        )

        ax.text(
            text_position[0],
            text_position[1],
            "RMST(%s) -\n   RMST(%s)=%.3f"
            % (model._label, model2._label, rmst - rmst2),
        )  # dynamically pick this.
    else:
        rmst = restricted_mean_survival_time(model, t=t)
        sf_exp_at_limit = (
            model.predict(np.append(model.timeline, t)).sort_index().loc[:t]
        )
        ax.fill_between(
            sf_exp_at_limit.index,
            sf_exp_at_limit.values,
            step="post",
            color=c,
            alpha=0.25,
        )
        ax.text(
            text_position[0], text_position[1], "RMST=%.3f" % rmst
        )  # dynamically pick this.
    ax.axvline(t, ls="--", color="k")
    ax.set_ylim(0, 1)
    return ax


def qq_plot(model, ax=None, scatter_color="k", **plot_kwargs):
    """
    Produces a quantile-quantile plot of the empirical CDF against
    the fitted parametric CDF. Large deviances away from the line y=x
    can invalidate a model (though we expect some natural deviance in the tails).

    Parameters
    -----------
    model: obj
        A fitted lifelines univariate parametric model, like ``WeibullFitter``
    plot_kwargs:
        kwargs for the plot.

    Returns
    --------
    ax:
        The axes which was used.

    Examples
    ---------
    .. code:: python

        from lifelines import *
        from lifelines.plotting import qq_plot
        from lifelines.datasets import load_rossi
        df = load_rossi()
        wf = WeibullFitter().fit(df['week'], df['arrest'])
        qq_plot(wf)

    Notes
    ------
    The interval censoring case uses the mean between the upper and lower bounds.

    """
    from lifelines.utils import qth_survival_times
    from lifelines import KaplanMeierFitter
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    dist = get_distribution_name_of_lifelines_model(model)
    dist_object = create_scipy_stats_model_from_lifelines_model(model)

    COL_EMP = "empirical quantiles"
    COL_THEO = "fitted %s quantiles" % dist

    if CensoringType.is_left_censoring(model):
        kmf = KaplanMeierFitter().fit_left_censoring(
            model.durations,
            model.event_observed,
            label=COL_EMP,
            weights=model.weights,
            entry=model.entry,
        )
        sf, cdf = kmf.survival_function_[COL_EMP], kmf.cumulative_density_[COL_EMP]
    elif CensoringType.is_right_censoring(model):
        kmf = KaplanMeierFitter().fit_right_censoring(
            model.durations,
            model.event_observed,
            label=COL_EMP,
            weights=model.weights,
            entry=model.entry,
        )
        sf, cdf = kmf.survival_function_[COL_EMP], kmf.cumulative_density_[COL_EMP]
    elif CensoringType.is_interval_censoring(model):
        kmf = KaplanMeierFitter().fit_interval_censoring(
            model.lower_bound,
            model.upper_bound,
            label=COL_EMP,
            weights=model.weights,
            entry=model.entry,
        )
        sf, cdf = (
            kmf.survival_function_.mean(1),
            kmf.cumulative_density_[COL_EMP + "_lower"],
        )
    q = np.unique(cdf.values)

    quantiles = qth_survival_times(1 - q, sf)
    quantiles[COL_THEO] = dist_object.ppf(q)
    quantiles = quantiles.replace([-np.inf, 0, np.inf], np.nan).dropna()

    max_, min_ = quantiles[COL_EMP].max(), quantiles[COL_EMP].min()

    quantiles.plot.scatter(
        COL_THEO, COL_EMP, c="none", edgecolor=scatter_color, lw=0.5, ax=ax
    )
    ax.plot([min_, max_], [min_, max_], c="k", ls=":", lw=1.0)
    ax.set_ylim(min_, max_)
    ax.set_xlim(min_, max_)

    return ax


def is_latex_enabled():
    """
    Returns True if LaTeX is enabled in matplotlib's rcParams,
    False otherwise
    """
    import matplotlib as mpl

    return mpl.rcParams["text.usetex"]


def remove_spines(ax, sides):
    """
    Remove spines of axis.

    Parameters:
      ax: axes to operate on
      sides: list of sides: top, left, bottom, right

    Examples:
    removespines(ax, ['top'])
    removespines(ax, ['top', 'bottom', 'right', 'left'])
    """
    for side in sides:
        ax.spines[side].set_visible(False)
    return ax


def move_spines(ax, sides, dists):
    """
    Move the entire spine relative to the figure.

    Parameters:
      ax: axes to operate on
      sides: list of sides to move. Sides: top, left, bottom, right
      dists: list of float distances to move. Should match sides in length.

    Example:
    move_spines(ax, sides=['left', 'bottom'], dists=[-0.02, 0.1])
    """
    for side, dist in zip(sides, dists):
        ax.spines[side].set_position(("axes", dist))
    return ax


def remove_ticks(ax, x=False, y=False):
    """
    Remove ticks from axis.

    Parameters:
      ax: axes to work on
      x: if True, remove xticks. Default False.
      y: if True, remove yticks. Default False.

    Examples:
    removeticks(ax, x=True)
    removeticks(ax, x=True, y=True)
    """
    if x:
        ax.xaxis.set_ticks_position("none")
    if y:
        ax.yaxis.set_ticks_position("none")
    return ax


def add_at_risk_counts(
    *fitters,
    labels: Optional[Union[Iterable, bool]] = None,
    rows_to_show=None,
    ypos=-0.6,
    xticks=None,
    ax=None,
    at_risk_count_from_start_of_period=False,
    **kwargs
):
    """
    Add counts showing how many individuals were at risk, censored, and observed, at each time point in
    survival/hazard plots.

    Tip: you probably want to call ``plt.tight_layout()`` afterwards.

    Parameters
    ----------
    fitters:
      One or several fitters, for example KaplanMeierFitter, WeibullFitter,
      NelsonAalenFitter, etc...
    labels:
        provide labels for the fitters, default is to use the provided fitter label. Set to
        False for no labels.
    rows_to_show: list
        a sub-list of ['At risk', 'Censored', 'Events']. Default to show all.
    ypos:
        make more positive to move the table up.
    xticks: list
        specify the time periods (as a list) you want to evaluate the counts at.
    at_risk_count_from_start_of_period: bool, default False.
        By default, we use the at-risk count from the end of the period. This is what other packages, and KMunicate suggests, but
        the same issue keeps coming up with users. #1383, #1316 and discussion #1229. This makes the adjustment.
    ax:
        a matplotlib axes

    Returns
    --------
      ax:
        The axes which was used.

    Examples
    --------
    .. code:: python

        # First train some fitters and plot them
        fig = plt.figure()
        ax = plt.subplot(111)

        f1 = KaplanMeierFitter()
        f1.fit(data)
        f1.plot(ax=ax)

        f2 = KaplanMeierFitter()
        f2.fit(data)
        f2.plot(ax=ax)

        # These calls below are equivalent
        add_at_risk_counts(f1, f2)
        add_at_risk_counts(f1, f2, ax=ax, fig=fig)
        plt.tight_layout()

        # This overrides the labels
        add_at_risk_counts(f1, f2, labels=['fitter one', 'fitter two'])
        plt.tight_layout()

        # This hides the labels
        add_at_risk_counts(f1, f2, labels=False)
        plt.tight_layout()

        # Only show at-risk:
        add_at_risk_counts(f1, f2, rows_to_show=['At risk'])
        plt.tight_layout()

    References
    -----------
     Morris TP, Jarvis CI, Cragg W, et al. Proposals on Kaplan–Meier plots in medical research and a survey of stakeholder views: KMunicate. BMJ Open 2019;9:e030215. doi:10.1136/bmjopen-2019-030215

    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = plt.gcf()
    if labels is None:
        labels = [f._label for f in fitters]
    elif labels is False:
        labels = [None] * len(fitters)
    if rows_to_show is None:
        rows_to_show = ["At risk", "Censored", "Events"]
    else:
        assert all(
            row in ["At risk", "Censored", "Events"] for row in rows_to_show
        ), 'must be one of ["At risk", "Censored", "Events"]'
    n_rows = len(rows_to_show)

    # Create another axes where we can put size ticks
    ax2 = plt.twiny(ax=ax)
    # Move the ticks below existing axes
    # Appropriate length scaled for 6 inches. Adjust for figure size.
    ax_height = (
        ax.get_position().y1 - ax.get_position().y0
    ) * fig.get_figheight()  # axis height
    ax2_ypos = ypos / ax_height

    move_spines(ax2, ["bottom"], [ax2_ypos])
    # Hide all fluff
    remove_spines(ax2, ["top", "right", "bottom", "left"])
    # Set ticks and labels on bottom
    ax2.xaxis.tick_bottom()
    # Set limit
    min_time, max_time = ax.get_xlim()
    ax2.set_xlim(min_time, max_time)
    # Set ticks to kwarg or visible ticks
    if xticks is None:
        xticks = [xtick for xtick in ax.get_xticks() if min_time <= xtick <= max_time]
    ax2.set_xticks(xticks)
    # Remove ticks, need to do this AFTER moving the ticks
    remove_ticks(ax2, x=True, y=True)

    ticklabels = []

    for tick in ax2.get_xticks():
        lbl = ""

        # Get counts at tick
        counts = []
        for f in fitters:
            # this is a messy:
            # a) to align with R (and intuition), we do a subtraction off the at_risk column
            # b) we group by the tick intervals
            # c) we want to start at 0, so we give it it's own interval
            if at_risk_count_from_start_of_period:
                event_table_slice = f.event_table.assign(at_risk=lambda x: x.at_risk)
            else:
                event_table_slice = f.event_table.assign(
                    at_risk=lambda x: x.at_risk - x.removed
                )
            if not event_table_slice.loc[:tick].empty:
                event_table_slice = (
                    event_table_slice.loc[:tick, ["at_risk", "censored", "observed"]]
                    .agg(
                        {
                            "at_risk": lambda x: x.tail(1).values,
                            "censored": "sum",
                            "observed": "sum",
                        }
                    )  # see #1385
                    .rename(
                        {
                            "at_risk": "At risk",
                            "censored": "Censored",
                            "observed": "Events",
                        }
                    )
                    .fillna(0)
                )
                counts.extend([int(c) for c in event_table_slice.loc[rows_to_show]])
            else:
                counts.extend([0 for _ in range(n_rows)])
        if n_rows > 1:
            if tick == ax2.get_xticks()[0]:
                max_length = len(str(max(counts)))
                for i, c in enumerate(counts):
                    if i % n_rows == 0:
                        if is_latex_enabled():
                            lbl += (
                                ("\n" if i > 0 else "")
                                + r"\textbf{%s}" % labels[int(i / n_rows)]
                                + "\n"
                            )
                        else:
                            lbl += (
                                ("\n" if i > 0 else "")
                                + r"%s" % labels[int(i / n_rows)]
                                + "\n"
                            )
                    l = rows_to_show[i % n_rows]
                    s = (
                        "{}".format(l.rjust(10, " "))
                        + (" " * (max_length - len(str(c)) + 3))
                        + "{{:>{}d}}\n".format(max_length)
                    )

                    lbl += s.format(c)
            else:
                # Create tick label
                lbl += ""
                for i, c in enumerate(counts):
                    if i % n_rows == 0 and i > 0:
                        lbl += "\n\n"
                    s = "\n{}"
                    lbl += s.format(c)
        else:
            # if only one row to show, show in "condensed" version
            if tick == ax2.get_xticks()[0]:
                max_length = len(str(max(counts)))

                lbl += rows_to_show[0] + "\n"

                for i, c in enumerate(counts):
                    s = (
                        "{}".format(labels[i].rjust(10, " "))
                        + (" " * (max_length - len(str(c)) + 3))
                        + "{{:>{}d}}\n".format(max_length)
                    )
                    lbl += s.format(c)
            else:
                # Create tick label
                lbl += ""
                for i, c in enumerate(counts):
                    s = "\n{}"
                    lbl += s.format(c)
        ticklabels.append(lbl)
    # Align labels to the right so numbers can be compared easily
    ax2.set_xticklabels(ticklabels, ha="right", **kwargs)

    return ax


def plot_interval_censored_lifetimes(
    lower_bound,
    upper_bound,
    entry=None,
    left_truncated=False,
    sort_by_lower_bound=True,
    event_observed_color="#A60628",
    event_right_censored_color="#348ABD",
    ax=None,
    **kwargs
):
    """
    Returns a lifetime plot for interval censored data.

    Parameters
    -----------
    lower_bound: (n,) numpy array or pd.Series
      the start of the period the subject experienced the event in.
    upper_bound: (n,) numpy array or pd.Series
      the end of the period the subject experienced the event in. If the value is equal to the corresponding value in lower_bound, then
      the individual's event was observed (not censored).
    entry: (n,) numpy array or pd.Series
      offsetting the births away from t=0. This could be from left-truncation, or delayed entry into study.
    left_truncated: boolean
      if entry is provided, and the data is left-truncated, this will display additional information in the plot to reflect this.
    sort_by_lower_bound: boolean
      sort by the lower_bound vector
    event_observed_color: str
      default: "#A60628"
    event_right_censored_color: str
      default: "#348ABD"
      applies to any individual with an upper bound of infinity.

    Returns
    -------
    ax:

    Examples
    ---------
    .. code:: python

        import pandas as pd
        import numpy as np
        from lifelines.plotting import plot_interval_censored_lifetimes
        df = pd.DataFrame({'lb':[20,15,30, 10, 20, 30], 'ub':[25, 15, np.infty, 20, 20, np.infty]})
        ax = plot_interval_censored_lifetimes(lower_bound=df['lb'], upper_bound=df['ub'])
    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    # If lower_bounds is pd.Series with non-default index, then use index values as y-axis labels.
    label_plot_bars = (
        type(lower_bound) is pd.Series and type(lower_bound.index) is not pd.RangeIndex
    )

    N = lower_bound.shape[0]
    if N > 25:
        warnings.warn(
            "For less visual clutter, you may want to subsample to less than 25 individuals."
        )
    assert upper_bound.shape[0] == N

    if sort_by_lower_bound:
        ix = np.argsort(lower_bound, 0)
        upper_bound = _iloc(upper_bound, ix)
        lower_bound = _iloc(lower_bound, ix)
        if entry is not None:
            entry = _iloc(entry, ix)
    if entry is None:
        entry = np.zeros(N)
    for i in range(N):
        if np.isposinf(_iloc(upper_bound, i)):
            c = event_right_censored_color
            ax.hlines(i, _iloc(entry, i), _iloc(lower_bound, i), color=c, lw=1.5)
        else:
            c = event_observed_color
            ax.hlines(i, _iloc(entry, i), _iloc(upper_bound, i), color=c, lw=1.5)
            if _iloc(lower_bound, i) == _iloc(upper_bound, i):
                ax.scatter(_iloc(lower_bound, i), i, color=c, marker="o", s=13)
            else:
                ax.scatter(_iloc(lower_bound, i), i, color=c, marker=">", s=13)
                ax.scatter(_iloc(upper_bound, i), i, color=c, marker="<", s=13)
        if left_truncated:
            ax.hlines(i, 0, _iloc(entry, i), color=c, lw=1.0, linestyle="--")
    if label_plot_bars:
        ax.set_yticks(range(0, N))
        ax.set_yticklabels(lower_bound.index)
    else:
        from matplotlib.ticker import MaxNLocator

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0)
    ax.set_ylim(-0.5, N)
    return ax


def plot_lifetimes(
    durations,
    event_observed=None,
    entry=None,
    left_truncated=False,
    sort_by_duration=True,
    event_observed_color="#A60628",
    event_censored_color="#348ABD",
    ax=None,
    **kwargs
):
    """
    Returns a lifetime plot, see examples: https://lifelines.readthedocs.io/en/latest/Survival%20Analysis%20intro.html#Censoring

    Parameters
    -----------
    durations: (n,) numpy array or pd.Series
      duration (relative to subject's birth) the subject was alive for.
    event_observed: (n,) numpy array or pd.Series
      array of booleans: True if event observed, else False.
    entry: (n,) numpy array or pd.Series
      offsetting the births away from t=0. This could be from left-truncation, or delayed entry into study.
    left_truncated: boolean
      if entry is provided, and the data is left-truncated, this will display additional information in the plot to reflect this.
    sort_by_duration: boolean
      sort by the duration vector
    event_observed_color: str
      default: "#A60628"
    event_censored_color: str
      default: "#348ABD"

    Returns
    -------
    ax:

    Examples
    ---------
    .. code:: python

        from lifelines.datasets import load_waltons
        from lifelines.plotting import plot_lifetimes
        T, E = load_waltons()["T"], load_waltons()["E"]
        ax = plot_lifetimes(T.loc[:50], event_observed=E.loc[:50])

    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    # If durations is pd.Series with non-default index, then use index values as y-axis labels.
    label_plot_bars = (
        type(durations) is pd.Series and type(durations.index) is not pd.RangeIndex
    )

    N = durations.shape[0]
    if N > 25:
        warnings.warn(
            "For less visual clutter, you may want to subsample to less than 25 individuals."
        )
    if event_observed is None:
        event_observed = np.ones(N, dtype=bool)
    if entry is None:
        entry = np.zeros(N)
    assert durations.shape[0] == N
    assert event_observed.shape[0] == N

    if sort_by_duration:
        # order by length of lifetimes;
        ix = np.argsort(entry + durations, 0)
        durations = _iloc(durations, ix)
        event_observed = _iloc(event_observed, ix)
        entry = _iloc(entry, ix)
    for i in range(N):
        c = event_observed_color if _iloc(event_observed, i) else event_censored_color
        ax.hlines(i, _iloc(entry, i), _iloc(durations, i), color=c, lw=1.5)
        if left_truncated:
            ax.hlines(i, 0, _iloc(entry, i), color=c, lw=1.0, linestyle="--")
        m = "" if not _iloc(event_observed, i) else "o"
        ax.scatter(_iloc(durations, i), i, color=c, marker=m, s=13)
    if label_plot_bars:
        ax.set_yticks(range(0, N))
        ax.set_yticklabels(durations.index)
    else:
        from matplotlib.ticker import MaxNLocator

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0)
    ax.set_ylim(-0.5, N)
    return ax


def set_kwargs_color(kwargs):
    kwargs["color"] = coalesce(
        kwargs.pop("c", None),
        kwargs.pop("color", None),
        kwargs["ax"]._get_lines.get_next_color(),
    )


def set_kwargs_drawstyle(kwargs, default="steps-post"):
    kwargs["drawstyle"] = kwargs.get("drawstyle", default)


def set_kwargs_label(kwargs, cls):
    kwargs["label"] = kwargs.get("label", cls._label)


def create_dataframe_slicer(iloc, loc, timeline):
    if (loc is not None) and (iloc is not None):
        raise ValueError("Cannot set both loc and iloc in call to .plot().")
    user_did_not_specify_certain_indexes = (iloc is None) and (loc is None)
    user_submitted_slice = (
        slice(timeline.min(), timeline.max())
        if user_did_not_specify_certain_indexes
        else coalesce(loc, iloc)
    )

    get_method = "iloc" if iloc is not None else "loc"
    return lambda df: getattr(df, get_method)[user_submitted_slice]


def loglogs_plot(
    cls, loc=None, iloc=None, show_censors=False, censor_styles=None, ax=None, **kwargs
):
    """
    Specifies a plot of the log(-log(SV)) versus log(time) where SV is the estimated survival function.
    """
    from matplotlib import pyplot as plt

    def loglog(s):
        return np.log(-np.log(s))

    if (loc is not None) and (iloc is not None):
        raise ValueError("Cannot set both loc and iloc in call to .plot().")
    if censor_styles is None:
        censor_styles = {}
    if ax is None:
        ax = plt.gca()
    kwargs["ax"] = ax
    set_kwargs_color(kwargs)
    set_kwargs_drawstyle(kwargs)

    dataframe_slicer = create_dataframe_slicer(iloc, loc, cls.timeline)

    # plot censors
    colour = kwargs["color"]

    if show_censors and cls.event_table["censored"].sum() > 0:
        cs = {"marker": "|", "ms": 12, "mew": 1}
        cs.update(censor_styles)
        times = dataframe_slicer(
            cls.event_table.loc[(cls.event_table["censored"] > 0)]
        ).index.values.astype(float)
        v = cls.predict(times)
        ax.plot(np.log(times), loglog(v), linestyle="None", color=colour, **cs)
    # plot estimate
    sliced_estimates = dataframe_slicer(loglog(cls.survival_function_))
    sliced_estimates["log(timeline)"] = np.log(sliced_estimates.index)
    sliced_estimates.plot(x="log(timeline)", **kwargs)
    ax.set_xlabel("log(timeline)")
    ax.set_ylabel("log(-log(survival_function_))")

    return ax


def _plot_estimate(
    cls,
    estimate=None,  # string like "survival_function_", "cumulative_density_", "hazard_", "cumulative_hazard_"
    loc=None,
    iloc=None,
    show_censors=False,
    censor_styles=None,
    ci_legend=False,
    ci_force_lines=False,
    ci_only_lines=False,
    ci_no_lines=False,
    ci_alpha=0.25,
    ci_show=True,
    at_risk_counts=False,
    logx: bool = False,
    ax=None,
    **kwargs
):
    """
    Plots a pretty figure of estimates

    Matplotlib plot arguments can be passed in inside the kwargs, plus

    Parameters
    -----------
    show_censors: bool
        place markers at censorship events. Default: False
    censor_styles: bool
        If show_censors, this dictionary will be passed into the plot call.
    ci_show: bool
        show confidence intervals. Default: True
    ci_alpha: bool
        the transparency level of the confidence interval. Default: 0.3
    ci_force_lines: bool
        make the confidence intervals to be line plots (versus default shaded areas + lines). Default: False
        Deprecated: use ``ci_only_lines`` instead.
    ci_only_lines: bool
        make the confidence intervals to be line plots (versus default shaded areas + lines). Default: False.
    ci_no_lines: bool
        Only show the shaded area, with no boarding lines. Default: False
    ci_legend: bool
        if ci_force_lines is True, this is a boolean flag to add the lines' labels to the legend. Default: False
    at_risk_counts: bool
        show group sizes at time points. See function ``add_at_risk_counts`` for details. Default: False
    loc: slice
        specify a time-based subsection of the curves to plot, ex:

        >>> model.plot(loc=slice(0.,10.))

        will plot the time values between t=0. and t=10.
    iloc: slice
        specify a location-based subsection of the curves to plot, ex:

        >>> model.plot(iloc=slice(0,10))

        will plot the first 10 time points.
    logx: bool
        Use log scaling on x axis


    Returns
    -------
    ax:
        a pyplot axis object
    """
    from matplotlib import pyplot as plt

    if ci_force_lines:
        warnings.warn(
            "ci_force_lines is deprecated. Use ci_only_lines instead (no functional difference, only a name change).",
            DeprecationWarning,
        )
        ci_only_lines = ci_force_lines
    if "point_in_time" in kwargs:  # marker for given point
        point_in_time = kwargs["point_in_time"]
        kwargs.pop("point_in_time")
    plot_estimate_config = PlotEstimateConfig(
        cls, estimate, loc, iloc, show_censors, censor_styles, logx, ax, **kwargs
    )

    dataframe_slicer = create_dataframe_slicer(iloc, loc, cls.timeline)

    if show_censors and cls.event_table["censored"].sum() > 0:
        cs = {"marker": "+", "ms": 12, "mew": 1}
        cs.update(plot_estimate_config.censor_styles)
        censored_times = dataframe_slicer(
            cls.event_table.loc[(cls.event_table["censored"] > 0)]
        ).index.values.astype(float)
        v = plot_estimate_config.predict_at_times(censored_times).values
        plot_estimate_config.ax.plot(
            censored_times, v, linestyle="None", color=plot_estimate_config.colour, **cs
        )
    dataframe_slicer(plot_estimate_config.estimate_).rename(
        columns=lambda _: plot_estimate_config.kwargs.pop("label")
    ).plot(logx=plot_estimate_config.logx, **plot_estimate_config.kwargs)

    # plot confidence intervals
    if ci_show:
        if ci_only_lines:
            # see https://github.com/CamDavidsonPilon/lifelines/issues/928
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (
                    dataframe_slicer(plot_estimate_config.confidence_interval_)
                    .rename(columns=lambda s: ("" if ci_legend else "_") + s)
                    .plot(
                        linestyle="-",
                        linewidth=1,
                        color=[plot_estimate_config.colour],
                        drawstyle=plot_estimate_config.kwargs["drawstyle"],
                        alpha=0.6,
                        logx=plot_estimate_config.logx,
                        ax=plot_estimate_config.ax,
                    )
                )
        else:
            x = dataframe_slicer(
                plot_estimate_config.confidence_interval_
            ).index.values.astype(float)
            lower = dataframe_slicer(
                plot_estimate_config.confidence_interval_.iloc[:, [0]]
            ).values[:, 0]
            upper = dataframe_slicer(
                plot_estimate_config.confidence_interval_.iloc[:, [1]]
            ).values[:, 0]

            if plot_estimate_config.kwargs["drawstyle"] == "default":
                step = None
            elif plot_estimate_config.kwargs["drawstyle"].startswith("step"):
                step = plot_estimate_config.kwargs["drawstyle"].replace("steps-", "")
            plot_estimate_config.ax.fill_between(
                x,
                lower,
                upper,
                alpha=ci_alpha,
                color=plot_estimate_config.colour,
                linewidth=0.0 if ci_no_lines else 1.0,
                step=step,
            )
    if at_risk_counts:
        add_at_risk_counts(cls, ax=plot_estimate_config.ax)
        plt.tight_layout()
    if "point_in_time" in locals():
        plot_estimate_config.ax.scatter(
            point_in_time, cls.survival_function_at_times(point_in_time)
        )
    return plot_estimate_config.ax


class PlotEstimateConfig:
    def __init__(
        self,
        cls,
        estimate: Union[str, pd.DataFrame],
        loc,
        iloc,
        show_censors,
        censor_styles,
        logx,
        ax,
        **kwargs
    ):
        from matplotlib import pyplot as plt

        self.censor_styles = coalesce(censor_styles, {})

        if ax is None:
            ax = plt.gca()
        kwargs["ax"] = ax
        set_kwargs_color(kwargs)
        set_kwargs_drawstyle(kwargs)
        set_kwargs_label(kwargs, cls)

        self.loc = loc
        self.iloc = iloc
        self.show_censors = show_censors
        # plot censors
        self.ax = ax
        self.colour = kwargs["color"]
        self.logx = logx
        self.kwargs = kwargs

        if isinstance(estimate, str):
            self.estimate_ = getattr(cls, estimate)
            self.confidence_interval_ = getattr(cls, "confidence_interval_" + estimate)
            self.predict_at_times = getattr(cls, estimate + "at_times")
        else:
            self.estimate_ = estimate
            self.confidence_interval_ = kwargs.pop("confidence_intervals")
