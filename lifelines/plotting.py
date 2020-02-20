# -*- coding: utf-8 -*-

import warnings
from typing import Union

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd

from lifelines.utils import coalesce, CensoringType


__all__ = ["add_at_risk_counts", "plot_lifetimes", "qq_plot", "cdf_plot", "rmst_plot", "loglogs_plot"]


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


    """
    from lifelines import KaplanMeierFitter

    if ax is None:
        ax = plt.gca()

    if timeline is None:
        timeline = model.timeline

    COL_EMP = "empirical CDF"

    if CensoringType.is_left_censoring(model):
        empirical_kmf = KaplanMeierFitter().fit_left_censoring(
            model.durations, model.event_observed, label=COL_EMP, timeline=timeline
        )
    elif CensoringType.is_right_censoring(model):
        empirical_kmf = KaplanMeierFitter().fit_right_censoring(
            model.durations, model.event_observed, label=COL_EMP, timeline=timeline
        )
    elif CensoringType.is_interval_censoring(model):
        raise NotImplementedError("lifelines does not have a non-parametric interval model yet.")

    empirical_kmf.plot_cumulative_density(ax=ax, **plot_kwargs)

    dist = get_distribution_name_of_lifelines_model(model)
    dist_object = create_scipy_stats_model_from_lifelines_model(model)
    ax.plot(timeline, dist_object.cdf(timeline), label="fitted %s" % dist, **plot_kwargs)
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

    >>> from lifelines.utils import restricted_mean_survival_time
    >>> from lifelines.datasets import load_waltons
    >>> from lifelines.plotting import rmst_plot
    >>>
    >>> df = load_waltons()
    >>> ix = df['group'] == 'miR-137'
    >>> T, E = df['T'], df['E']
    >>> time_limit = 50
    >>>
    >>> kmf_exp = KaplanMeierFitter().fit(T[ix], E[ix], label='exp')
    >>> kmf_con = KaplanMeierFitter().fit(T[~ix], E[~ix], label='control')
    >>>
    >>> ax = plt.subplot(311)
    >>> rmst_plot(kmf_exp, t=time_limit, ax=ax)
    >>>
    >>> ax = plt.subplot(312)
    >>> rmst_plot(kmf_con, t=time_limit, ax=ax)
    >>>
    >>> ax = plt.subplot(313)
    >>> rmst_plot(kmf_exp, model2=kmf_con, t=time_limit, ax=ax)



    """
    from lifelines.utils import restricted_mean_survival_time

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
            "RMST(%s) -\n   RMST(%s)=%.3f" % (model._label, model2._label, rmst - rmst2),
        )  # dynamically pick this.
    else:
        rmst = restricted_mean_survival_time(model, t=t)
        sf_exp_at_limit = model.predict(np.append(model.timeline, t)).sort_index().loc[:t]
        ax.fill_between(sf_exp_at_limit.index, sf_exp_at_limit.values, step="post", color=c, alpha=0.25)
        ax.text(text_position[0], text_position[1], "RMST=%.3f" % rmst)  # dynamically pick this.

    ax.axvline(t, ls="--", color="k")
    ax.set_ylim(0, 1)
    return ax


def qq_plot(model, ax=None, **plot_kwargs):
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

    >>> from lifelines import *
    >>> from lifelines.plotting import qq_plot
    >>> from lifelines.datasets import load_rossi
    >>> df = load_rossi()
    >>> wf = WeibullFitter().fit(df['week'], df['arrest'])
    >>> qq_plot(wf)


    """
    from lifelines.utils import qth_survival_times
    from lifelines import KaplanMeierFitter

    if ax is None:
        ax = plt.gca()

    dist = get_distribution_name_of_lifelines_model(model)
    dist_object = create_scipy_stats_model_from_lifelines_model(model)

    COL_EMP = "empirical quantiles"
    COL_THEO = "fitted %s quantiles" % dist

    if CensoringType.is_left_censoring(model):
        kmf = KaplanMeierFitter().fit_left_censoring(model.durations, model.event_observed, label=COL_EMP)
    elif CensoringType.is_right_censoring(model):
        kmf = KaplanMeierFitter().fit_right_censoring(model.durations, model.event_observed, label=COL_EMP)
    elif CensoringType.is_interval_censoring(model):
        raise NotImplementedError("lifelines does not have a non-parametric interval model yet.")

    q = np.unique(kmf.cumulative_density_.values[:, 0])
    # this is equivalent to the old code `qth_survival_times(q, kmf.cumulative_density, cdf=True)`
    quantiles = qth_survival_times(1 - q, kmf.survival_function_)
    quantiles[COL_THEO] = dist_object.ppf(q)
    quantiles = quantiles.replace([-np.inf, 0, np.inf], np.nan).dropna()

    max_, min_ = quantiles[COL_EMP].max(), quantiles[COL_EMP].min()

    quantiles.plot.scatter(COL_THEO, COL_EMP, c="none", edgecolor="k", lw=0.5, ax=ax)
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


def add_at_risk_counts(*fitters, **kwargs):
    """
    Add counts showing how many individuals were at risk at each time point in
    survival/hazard plots.

    Parameters
    ----------
    fitters:
      One or several fitters, for example KaplanMeierFitter,
      NelsonAalenFitter, etc...


    Returns
    --------
      ax:
        The axes which was used.

    Examples
    --------
    >>> # First train some fitters and plot them
    >>> fig = plt.figure()
    >>> ax = plt.subplot(111)
    >>>
    >>> f1 = KaplanMeierFitter()
    >>> f1.fit(data)
    >>> f1.plot(ax=ax)
    >>>
    >>> f2 = KaplanMeierFitter()
    >>> f2.fit(data)
    >>> f2.plot(ax=ax)
    >>>
    >>> # There are equivalent
    >>> add_at_risk_counts(f1, f2)
    >>> add_at_risk_counts(f1, f2, ax=ax, fig=fig)
    >>>
    >>> # This overrides the labels
    >>> add_at_risk_counts(f1, f2, labels=['fitter one', 'fitter two'])
    >>>
    >>> # This hides the labels
    >>> add_at_risk_counts(f1, f2, labels=None)
    """
    # Axes and Figure can't be None
    ax = kwargs.pop("ax", None)
    if ax is None:
        ax = plt.gca()

    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = plt.gcf()

    if "labels" not in kwargs:
        labels = [f._label for f in fitters]
    else:
        # Allow None, in which case no labels should be used
        labels = kwargs.pop("labels", None)
        if labels is None:
            labels = [None] * len(fitters)
    # Create another axes where we can put size ticks
    ax2 = plt.twiny(ax=ax)
    # Move the ticks below existing axes
    # Appropriate length scaled for 6 inches. Adjust for figure size.
    ax2_ypos = -0.15 * 6.0 / fig.get_figheight()
    move_spines(ax2, ["bottom"], [ax2_ypos])
    # Hide all fluff
    remove_spines(ax2, ["top", "right", "bottom", "left"])
    # Set ticks and labels on bottom
    ax2.xaxis.tick_bottom()
    # Set limit
    min_time, max_time = ax.get_xlim()
    ax2.set_xlim(min_time, max_time)
    # Set ticks to kwarg or visible ticks
    xticks = kwargs.pop("xticks", None)
    if xticks is None:
        xticks = [xtick for xtick in ax.get_xticks() if min_time <= xtick <= max_time]
    ax2.set_xticks(xticks)
    # Remove ticks, need to do this AFTER moving the ticks
    remove_ticks(ax2, x=True, y=True)
    # Add population size at times
    ticklabels = []
    for tick in ax2.get_xticks():
        lbl = ""
        # Get counts at tick
        counts = [f.durations[f.durations >= tick].shape[0] for f in fitters]
        # Create tick label
        for l, c in zip(labels, counts):
            # First tick is prepended with the label
            if tick == ax2.get_xticks()[0] and l is not None:
                # Get length of largest count
                max_length = len(str(max(counts)))
                if is_latex_enabled():
                    s = "\n{}\\quad".format(l) + "{{:>{}d}}".format(max_length)
                else:
                    s = "\n{}   ".format(l) + "{{:>{}d}}".format(max_length)
            else:
                s = "\n{}"
            lbl += s.format(c)
        ticklabels.append(lbl.strip())
    # Align labels to the right so numbers can be compared easily
    ax2.set_xticklabels(ticklabels, ha="right", **kwargs)

    # Add a descriptive headline.
    ax2.xaxis.set_label_coords(0, ax2_ypos)
    ax2.set_xlabel("At risk")

    plt.tight_layout()
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
      duration subject was observed for.
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
    >>> from lifelines.datasets import load_waltons
    >>> from lifelines.plotting import plot_lifetimes
    >>> T, E = load_waltons()["T"], load_waltons()["E"]
    >>> ax = plot_lifetimes(T.loc[:50], event_observed=E.loc[:50])

    """
    if ax is None:
        ax = plt.gca()

    N = durations.shape[0]
    if N > 80:
        warnings.warn("For less visual clutter, you may want to subsample to less than 80 individuals.")

    if event_observed is None:
        event_observed = np.ones(N, dtype=bool)

    if entry is None:
        entry = np.zeros(N)

    assert durations.shape[0] == N
    assert event_observed.shape[0] == N

    if sort_by_duration:
        # order by length of lifetimes;
        ix = np.argsort(entry + durations, 0)
        durations = durations[ix]
        event_observed = event_observed[ix]
        entry = entry[ix]

    for i in range(N):
        c = event_observed_color if event_observed[i] else event_censored_color
        ax.hlines(i, entry[i], entry[i] + durations[i], color=c, lw=1.5)
        if left_truncated:
            ax.hlines(i, 0, entry[i], color=c, lw=1.0, linestyle="--")
        m = "" if not event_observed[i] else "o"
        ax.scatter(entry[i] + durations[i], i, color=c, marker=m, s=10)

    ax.set_ylim(-0.5, N)
    return ax


def set_kwargs_color(kwargs):
    kwargs["color"] = coalesce(kwargs.get("c"), kwargs.get("color"), kwargs["ax"]._get_lines.get_next_color())


def set_kwargs_drawstyle(kwargs, default="steps-post"):
    kwargs["drawstyle"] = kwargs.get("drawstyle", default)


def set_kwargs_label(kwargs, cls):
    kwargs["label"] = kwargs.get("label", cls._label)


def create_dataframe_slicer(iloc, loc, timeline):
    if (loc is not None) and (iloc is not None):
        raise ValueError("Cannot set both loc and iloc in call to .plot().")

    user_did_not_specify_certain_indexes = (iloc is None) and (loc is None)
    user_submitted_slice = (
        slice(timeline.min(), timeline.max()) if user_did_not_specify_certain_indexes else coalesce(loc, iloc)
    )

    get_method = "iloc" if iloc is not None else "loc"
    return lambda df: getattr(df, get_method)[user_submitted_slice]


def loglogs_plot(cls, loc=None, iloc=None, show_censors=False, censor_styles=None, ax=None, **kwargs):
    """
    Specifies a plot of the log(-log(SV)) versus log(time) where SV is the estimated survival function.
    """

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
    kwargs["logx"] = True

    dataframe_slicer = create_dataframe_slicer(iloc, loc, cls.timeline)

    # plot censors
    colour = kwargs["color"]

    if show_censors and cls.event_table["censored"].sum() > 0:
        cs = {"marker": "|", "ms": 12, "mew": 1}
        cs.update(censor_styles)
        times = dataframe_slicer(cls.event_table.loc[(cls.event_table["censored"] > 0)]).index.values.astype(float)
        v = cls.predict(times)
        # don't log times, as Pandas will take care of all log-scaling later.
        ax.plot(times, loglog(v), linestyle="None", color=colour, **cs)

    # plot estimate
    dataframe_slicer(loglog(cls.survival_function_)).plot(**kwargs)
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
    ax=None,
    **kwargs
):

    """
    Plots a pretty figure of {0}.{1}

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

    Returns
    -------
    ax:
        a pyplot axis object
    """
    if ci_force_lines:
        warnings.warn(
            "ci_force_lines is deprecated. Use ci_only_lines instead (no functional difference, only a name change).",
            DeprecationWarning,
        )
        ci_only_lines = ci_force_lines

    plot_estimate_config = PlotEstimateConfig(cls, estimate, loc, iloc, show_censors, censor_styles, ax, **kwargs)

    dataframe_slicer = create_dataframe_slicer(iloc, loc, cls.timeline)

    if show_censors and cls.event_table["censored"].sum() > 0:
        cs = {"marker": "+", "ms": 12, "mew": 1}
        cs.update(plot_estimate_config.censor_styles)
        censored_times = dataframe_slicer(cls.event_table.loc[(cls.event_table["censored"] > 0)]).index.values.astype(
            float
        )
        v = plot_estimate_config.predict_at_times(censored_times).values
        plot_estimate_config.ax.plot(censored_times, v, linestyle="None", color=plot_estimate_config.colour, **cs)

    dataframe_slicer(plot_estimate_config.estimate_).rename(
        columns=lambda _: plot_estimate_config.kwargs.pop("label")
    ).plot(**plot_estimate_config.kwargs)

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
                        ax=plot_estimate_config.ax,
                        alpha=0.6,
                    )
                )
        else:
            x = dataframe_slicer(plot_estimate_config.confidence_interval_).index.values.astype(float)
            lower = dataframe_slicer(plot_estimate_config.confidence_interval_.iloc[:, [0]]).values[:, 0]
            upper = dataframe_slicer(plot_estimate_config.confidence_interval_.iloc[:, [1]]).values[:, 0]

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

    return plot_estimate_config.ax


class PlotEstimateConfig:
    def __init__(self, cls, estimate: Union[str, pd.DataFrame], loc, iloc, show_censors, censor_styles, ax, **kwargs):

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
        self.kwargs = kwargs

        if isinstance(estimate, str):
            self.estimate_ = getattr(cls, estimate)
            self.confidence_interval_ = getattr(cls, "confidence_interval_" + estimate)
            self.predict_at_times = getattr(cls, estimate + "at_times")
        else:
            self.estimate_ = estimate
            self.confidence_interval_ = kwargs.pop("confidence_intervals")
