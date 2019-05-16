# -*- coding: utf-8 -*-

import warnings
import numpy as np
from lifelines.utils import coalesce, CensoringType
from scipy import stats

__all__ = ["add_at_risk_counts", "plot_lifetimes", "qq_plot", "cdf_plot"]


def get_distribution_name_of_lifelines_model(model):
    return model._class_name.replace("Fitter", "").replace("AFT", "").lower()


def create_scipy_stats_model_from_lifelines_model(model):
    from lifelines.fitters import KnownModelParametericUnivariateFitter, ParametericAFTRegressionFitter

    is_univariate_model = isinstance(model, KnownModelParametericUnivariateFitter)
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


def cdf_plot(model, timeline=None, **plot_kwargs):
    from lifelines import KaplanMeierFitter

    set_kwargs_ax(plot_kwargs)
    ax = plot_kwargs.pop("ax")

    if timeline is None:
        timeline = model.timeline

    COL_EMP = "empirical quantiles"

    if CensoringType.is_left_censoring(model):
        kmf = KaplanMeierFitter().fit_left_censoring(
            model.durations, model.event_observed, label=COL_EMP, timeline=timeline
        )
    elif CensoringType.is_right_censoring(model):
        kmf = KaplanMeierFitter().fit_right_censoring(
            model.durations, model.event_observed, label=COL_EMP, timeline=timeline
        )
    elif CensoringType.is_interval_censoring(model):
        raise NotImplementedError()

    kmf.plot_cumulative_density(ax=ax, **plot_kwargs)

    dist = get_distribution_name_of_lifelines_model(model)
    dist_object = create_scipy_stats_model_from_lifelines_model(model)
    ax.plot(timeline, dist_object.cdf(timeline), label="fitted %s" % dist, **plot_kwargs)
    ax.legend()
    return ax


def qq_plot(model, **plot_kwargs):
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
    ax: axis object

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

    set_kwargs_ax(plot_kwargs)
    ax = plot_kwargs.pop("ax")

    dist = get_distribution_name_of_lifelines_model(model)
    dist_object = create_scipy_stats_model_from_lifelines_model(model)

    COL_EMP = "empirical quantiles"
    COL_THEO = "fitted %s quantiles" % dist

    if CensoringType.is_left_censoring(model):
        kmf = KaplanMeierFitter().fit_left_censoring(model.durations, model.event_observed, label=COL_EMP)
    elif CensoringType.is_right_censoring(model):
        kmf = KaplanMeierFitter().fit_right_censoring(model.durations, model.event_observed, label=COL_EMP)
    elif CensoringType.is_interval_censoring(model):
        raise NotImplementedError()

    q = np.unique(kmf.cumulative_density_.values[:, 0])
    quantiles = qth_survival_times(q, kmf.cumulative_density_, cdf=True)
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
      ax: The axes which was used.

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
    from matplotlib import pyplot as plt

    # Axes and Figure can't be None
    ax = kwargs.get("ax", None)
    if ax is None:
        ax = plt.gca()

    fig = kwargs.get("fig", None)
    if fig is None:
        fig = plt.gcf()

    if "labels" not in kwargs:
        labels = [f._label for f in fitters]
    else:
        # Allow None, in which case no labels should be used
        labels = kwargs["labels"]
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
    # Match tick numbers and locations
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    # Remove ticks, need to do this AFTER moving the ticks
    remove_ticks(ax2, x=True, y=True)
    # Add population size at times
    ticklabels = []
    for tick in ax2.get_xticks():
        lbl = ""
        for f, l in zip(fitters, labels):
            # First tick is prepended with the label
            if tick == ax2.get_xticks()[0] and l is not None:
                if is_latex_enabled():
                    s = "\n{}\\quad".format(l) + "{}"
                else:
                    s = "\n{}   ".format(l) + "{}"
            else:
                s = "\n{}"
            lbl += s.format(f.durations[f.durations >= tick].shape[0])
        ticklabels.append(lbl.strip())
    # Align labels to the right so numbers can be compared easily
    ax2.set_xticklabels(ticklabels, ha="right")

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
    ax

    Examples
    ---------
    >>> from lifelines.datasets import load_waltons
    >>> from lifelines.plotting import plot_lifetimes
    >>> T, E = load_waltons()["T"], load_waltons()["E"]
    >>> ax = plot_lifetimes(T.loc[:50], event_observed=E.loc[:50])

    """
    set_kwargs_ax(kwargs)
    ax = kwargs.pop("ax")

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


def set_kwargs_ax(kwargs):
    from matplotlib import pyplot as plt

    if "ax" not in kwargs:
        kwargs["ax"] = plt.figure().add_subplot(111)


def set_kwargs_color(kwargs):
    kwargs["c"] = coalesce(kwargs.get("c"), kwargs.get("color"), kwargs["ax"]._get_lines.get_next_color())


def set_kwargs_drawstyle(kwargs, default="steps-post"):
    kwargs["drawstyle"] = kwargs.get("drawstyle", default)


def set_kwargs_label(kwargs, cls):
    kwargs["label"] = kwargs.get("label", cls._label)


def create_dataframe_slicer(iloc, loc):
    user_did_not_specify_certain_indexes = (iloc is None) and (loc is None)
    user_submitted_slice = slice(None) if user_did_not_specify_certain_indexes else coalesce(loc, iloc)

    get_method = "loc" if loc is not None else "iloc"
    return lambda df: getattr(df, get_method)[user_submitted_slice]


def plot_loglogs(cls, loc=None, iloc=None, show_censors=False, censor_styles=None, **kwargs):
    """
    Specifies a plot of the log(-log(SV)) versus log(time) where SV is the estimated survival function.
    """

    def loglog(s):
        return np.log(-np.log(s))

    if (loc is not None) and (iloc is not None):
        raise ValueError("Cannot set both loc and iloc in call to .plot().")

    if censor_styles is None:
        censor_styles = {}

    set_kwargs_ax(kwargs)
    set_kwargs_color(kwargs)
    set_kwargs_drawstyle(kwargs)
    kwargs["logx"] = True

    dataframe_slicer = create_dataframe_slicer(iloc, loc)

    # plot censors
    ax = kwargs["ax"]
    colour = kwargs["c"]

    if show_censors and cls.event_table["censored"].sum() > 0:
        cs = {"marker": "+", "ms": 12, "mew": 1}
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
    estimate=None,
    confidence_intervals=None,
    loc=None,
    iloc=None,
    show_censors=False,
    censor_styles=None,
    ci_legend=False,
    ci_force_lines=False,
    ci_alpha=0.25,
    ci_show=True,
    at_risk_counts=False,
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
    ci_alpha: bool
        the transparency level of the confidence interval. Default: 0.3
    ci_force_lines: bool
        force the confidence intervals to be line plots (versus default shaded areas). Default: False
    ci_show: bool
        show confidence intervals. Default: True
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
    plot_estimate_config = PlotEstimateConfig(
        cls, estimate, confidence_intervals, loc, iloc, show_censors, censor_styles, **kwargs
    )

    dataframe_slicer = create_dataframe_slicer(iloc, loc)

    if show_censors and cls.event_table["censored"].sum() > 0:
        cs = {"marker": "+", "ms": 12, "mew": 1}
        cs.update(plot_estimate_config.censor_styles)
        times = dataframe_slicer(cls.event_table.loc[(cls.event_table["censored"] > 0)]).index.values.astype(float)
        v = cls.predict(times)
        plot_estimate_config.ax.plot(times, v, linestyle="None", color=plot_estimate_config.colour, **cs)

    dataframe_slicer(plot_estimate_config.estimate_).rename(
        columns=lambda _: plot_estimate_config.kwargs.pop("label")
    ).plot(**plot_estimate_config.kwargs)

    # plot confidence intervals
    if ci_show:
        if ci_force_lines:
            dataframe_slicer(plot_estimate_config.confidence_interval_).plot(
                linestyle="-",
                linewidth=1,
                color=[plot_estimate_config.colour],
                legend=ci_legend,
                drawstyle=plot_estimate_config.kwargs["drawstyle"],
                ax=plot_estimate_config.ax,
                alpha=0.6,
            )
        else:
            x = dataframe_slicer(plot_estimate_config.confidence_interval_).index.values.astype(float)
            lower = dataframe_slicer(plot_estimate_config.confidence_interval_.filter(like="lower")).values[:, 0]
            upper = dataframe_slicer(plot_estimate_config.confidence_interval_.filter(like="upper")).values[:, 0]

            if plot_estimate_config.kwargs["drawstyle"] == "default":
                step = None
            elif plot_estimate_config.kwargs["drawstyle"].startswith("step"):
                step = plot_estimate_config.kwargs["drawstyle"].replace("steps-", "")

            plot_estimate_config.ax.fill_between(
                x, lower, upper, alpha=ci_alpha, color=plot_estimate_config.colour, linewidth=1.0, step=step
            )

    if at_risk_counts:
        add_at_risk_counts(cls, ax=plot_estimate_config.ax)

    return plot_estimate_config.ax


class PlotEstimateConfig:
    def __init__(self, cls, estimate, confidence_intervals, loc, iloc, show_censors, censor_styles, **kwargs):

        self.censor_styles = coalesce(censor_styles, {})

        set_kwargs_ax(kwargs)
        set_kwargs_color(kwargs)
        set_kwargs_drawstyle(kwargs)
        set_kwargs_label(kwargs, cls)

        self.loc = loc
        self.iloc = iloc
        self.show_censors = show_censors
        # plot censors
        self.ax = kwargs["ax"]
        self.colour = kwargs["c"]
        self.kwargs = kwargs

        if (self.loc is not None) and (self.iloc is not None):
            raise ValueError("Cannot set both loc and iloc in call to .plot().")
        else:
            self.estimate_ = estimate
            self.confidence_interval_ = confidence_intervals
