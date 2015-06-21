# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from lifelines.utils import coalesce


def is_latex_enabled():
    '''
    Returns True if LaTeX is enabled in matplotlib's rcParams,
    False otherwise
    '''
    import matplotlib as mpl

    return mpl.rcParams['text.usetex']


def remove_spines(ax, sides):
    '''
    Remove spines of axis.

    Parameters:
      ax: axes to operate on
      sides: list of sides: top, left, bottom, right

    Examples:
    removespines(ax, ['top'])
    removespines(ax, ['top', 'bottom', 'right', 'left'])
    '''
    for side in sides:
        ax.spines[side].set_visible(False)
    return ax


def move_spines(ax, sides, dists):
    '''
    Move the entire spine relative to the figure.

    Parameters:
      ax: axes to operate on
      sides: list of sides to move. Sides: top, left, bottom, right
      dists: list of float distances to move. Should match sides in length.

    Example:
    move_spines(ax, sides=['left', 'bottom'], dists=[-0.02, 0.1])
    '''
    for side, dist in zip(sides, dists):
        ax.spines[side].set_position(('axes', dist))
    return ax


def remove_ticks(ax, x=False, y=False):
    '''
    Remove ticks from axis.

    Parameters:
      ax: axes to work on
      x: if True, remove xticks. Default False.
      y: if True, remove yticks. Default False.

    Examples:
    removeticks(ax, x=True)
    removeticks(ax, x=True, y=True)
    '''
    if x:
        ax.xaxis.set_ticks_position('none')
    if y:
        ax.yaxis.set_ticks_position('none')
    return ax


def add_at_risk_counts(*fitters, **kwargs):
    '''
    Add counts showing how many individuals were at risk at each time point in
    survival/hazard plots.

    Arguments:
      One or several fitters, for example KaplanMeierFitter,
      NelsonAalenFitter, etc...

    Keyword arguments (all optional):
      ax: The axes to add the labels to. Default is the current axes.
      fig: The figure of the axes. Default is the current figure.
      labels: The labels to use for the fitters. Default is whatever was
              specified in the fitters' fit-function. Giving 'None' will
              hide fitter labels.

    Returns:
      ax: The axes which was used.

    Examples:
        # First train some fitters and plot them
        fig = plt.figure()
        ax = plt.subplot(111)

        f1 = KaplanMeierFitter()
        f1.fit(data)
        f1.plot(ax=ax)

        f2 = KaplanMeierFitter()
        f2.fit(data)
        f2.plot(ax=ax)

        # There are equivalent
        add_at_risk_counts(f1, f2)
        add_at_risk_counts(f1, f2, ax=ax, fig=fig)
        # This overrides the labels
        add_at_risk_counts(f1, f2, labels=['fitter one', 'fitter two'])
        # This hides the labels
        add_at_risk_counts(f1, f2, labels=None)
    '''
    from matplotlib import pyplot as plt

    # Axes and Figure can't be None
    ax = kwargs.get('ax', None)
    if ax is None:
        ax = plt.gca()

    fig = kwargs.get('fig', None)
    if fig is None:
        fig = plt.gcf()

    if 'labels' not in kwargs:
        labels = [f._label for f in fitters]
    else:
        # Allow None, in which case no labels should be used
        labels = kwargs['labels']
        if labels is None:
            labels = [None] * len(fitters)
    # Create another axes where we can put size ticks
    ax2 = plt.twiny(ax=ax)
    # Move the ticks below existing axes
    # Appropriate length scaled for 6 inches. Adjust for figure size.
    ax2_ypos = -0.15 * 6.0 / fig.get_figheight()
    move_spines(ax2, ['bottom'], [ax2_ypos])
    # Hide all fluff
    remove_spines(ax2, ['top', 'right', 'bottom', 'left'])
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
    ax2.set_xticklabels(ticklabels, ha='right')

    # Add a descriptive headline.
    ax2.xaxis.set_label_coords(0, ax2_ypos)
    ax2.set_xlabel('At risk')

    plt.tight_layout()
    return ax


def plot_lifetimes(lifetimes, event_observed=None, birthtimes=None,
                   order=False):
    """
    Parameters:
      lifetimes: an (n,) numpy array of lifetimes.
      event_observed: an (n,) numpy array of booleans: True if event observed, else False.
      birthtimes: an (n,) numpy array offsetting the births away from t=0.


    Creates a lifetime plot, see
    examples:

    """
    from matplotlib import pyplot as plt

    N = lifetimes.shape[0]
    if N > 100:
        print("warning: you may want to subsample to less than 100 individuals.")

    if event_observed is None:
        event_observed = np.ones(N, dtype=bool)

    if birthtimes is None:
        birthtimes = np.zeros(N)

    if order:
        """order by length of lifetimes; probably not very informative."""
        ix = np.argsort(lifetimes, 0)
        lifetimes = lifetimes[ix, 0]
        event_observed = event_observed[ix, 0]
        birthtimes = birthtimes[ix]

    for i in range(N):
        c = "#A60628" if event_observed[i] else "#348ABD"
        plt.hlines(N - 1 - i, birthtimes[i], birthtimes[i] + lifetimes[i], color=c, lw=3)
        m = "|" if not event_observed[i] else 'o'
        plt.scatter((birthtimes[i]) + lifetimes[i], N - 1 - i, color=c, s=30, marker=m)

    plt.ylim(-0.5, N)
    plt.show()
    return


def shaded_plot(x, y, y_upper, y_lower, **kwargs):
    from matplotlib import pyplot as plt

    ax = kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(x, y, drawstyle='steps-post', **kwargs)
    fill_between_steps(x, y_lower, y2=y_upper, ax=ax, alpha=0.25,
                       color=base_line.get_color(), linewidth=1.0)
    return


def plot_regressions(self):
    def plot(ix=None, iloc=None, columns=[], legend=True, **kwargs):
        """"
        A wrapper around plotting. Matplotlib plot arguments can be passed in, plus:

          ix: specify a time-based subsection of the curves to plot, ex:
                   .plot(ix=slice(0.,10.)) will plot the time values between t=0. and t=10.
          iloc: specify a location-based subsection of the curves to plot, ex:
                   .plot(iloc=slice(0,10)) will plot the first 10 time points.
          columns: If not empty, plot a subset of columns from the cumulative_hazards_. Default all.
          legend: show legend in figure.

        """
        from matplotlib import pyplot as plt

        assert (ix is None or iloc is None), 'Cannot set both ix and iloc in call to .plot'

        get_method = "ix" if ix is not None else "iloc"
        if iloc == ix is None:
            user_submitted_ix = slice(0, None)
        else:
            user_submitted_ix = ix if ix is not None else iloc
        get_loc = lambda df: getattr(df, get_method)[user_submitted_ix]

        if len(columns) == 0:
            columns = self.cumulative_hazards_.columns

        if "ax" not in kwargs:
            kwargs["ax"] = plt.figure().add_subplot(111)

        x = get_loc(self.cumulative_hazards_).index.values.astype(float)
        for column in columns:
            y = get_loc(self.cumulative_hazards_[column]).values
            y_upper = get_loc(self.confidence_intervals_[column].ix['upper']).values
            y_lower = get_loc(self.confidence_intervals_[column].ix['lower']).values
            shaded_plot(x, y, y_upper, y_lower, ax=kwargs["ax"], label=coalesce(kwargs.get('label'), column))

        if legend:
            kwargs["ax"].legend()

        return kwargs["ax"]
    return plot


def plot_estimate(cls, estimate):
    doc_string = """"
        Plots a pretty version of the fitted %s.

        Matplotlib plot arguments can be passed in inside the kwargs, plus

        Parameters:
          flat: an opiniated design style with stepped lines and no shading.
                Similar to R's plotting. Default: False
          show_censors: place markers at censorship events. Default: False
          censor_styles: If show_censors, this dictionary will be passed into
                         the plot call.
          ci_alpha: the transparency level of the confidence interval.
                    Default: 0.3
          ci_force_lines: force the confidence intervals to be line plots
                          (versus default shaded areas). Default: False
          ci_show: show confidence intervals. Default: True
          ci_legend: if ci_force_lines is True, this is a boolean flag to add
                     the lines' labels to the legend. Default: False
          at_risk_counts: show group sizes at time points. See function
                          'add_at_risk_counts' for details. Default: False
          ix: specify a time-based subsection of the curves to plot, ex:
                   .plot(ix=slice(0.,10.))
              will plot the time values between t=0. and t=10.
          iloc: specify a location-based subsection of the curves to plot, ex:
                   .plot(iloc=slice(0,10))
                will plot the first 10 time points.
          bandwidth: specify the bandwidth of the kernel smoother for the
                     smoothed-hazard rate. Only used when called 'plot_hazard'.

        Returns:
          ax: a pyplot axis object
        """ % estimate

    def plot(ix=None, iloc=None, flat=False, show_censors=False,
             censor_styles=None, ci_legend=False, ci_force_lines=False,
             ci_alpha=0.25, ci_show=True, at_risk_counts=False,
             bandwidth=None, **kwargs):

        from matplotlib import pyplot as plt

        if censor_styles is None:
            censor_styles = {}

        if (ix is not None and iloc is not None):
            raise ValueError('Cannot set both ix and iloc in call to .plot().')

        if "ax" not in kwargs:
            kwargs["ax"] = plt.figure().add_subplot(111)
        kwargs['color'] = coalesce(kwargs.get('c'), kwargs.get('color'),
                                   next(kwargs["ax"]._get_lines.color_cycle))
        kwargs['drawstyle'] = coalesce(kwargs.get('drawstyle'), 'steps-post')

        # R-style graphics
        if flat:
            ci_force_lines = True
            show_censors = True

        if estimate == "hazard_":
            if bandwidth is None:
                raise ValueError('Must specify a bandwidth parameter in the ' +
                                 'call to plot_hazard.')
            estimate_ = cls.smoothed_hazard_(bandwidth)
            confidence_interval_ = \
                cls.smoothed_hazard_confidence_intervals_(bandwidth,
                                                          hazard_=estimate_.values[:, 0])
        else:
            confidence_interval_ = getattr(cls, 'confidence_interval_')
            estimate_ = getattr(cls, estimate)

        # did user specify certain indexes or locations?
        if iloc == ix is None:
            user_submitted_ix = slice(None)
        else:
            user_submitted_ix = ix if ix is not None else iloc

        get_method = "ix" if ix is not None else "iloc"
        get_loc = lambda df: getattr(df, get_method)[user_submitted_ix]

        # plot censors
        if show_censors and cls.event_table['censored'].sum() > 0:
            cs = {'marker': '+', 'ms': 12, 'mew': 1}
            cs.update(censor_styles)
            times = get_loc(cls.event_table.ix[(cls.event_table['censored'] > 0)]).index.values.astype(float)
            v = cls.predict(times)
            kwargs['ax'].plot(times, v, linestyle='None',
                              color=kwargs['color'], **cs)

        # plot estimate
        get_loc(estimate_).plot(**kwargs)

        # plot confidence intervals
        if ci_show:
            if ci_force_lines:
                get_loc(confidence_interval_).plot(linestyle="-", linewidth=1,
                                                   color=[kwargs['color']], legend=True,
                                                   drawstyle=kwargs.get('drawstyle', 'default'),
                                                   ax=kwargs['ax'], alpha=0.6)
            else:
                x = get_loc(confidence_interval_).index.values.astype(float)
                lower = get_loc(confidence_interval_.filter(like='lower')).values[:, 0]
                upper = get_loc(confidence_interval_.filter(like='upper')).values[:, 0]
                fill_between_steps(x, lower, y2=upper, ax=kwargs['ax'],
                                   alpha=ci_alpha, color=kwargs['color'],
                                   linewidth=1.0)

        if at_risk_counts:
            add_at_risk_counts(cls, ax=kwargs['ax'])

        return kwargs['ax']
    plot.__doc__ = doc_string
    return plot


def fill_between_steps(x, y1, y2=0, h_align='left', ax=None, **kwargs):
    ''' Fills a hole in matplotlib: Fill_between for step plots.
    https://gist.github.com/thriveth/8352565

    Parameters :
    ------------

    x : array-like
        Array/vector of index values. These are assumed to be equally-spaced.
        If not, the result will probably look weird...
    y1 : array-like
        Array/vector of values to be filled under.
    y2 : array-Like
        Array/vector or bottom values for filled area. Default is 0.

    **kwargs will be passed to the matplotlib fill_between() function.

    '''
    from matplotlib import pyplot as plt

    # If no Axes opject given, grab the current one:
    if ax is None:
        ax = plt.gca()
    # First, duplicate the x values
    xx = x.repeat(2)[1:]
    # Now: the average x binwidth
    xstep = (x[1:] - x[:-1]).mean()
    # Now: add one step at end of row.
    xx = np.append(xx, xx.max() + xstep)

    # Make it possible to change step alignment.
    if h_align == 'mid':
        xx -= xstep / 2.
    elif h_align == 'right':
        xx -= xstep

    # Also, duplicate each y coordinate in both arrays
    y1 = y1.repeat(2)
    if isinstance(y2, np.ndarray):
        y2 = y2.repeat(2)

    # now to the plotting part:
    ax.fill_between(xx, y1, y2=y2, **kwargs)

    return ax
