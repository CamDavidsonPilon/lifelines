#plotting 
import numpy as np
from matplotlib import pyplot as plt

def plot_lifetimes(lifetimes, censorship = None, birthtimes=None, order=False):
    """

    lifetimes: an (n,) numpy array of lifetimes. 
    censorship: an (n,) numpy array of booleans: True if event observed, else False. 
    birthtimes: an (n,) numpy array offsetting the births away from t=0. 


    Creates a lifetime plot, see 
    examples:

    """
    N = lifetimes.shape[0]
    if N>100:
      print "warning: you may want to subsample to less than 100 individuals."

    if censorship is None:
      censorship = np.ones(N, dtype=bool)

    if birthtimes is None:
      birthtimes = np.zeros(N)

    if order:
       """order by length of lifetimes; probably not very informative."""
       ix = np.argsort(lifetimes,0)
       lifetimes = lifetimes[ix,0]
       censorship = censorship[ix,0]
       birthtimes = birthtimes[ix]

    for i in range(N):
           c = "#A60628" if censorship[i] else "#348ABD"
           plt.hlines( N-1-i, birthtimes[i] , birthtimes[i] + lifetimes[i], color = c, lw=3)
           m = "|" if not censorship[i] else 'o'
           plt.scatter( (birthtimes[i]) + lifetimes[i] ,  N-1-i, color = c, s=30, marker = m)

    plt.ylim(-0.5, N)
    plt.show()


def shaded_plot(x, y, y_upper, y_lower, **kwargs):
    ax = kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(x, y, **kwargs)
    ax.fill_between(x, y_lower, y_upper, facecolor=base_line.get_color(), alpha=0.25)
    return

def plot_regressions(self):
    def plot(ix=None, columns = [], ci_legend=False, ci_force_lines=False, **kwargs):

        if ix == None:
          ix = slice(0,None)

        if len(columns)==0:
          columns = self.cumulative_hazards_.columns

        if "ax" in kwargs:
            ax = kwargs["ax"]
        else:   
            ax = plt.subplot(111)

        x = self.cumulative_hazards_.ix[ix].index.values.astype(float)
        for column in columns:
          y = self.cumulative_hazards_[column].ix[ix].values
          y_upper = self.confidence_intervals_[column].ix['upper'].ix[ix].values
          y_lower = self.confidence_intervals_[column].ix['lower'].ix[ix].values
          shaded_plot(x, y, y_upper, y_lower, ax=ax, label=column)

        return ax
    return plot


def plot_dataframes(self, estimate):
    def plot(c="#348ABD", ix=None, flat=False, ci_legend=False, ci_force_lines=False, bandwidth=None, **kwargs):
      """"
      A wrapper around plotting lines. Matplotlib plot arguments can be passed in, plus:

        flat: a design style with stepped lines and no shading. Similar to R.
        ci_force_lines: force the confidence intervals to be line plots (versus default shaded areas).
        ci_legend: if ci_force_lines, boolean flag to add the line's label to the legend.
        ix: specify a subsection of the curves to plot, ex:
               .plot(ix=slice(0,-3)) will plot all but the last three points in the estimate and confidence intervals.
               .plot(ix=slice(0.,10.)) will plot the time values between t=0. and t=10. (This is because they are floats, not ints.)
        bandwidth: specify the bandwidth of the kernel smoother for the smoothed-hazard rate. Only used 
         when called 'plot_hazard'

      """
      if flat:
          ci_force_lines=True
          kwargs["drawstyle"] = "steps-pre"

      if estimate=="hazard_":
          assert bandwidth != None, 'Must specify a bandwidth parameter in the call to plot_hazard'
          estimate_ = self.smoothed_hazard_(bandwidth)
          confidence_interval_ = self.smoothed_hazard_confidence_intervals_(bandwidth, hazard_=estimate_.values[:,0])
      else:
          confidence_interval_ = getattr(self, 'confidence_interval_')
          estimate_ = getattr(self, estimate)
       
      if ix == None:
        ix = slice(0,None)

      n = estimate_.shape[0]
      ax = estimate_.ix[ix].plot(c=c, marker='o', markeredgewidth=0, markersize=10./n, **kwargs)
      kwargs["ax"]=ax
      if ci_force_lines:
        kwargs["legend"]=ci_legend
        confidence_interval_.ix[ix].plot(c=c, linestyle="--", linewidth=1, **kwargs)
      else:
        x = self.confidence_interval_.index[ix].values.astype(float)
        lower = confidence_interval_.filter(like='lower').values[ix,0]
        upper = confidence_interval_.filter(like='upper').values[ix,0]
        plt.fill_between(x, lower, y2=upper, color=c, alpha=0.25)
      return ax
    return plot

