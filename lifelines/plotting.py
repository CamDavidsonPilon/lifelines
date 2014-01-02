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


def plot_dataframes(self, estimate):
    def plot(c="#348ABD", ix=None, hazard=False, ci_legend=False, ci_force_lines=False, bandwidth=None, **kwargs):
      """"
      A wrapper around plotting lines. Matplotlib plot arguments can be passed in, plus:

        ci_force_lines: force the confidence intervals to be line plots (versus default areas).
        ci_legend: if ci_force_lines, boolean flag to add the line's label to the legend.
        ix: specify a subsection of the curves to plot, i.e. .plot(ix=slice(0,-3)) will plot all 
         but the last three points in the estimate and confidence intervals.
        bandwidth: specify the bandwidth of the kernel smoother for the smoothed-hazard rate. Only used 
         when called 'plot_hazard'

      """
      if estimate=="hazard_":
          assert bandwidth != None, 'Must specify a bandwidth parameter in the call to plot_hazard'
          estimate_ = self.smoothed_hazard_(bandwidth)
          confidence_interval_ = self.hazard_confidence_intervals_(bandwidth,hazard_=estimate_.values[:,0])
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

