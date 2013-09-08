#plotting

import matplotlib.pyplot as plt
import numpy as np


"""
N = 20
current = 10
birthtimes = current*np.random.uniform(size=(N,))
T, C= generate_random_lifetimes(hz, t, size=N, censor=current - birthtimes )
plot_lifetimes(T, censorship=C, birthtimes=birthtimes)

"""

def plot_lifetimes(lifetimes, censorship = None, birthtimes=None, order=False):
    """

    lifetimes: an (nx1) numpy array of lifetimes. 
    censorship: an (nx1) numpy array of booleans: True if event observed, else False. 
    birthtimes: an (nx1) numpy array offsetting the births away from t=0. 


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
    def plot(c="#348ABD", **kwargs):
      ax = getattr(self, estimate).plot(drawstyle="steps-pre", c=c, **kwargs)
      kwargs["ax"] = ax
      self.confidence_interval_.plot( drawstyle="steps-pre", c=c, linestyle="--", **kwargs)
      return ax
    return plot

