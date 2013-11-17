from datetime import datetime

import numpy as np
import pandas as pd
from pandas import to_datetime

import pdb

def group_event_series( groups, durations, censorship, limit=-1):
    """
    Joins multiple event series together into dataframes. A generalization of 
    `dataframe_from_events_censorship` to groups.

    Parameters:
        groups: a (n,) array of individuals' group ids.
        durations: a (n,) array of durations of each individual
        censorship: a (n,) array of censorship, 1 if observed, 0 else. 
    """

    unique_groups = np.unique(groups)

    #set first group
    g = unique_groups[0]
    ix = groups == g
    T = durations[ix]
    C = censorship[ix]

    g_name = str(g)
    data = dataframe_from_events_censorship(T,C,columns=['removed:'+g_name, "observed:"+g_name, 'censored:'+g_name])
    for g in unique_groups[1:]:
        ix = groups == g
        T = durations[ix]
        C = censorship[ix]
        g_name = str(g)
        data = data.join(dataframe_from_events_censorship(T,C,columns=['removed:'+g_name, "observed:"+g_name, 'censored:'+g_name]), how='outer' )
    data = data.fillna(0)
    #hmmm pandas...its too bad I can't do data.ix[:limit] and leave out the if. 
    if int(limit) != -1:
        data = data.ix[:limit]
    return unique_groups, data.filter(like='removed:'), data.filter(like='observed:'), data.filter(like='censored:')


def dataframe_from_events_censorship(event_times, censorship, columns=["removed", "observed", "censored"], weights=None):
    """
    Parameters:
        event_times: (n,1) array of event times 
        censorship: if not None, (n,1) boolean array, 1 if observed event, 0 is censored
        columns: a 3-length array to call the, in order, removed individuals, observed deaths
          and censorships.

    Returns:
        Pandas DataFrame with index as the unique times in event_times. The columns named 
        'removed' refers to the number of individuals who were removed from the population 
        by the end of the period. The column 'observed' refers to the number of removed 
        individuals who were observed to have died (i.e. not censored.) The column
        'censored' is defined as 'removed' - 'observed' (the number of individuals who
         left the population due to censorship)

    """
    df = pd.DataFrame( event_times.astype(float), columns=["event_at"] )
    df[columns[0]] = 1 if weights == None else weights
    df[columns[1]] = censorship
    event_times = df.groupby("event_at").sum().sort_index()
    event_times[columns[2]] = event_times[columns[0]] - event_times[columns[1]]
    return event_times

def datetimes_to_durations( start_times, end_times, fill_date = None, freq='D', dayfirst=False, na_values=None ):
    """
    This is a very flexible function for transforming arrays of start_times and end_times 
    to the proper format for lifelines: duration and censorship arrays.

    Parameters:
        start_times: an array, series or dataframe of start times. These can be strings, or datetimes.
        end_times: an array, series or dataframe of end times. These can be strings, or datetimes.
                   These values can be None, or an empty string, which corresponds to censorship.
        fill_date: the date to use if end_times is a None or empty string. This corresponds to last date 
                  of observation.
        freq: the units of time to use. Default 'D' for days. See pandas freq 
        day_first: convert assuming European-style dates, i.e. day/month/year. 
        na_values : Additional string to recognize as NA/NaN. 

    Returns:
        T: a array of floats representing the durations with time units given by freq.
        C: a boolean array of censorship: 1 if death observed, 0 else. 

    """
    if not fill_date:
        fill_date = datetime.today()

    freq_string = 'timedelta64[%s]'%freq
    start_times = pd.Series(start_times).copy()
    end_times = pd.Series(end_times).copy()

    start_times_ = to_datetime(start_times, dayfirst=dayfirst)

    C = ~( pd.isnull(end_times).values + (end_times=='') + (end_times==na_values) )
    end_times[~C] = fill_date 
    end_times_ = to_datetime(end_times, dayfirst=dayfirst, coerce=True)

    T = (end_times_ - start_times_).map(lambda x: x.astype(freq_string).astype(float) )

    return T.values, C.values

def inv_normal_cdf(p):
    if p < 0.5:
      return -AandS_approximation(p)
    else:
      return AandS_approximation(1-p)

def AandS_approximation(p):
    #Formula 26.2.23 from A&S and help from John Cook ;)
    # http://www.johndcook.com/normal_cdf_inverse.html
    c_0 = 2.515517
    c_1 = 0.802853
    c_2 = 0.010328

    d_1 = 1.432788
    d_2 = 0.189269
    d_3 = 0.001308

    t = np.sqrt(-2*np.log(p))

    return t - (c_0+c_1*t+c_2*t**2)/(1+d_1*t+d_2*t*t+d_3*t**3)


def median_loss(T, T_pred):
    return np.abs(T-T_pred).mean()

def cross_validation(fitter, T, X, censorship=None, k=5, loss_function="median"):
    """
    This implements the censor-sensative loss function as given in 
    'Prediction Performance of Survival Models', Yan Yuan, 2008
    """

    pass

def yaun_loss(fitter, T_true, T_pred, X_train, T_train, censorship=None):


    if censorship==None:
        t = T_pred.shape[0]
        sC = np.ones((t,1))
        censorship = np.ones((t,1))
    else:
        #We are estimating S_c(*|z),is the probability that the 
        # jth individual survived to time yj without being censored
        fitter.fit(T_train,X_train, censorship=1-censorship, timeline=T_true) 
        sC = fitter.predict_survival_function(X_train).ix[T_true[:,0]].values.diagonal()[:,None] #dirty way to get this

    return (censorship*(T_pred-T_true)**2/sC).mean()

def quadrature(fx,x):
  t = x.shape[0]
  return (0.5*fx[:,0] + fx[:,1:-1].sum(1) + 0.5*fx[:,-1])*(x[-1] - x[0])*1.0/t

def basis(n,i):
    x = np.zeros((n,1))
    x[i] = 1
    return x


def kernel_smoother(timeline, hazards, sigma):
    pass
