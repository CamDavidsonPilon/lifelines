# -*- coding: utf-8 -*-

import numpy as np
from lifelines.utils import concordance_index

__all__ = ["concordance_index", "uncensored_l2_log_loss", "uncensored_l1_log_loss"]


def uncensored_l1_log_loss(event_times, predicted_event_times, event_observed=None):
    r"""
    Calculates the l1 log-loss of predicted event times to true event times for *non-censored*
    individuals only.

    .. math::  1/N \sum_{i} |log(t_i) - log(q_i)|

    Parameters
    ----------
      event_times:
        a (n,) array of observed survival times.
      predicted_event_times:
        a (n,) array of predicted survival times.
      event_observed:
        a (n,) array of censorship flags, 1 if observed,  0 if not. Default None assumes all observed.

    Returns
    -------
      l1-log-loss: a scalar
    """
    if event_observed is None:
        event_observed = np.ones_like(event_times, dtype=bool)

    ix = event_observed.astype(bool)
    return np.abs(np.log(event_times[ix]) - np.log(predicted_event_times[ix])).mean()


def uncensored_l2_log_loss(event_times, predicted_event_times, event_observed=None):
    r"""
    Calculates the l2 log-loss of predicted event times to true event times for *non-censored*
    individuals only.

    .. math::  1/N \sum_{i} (log(t_i) - log(q_i))**2

    Parameters
    ----------
      event_times:
        a (n,) array of observed survival times.
      predicted_event_times:
        a (n,) array of predicted survival times.
      event_observed:
        a (n,) array of censorship flags, 1 if observed,  0 if not. Default None assumes all observed.

    Returns
    -------
      l2-log-loss: a scalar
    """
    if event_observed is None:
        event_observed = np.ones_like(event_times, dtype=bool)

    ix = event_observed.astype(bool)
    return np.power(np.log(event_times[ix]) - np.log(predicted_event_times[ix]), 2).mean()
