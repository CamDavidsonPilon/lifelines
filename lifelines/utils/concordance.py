# -*- coding: utf-8 -*-

import numpy as np
from lifelines.utils.btree import _BTree


def somers_d(event_times, x, event_observed=None) -> float:
    """
    A measure of rank association between [-1, 1] between a censored variable, event_times,
    and another (uncensored) variable, x. -1 is strong anti-correlation, 1 is strong correlation.


    event_times: iterable
         a length-n iterable of observed survival times.
    x: iterable
        a length-n iterable to compare against
    event_observed: iterable, optional
        a length-n iterable censoring flags, 1 if observed, 0 if not. Default None assumes all observed.


    Examples
    --------
    .. code:: python
        from lifelines.datasets import load_rossi
        from lifelines.utils

        T, E = df['week'], df['arrest']
        x = df['age']
        somers_d(T, x, E)


    """
    return 2 * concordance_index(event_times, x, event_observed) - 1


def concordance_index(event_times, predicted_scores, event_observed=None) -> float:
    """
    Calculates the concordance index (C-index) between a series
    of event times and a predicted score. The first is the real survival times from
    the observational data, and the other is the predicted score from a model of some kind.

    The c-index is the average of how often a model says X is greater than Y when, in the observed
    data, X is indeed greater than Y. The c-index also handles how to handle censored values
    (obviously, if Y is censored, it's hard to know if X is truly greater than Y).


    The concordance index is a value between 0 and 1 where:

    - 0.5 is the expected result from random predictions,
    - 1.0 is perfect concordance and,
    - 0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)

    The calculation internally done is

    >>> (pairs_correct + 0.5 * pairs_tied) / admissable_pairs

    where ``pairs_correct`` is the number of pairs s.t. if ``t_x > t_y``, then ``s_x > s_y``, pairs,
    ``pairs_tied`` is the number of pairs where ``s_x = s_y``, and ``admissable_pairs`` is all possible pairs. The subtleties
    are in how censored observation are handled (ex: not all pairs can be evaluated due to censoring).


    Parameters
    ----------
    event_times: iterable
         a length-n iterable of observed survival times.
    predicted_scores: iterable
        a length-n iterable of predicted scores - these could be survival times, or hazards, etc. See https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
    event_observed: iterable, optional
        a length-n iterable censoring flags, 1 if observed, 0 if not. Default None assumes all observed.

    Returns
    -------
    c-index: float
      a value between 0 and 1.

    References
    -----------
    Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in
    developing models, evaluating assumptions and adequacy, and measuring and
    reducing errors. Statistics in Medicine 1996;15(4):361-87.

    Examples
    --------
    .. code:: python

        from lifelines.utils import concordance_index
        cph = CoxPHFitter().fit(df, 'T', 'E')
        concordance_index(df['T'], -cph.predict_partial_hazard(df), df['E'])

    """
    event_times, predicted_scores, event_observed = _preprocess_scoring_data(event_times, predicted_scores, event_observed)
    num_correct, num_tied, num_pairs = _concordance_summary_statistics(event_times, predicted_scores, event_observed)

    return _concordance_ratio(num_correct, num_tied, num_pairs)


def _concordance_ratio(num_correct: int, num_tied: int, num_pairs: int) -> float:
    if num_pairs == 0:
        raise ZeroDivisionError("No admissable pairs in the dataset.")
    return (num_correct + num_tied / 2) / num_pairs


def _concordance_summary_statistics(event_times, predicted_event_times, event_observed):  # pylint: disable=too-many-locals
    """Find the concordance index in n * log(n) time.

    Assumes the data has been verified by lifelines.utils.concordance_index first.
    """
    # Here's how this works.
    #
    # It would be pretty easy to do if we had no censored data and no ties. There, the basic idea
    # would be to iterate over the cases in order of their true event time (from least to greatest),
    # while keeping track of a pool of *predicted* event times for all cases previously seen (= all
    # cases that we know should be ranked lower than the case we're looking at currently).
    #
    # If the pool has O(log n) insert and O(log n) RANK (i.e., "how many things in the pool have
    # value less than x"), then the following algorithm is n log n:
    #
    # Sort the times and predictions by time, increasing
    # n_pairs, n_correct := 0
    # pool := {}
    # for each prediction p:
    #     n_pairs += len(pool)
    #     n_correct += rank(pool, p)
    #     add p to pool
    #
    # There are three complications: tied ground truth values, tied predictions, and censored
    # observations.
    #
    # - To handle tied true event times, we modify the inner loop to work in *batches* of observations
    # p_1, ..., p_n whose true event times are tied, and then add them all to the pool
    # simultaneously at the end.
    #
    # - To handle tied predictions, which should each count for 0.5, we switch to
    #     n_correct += min_rank(pool, p)
    #     n_tied += count(pool, p)
    #
    # - To handle censored observations, we handle each batch of tied, censored observations just
    # after the batch of observations that died at the same time (since those censored observations
    # are comparable all the observations that died at the same time or previously). However, we do
    # NOT add them to the pool at the end, because they are NOT comparable with any observations
    # that leave the study afterward--whether or not those observations get censored.
    if np.logical_not(event_observed).all():
        return (0, 0, 0)

    died_mask = event_observed.astype(bool)
    # TODO: is event_times already sorted? That would be nice...
    died_truth = event_times[died_mask]
    ix = np.argsort(died_truth)
    died_truth = died_truth[ix]
    died_pred = predicted_event_times[died_mask][ix]

    censored_truth = event_times[~died_mask]
    ix = np.argsort(censored_truth)
    censored_truth = censored_truth[ix]
    censored_pred = predicted_event_times[~died_mask][ix]

    censored_ix = 0
    died_ix = 0
    times_to_compare = _BTree(np.unique(died_pred))
    num_pairs = np.int64(0)
    num_correct = np.int64(0)
    num_tied = np.int64(0)

    # we iterate through cases sorted by exit time:
    # - First, all cases that died at time t0. We add these to the sortedlist of died times.
    # - Then, all cases that were censored at time t0. We DON'T add these since they are NOT
    #   comparable to subsequent elements.
    while True:
        has_more_censored = censored_ix < len(censored_truth)
        has_more_died = died_ix < len(died_truth)
        # Should we look at some censored indices next, or died indices?
        if has_more_censored and (not has_more_died or died_truth[died_ix] > censored_truth[censored_ix]):
            pairs, correct, tied, next_ix = _handle_pairs(censored_truth, censored_pred, censored_ix, times_to_compare)
            censored_ix = next_ix
        elif has_more_died and (not has_more_censored or died_truth[died_ix] <= censored_truth[censored_ix]):
            pairs, correct, tied, next_ix = _handle_pairs(died_truth, died_pred, died_ix, times_to_compare)
            for pred in died_pred[died_ix:next_ix]:
                times_to_compare.insert(pred)
            died_ix = next_ix
        else:
            assert not (has_more_died or has_more_censored)
            break

        num_pairs += pairs
        num_correct += correct
        num_tied += tied

    return (num_correct, num_tied, num_pairs)


def _handle_pairs(truth, pred, first_ix, times_to_compare):
    """
    Handle all pairs that exited at the same time as truth[first_ix].

    Returns
    -------
      (pairs, correct, tied, next_ix)
      new_pairs: The number of new comparisons performed
      new_correct: The number of comparisons correctly predicted
      next_ix: The next index that needs to be handled
    """
    next_ix = first_ix
    while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
        next_ix += 1
    pairs = len(times_to_compare) * (next_ix - first_ix)
    correct = np.int64(0)
    tied = np.int64(0)
    for i in range(first_ix, next_ix):
        rank, count = times_to_compare.rank(pred[i])
        correct += rank
        tied += count

    return (pairs, correct, tied, next_ix)


def _naive_concordance_summary_statistics(event_times, predicted_event_times, event_observed):
    """
    Fallback, simpler method to compute concordance.

    """

    def _valid_comparison(time_a, time_b, event_a, event_b):
        """True if times can be compared."""
        if time_a == time_b:
            # Ties are only informative if exactly one event happened
            return event_a != event_b
        if event_a and event_b:
            return True
        if event_a and time_a < time_b:
            return True
        if event_b and time_b < time_a:
            return True
        return False

    def _concordance_value(time_a, time_b, pred_a, pred_b, event_a, event_b):
        if pred_a == pred_b:
            # Same as random
            return (0, 1)
        if pred_a < pred_b:
            return (time_a < time_b) or (time_a == time_b and event_a and not event_b), 0
        # pred_a > pred_b
        return (time_a > time_b) or (time_a == time_b and not event_a and event_b), 0

    num_pairs = 0.0
    num_correct = 0.0
    num_tied = 0.0

    for a, time_a in enumerate(event_times):
        pred_a = predicted_event_times[a]
        event_a = event_observed[a]
        # Don't want to double count
        for b in range(a + 1, len(event_times)):
            time_b = event_times[b]
            pred_b = predicted_event_times[b]
            event_b = event_observed[b]

            if _valid_comparison(time_a, time_b, event_a, event_b):
                num_pairs += 1.0
                crct, ties = _concordance_value(time_a, time_b, pred_a, pred_b, event_a, event_b)
                num_correct += crct
                num_tied += ties

    return (num_correct, num_tied, num_pairs)


def naive_concordance_index(event_times, predicted_event_times, event_observed=None) -> float:
    event_times, predicted_event_times, event_observed = _preprocess_scoring_data(
        event_times, predicted_event_times, event_observed
    )
    return _concordance_ratio(*_naive_concordance_summary_statistics(event_times, predicted_event_times, event_observed))


def _preprocess_scoring_data(event_times, predicted_scores, event_observed):
    event_times = np.asarray(event_times, dtype=float)
    predicted_scores = np.asarray(predicted_scores, dtype=float)

    # Allow for (n, 1) or (1, n) arrays
    if event_times.ndim == 2 and (event_times.shape[0] == 1 or event_times.shape[1] == 1):
        # Flatten array
        event_times = event_times.ravel()
    # Allow for (n, 1) or (1, n) arrays
    if predicted_scores.ndim == 2 and (predicted_scores.shape[0] == 1 or predicted_scores.shape[1] == 1):
        # Flatten array
        predicted_scores = predicted_scores.ravel()

    if event_times.shape != predicted_scores.shape:
        raise ValueError("Event times and predictions must have the same shape")
    if event_times.ndim != 1:
        raise ValueError("Event times can only be 1-dimensional: (n,)")

    if event_observed is None:
        event_observed = np.ones(event_times.shape[0], dtype=float)
    else:
        event_observed = np.asarray(event_observed, dtype=float).ravel()
        if event_observed.shape != event_times.shape:
            raise ValueError("Observed events must be 1-dimensional of same length as event times")

    # check for NaNs
    for a in [event_times, predicted_scores, event_observed]:
        if np.isnan(a).any():
            raise ValueError("NaNs detected in inputs, please correct or drop.")

    return event_times, predicted_scores, event_observed
