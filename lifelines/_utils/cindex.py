from __future__ import division

import numpy as np
import blist

def fast_concordance_index(event_times, predicted_event_times, event_observed):
    """n * log(n) concordance index algorithm prototype."""
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
    times_to_compare = blist.sortedlist()
    num_pairs = 0
    num_correct = 0
    num_tied = 0

    def handle_pairs(truth, pred, first_ix):
        """Handle all pairs that exited at the same time as truth[first_ix].

        Returns:
          (pairs, correct, tied, next_ix)
          new_pairs: The number of new comparisons performed
          new_correct: The number of comparisons correctly predicted
          next_ix: The next index that needs to be handled
        """
        next_ix = first_ix
        while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
            next_ix += 1
        pairs = len(times_to_compare) * (next_ix - first_ix)
        correct = 0
        tied = 0
        for i in xrange(first_ix, next_ix):
            correct += times_to_compare.bisect_left(pred[i])
            tied += times_to_compare.count(pred[i])

        return (pairs, correct, tied, next_ix)

    # we iterate through cases sorted by exit time:
    # - First, all cases that died at time t0. We add these to the sortedlist of died times.
    # - Then, all cases that were censored at time t0. We DON'T add these since they are NOT
    #   comparable to subsequent elements.
    while True:
        has_more_censored = censored_ix < len(censored_truth)
        has_more_died = died_ix < len(died_truth)
        # Should we look at some censored indices next, or died indices?
        if has_more_censored and (not has_more_died
                                  or died_truth[died_ix] > censored_truth[censored_ix]):
            pairs, correct, tied, next_ix = handle_pairs(censored_truth, censored_pred, censored_ix)
            censored_ix = next_ix
        elif has_more_died and (not has_more_censored
                                or died_truth[died_ix] <= censored_truth[censored_ix]):
            pairs, correct, tied, next_ix = handle_pairs(died_truth, died_pred, died_ix)
            times_to_compare.update(died_pred[died_ix:next_ix])
            died_ix = next_ix
        else:
            assert not (has_more_died or has_more_censored)
            break

        num_pairs += pairs
        num_correct += correct
        num_tied += tied

    return (num_correct + num_tied / 2) / num_pairs


def concordance_index(event_times, predicted_event_times, event_observed):
    """
    Fallback method if the Fortran code hasn't been compiled. Assumes the data
    has been verified by lifelines.utils.concordance_index first.
    """
    def valid_comparison(time_a, time_b, event_a, event_b):
        """True if times can be compared."""
        if time_a == time_b:
            # Ties are not informative
            return False
        elif event_a and event_b:
            return True
        elif event_a and time_a < time_b:
            return True
        elif event_b and time_b < time_a:
            return True
        else:
            return False

    def concordance_value(time_a, time_b, pred_a, pred_b):
        if pred_a == pred_b:
            # Same as random
            return 0.5
        elif time_a < time_b and pred_a < pred_b:
            return 1.0
        elif time_b < time_a and pred_b < pred_a:
            return 1.0
        else:
            return 0.0

    paircount = 0.0
    csum = 0.0

    for a in range(0, len(event_times)):
        time_a = event_times[a]
        pred_a = predicted_event_times[a]
        event_a = event_observed[a]
        # Don't want to double count
        for b in range(a + 1, len(event_times)):
            time_b = event_times[b]
            pred_b = predicted_event_times[b]
            event_b = event_observed[b]

            if valid_comparison(time_a, time_b, event_a, event_b):
                paircount += 1.0
                csum += concordance_value(time_a, time_b, pred_a, pred_b)

    return csum / paircount
