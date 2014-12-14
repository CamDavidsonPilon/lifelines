def concordance_index(event_times, predicted_event_times, event_observed):
    """
    Fallback method if the Fortran code hasn't been compiled. Assumes the data
    has been verified by lifelines.utils.concordance_index first.
    """
    def valid_comparison(time_a, time_b, event_a, event_b):
        """True if times can be compared."""
        if event_a and event_b:
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

    for a, (time_a, pred_a, event_a) in enumerate(zip(event_times,
                                                      predicted_event_times,
                                                      event_observed)):
        # Don't want to double count
        for b in range(a + 1, len(event_times)):
            time_b = event_times[b]
            pred_b = predicted_event_times[b]
            event_b = event_observed[b]

            if valid_comparison(time_a, time_b, event_a, event_b):
                paircount += 1.0
                csum += concordance_value(time_a, time_b, pred_a, pred_b)

    return csum / paircount
