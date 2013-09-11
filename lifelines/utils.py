
import numpy as np
import pandas as pd


def dataframe_from_events_censorship(event_times, censorship):
    """Accepts:
        event_times: (n,1) array of event times 
        censorship: if not None, (n,1) boolean array, 1 if observed event, 0 is censored
    """
    df = pd.DataFrame( event_times, columns=["event_at"] )
    df["removed"] = 1
    df["observed"] = censorship
    event_times = df.groupby("event_at").sum().sort_index()
    return event_times

def basis(n,i):
    x = np.zeros((n,1))
    x[i] = 1
    return x

