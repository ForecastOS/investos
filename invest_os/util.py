import copy

import numpy as np
import pandas as pd

def deep_dict_merge(default_d, update_d):
    "Deep copies update_d onto default_d recursively"
    
    default_d = copy.deepcopy(default_d)
    update_d = copy.deepcopy(update_d)

    def deep_dict_merge_inner(default_d, update_d):
        for k, v in update_d.items():
            if (k in default_d and isinstance(default_d[k], dict) and isinstance(update_d[k], dict)):
                deep_dict_merge_inner(default_d[k], update_d[k])
            else:
                default_d[k] = update_d[k]

    deep_dict_merge_inner(default_d, update_d)
    return default_d # With update_d values copied onto it

def values_in_time(obj, t, tau=None):
    """
    From CVXPortfolio:

    Obtain value(s) of object at time t, or right before.

    Optionally specify time tau>=t for which we want a prediction,
    otherwise it is assumed tau = t.

    obj: callable, pd.Series, pd.DataFrame, or something else.

        If a callable, we return obj(t,tau).

        If obj has an index attribute,
        we try to return obj.loc[t],
        or obj.loc[t, tau], if the index is a MultiIndex.
        If not available, we return obj.

        Otherwise, we return obj.

    t: np.Timestamp (or similar). Time at which we want
        the value.

    tau: np.Timestamp (or similar), or None. Time tau >= t
        of the prediction,  e.g., tau could be tomorrow, t
        today, and we ask for prediction of market volume tomorrow,
        made today. If None, then it is assumed tau = t.

    """

    if hasattr(obj, '__call__'):
        return obj(t, tau)

    if isinstance(obj, pd.Series) or isinstance(obj, pd.DataFrame):
        try:
            if isinstance(obj.index, pd.MultiIndex):
                return obj.loc[(t, tau)]
            else:
                return obj.loc[t]
        except KeyError:
            return obj

    return obj