import copy
from functools import wraps

import numpy as np
import pandas as pd


def deep_dict_merge(default_d, update_d):
    "Deep copies update_d onto default_d recursively"

    default_d = copy.deepcopy(default_d)
    update_d = copy.deepcopy(update_d)

    def deep_dict_merge_inner(default_d, update_d):
        for k in update_d.keys():
            if (
                k in default_d
                and isinstance(default_d[k], dict)
                and isinstance(update_d[k], dict)
            ):
                deep_dict_merge_inner(default_d[k], update_d[k])
            else:
                default_d[k] = update_d[k]

    deep_dict_merge_inner(default_d, update_d)
    return default_d  # With update_d values copied onto it


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

    if callable(obj):
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


def clip_for_dates(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        pd_obj = func(self, *args, **kwargs)
        return pd_obj[
            (pd_obj.index >= self.start_date) & (pd_obj.index <= self.end_date)
        ]

    return wrapper


def remove_excluded_columns_pd(arg, exclude_assets=None, include_assets=None):
    if include_assets:
        if isinstance(arg, pd.DataFrame):
            return arg[[col for col in include_assets if col in arg.columns]]
        elif isinstance(arg, pd.Series):
            return arg[[col for col in include_assets if col in arg]]
        else:
            return arg
    else:
        if isinstance(arg, pd.DataFrame):
            return arg.drop(columns=exclude_assets, errors="ignore")
        elif isinstance(arg, pd.Series):
            return arg.drop(exclude_assets, errors="ignore")
        else:
            return arg


def remove_excluded_columns_np(
    np_arr, holdings_cols, exclude_assets=None, include_assets=None
):
    if include_assets:
        idx_incl_assets = holdings_cols.get_indexer(include_assets)
        # Filter out -1 values (i.e. assets with no match)
        idx_incl_assets = idx_incl_assets[idx_incl_assets != -1]
        # Create a boolean array of False values
        mask = np.zeros(np_arr.shape, dtype=bool)
        # Set the values at the indices to exclude to False
        mask[idx_incl_assets] = True
        return np_arr[mask]
    elif exclude_assets:
        idx_excl_assets = holdings_cols.get_indexer(exclude_assets)
        # Filter out -1 values (i.e. assets with no match)
        idx_excl_assets = idx_excl_assets[idx_excl_assets != -1]
        # Create a boolean array of True values
        mask = np.ones(np_arr.shape, dtype=bool)
        # Set the values at the indices to exclude to False
        mask[idx_excl_assets] = False
        return np_arr[mask]
    else:
        return np_arr
