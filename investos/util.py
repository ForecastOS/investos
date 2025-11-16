import copy
from functools import wraps

import cvxpy as cvx
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


def get_value_at_t(source, current_time, prediction_time=None, use_lookback=False):
    """
    Obtain the value(s) of a source object at a given time.

    Parameters
    ----------
    source : callable, pd.Series, pd.DataFrame, or other object
        - If callable, returns source(current_time, prediction_time).
        - If a pandas object, returns the value at the index matching
          current_time (or (current_time, prediction_time) for MultiIndex).
        - If no matching index is found, returns the object itself unless
          use_lookback=True, in which case it returns the most recent prior index.

    current_time : np.Timestamp
        Time at which the value is desired.

    prediction_time : np.Timestamp or None
        Optional forecast time. If None, defaults to current_time.

    use_lookback : bool
        If True, and no exact index match exists, return the value at the
        closest index strictly before current_time.

    Returns
    -------
    The retrieved value or the original source object.
    """
    if prediction_time is None:
        prediction_time = current_time

    # Case 1: Callable source
    if callable(source):
        return source(current_time, prediction_time)

    # Case 2: Pandas Series/DataFrame
    if isinstance(source, (pd.Series, pd.DataFrame)):
        try:
            if isinstance(source.index, pd.MultiIndex):
                return source.loc[(current_time, prediction_time)]
            else:
                return source.loc[current_time]
        except KeyError:
            if not use_lookback:
                return source

            # Lookback mode: find the closest earlier timestamp
            time_index = source.index.get_level_values(0)
            earlier_times = time_index[time_index < current_time]

            if not earlier_times.empty:
                closest_time = earlier_times.max()
                return source.loc[closest_time]
            else:
                return source

    # Case 3: Fallback
    return source


def clip_for_dates(func):
    """
    Decorator that restricts the returned pandas object to the date range
    defined by the instance's `start_date` and `end_date`.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        pd_obj = func(self, *args, **kwargs)
        return pd_obj[
            (pd_obj.index >= self.start_date) & (pd_obj.index <= self.end_date)
        ]

    return wrapper


def remove_excluded_columns_pd(arg, exclude_assets=None, include_assets=None):
    """Filter a DataFrame or Series by keeping `include_assets` if provided, otherwise dropping `exclude_assets`."""
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
    """Filter a NumPy array by including or excluding columns based on asset names."""
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


def get_max_key_lt_or_eq_value(dictionary, value):
    """
    Returns the maximum key in the dictionary that is less than or equal to the given value.
    If no such key exists, returns None.

    Useful for looking up values by datetime.
    """
    # Filter keys that are less than or equal to the value
    valid_keys = [k for k in dictionary.keys() if k <= value]

    # Return the max of the valid keys if the list is not empty
    if valid_keys:
        return max(valid_keys)
    else:
        return None


def _solve_and_extract_trade_weights(
    prob, weights_trades, t, solver, solver_opts, holdings
):
    try:
        prob.solve(solver=solver, **solver_opts)
        return (
            t,
            weights_trades.value,
        )  # Return the value of weights_trades after solving
    except (cvx.SolverError, cvx.DCPError, TypeError):
        return t, pd.Series(index=holdings.index, data=0.0).values  # Zero trade
