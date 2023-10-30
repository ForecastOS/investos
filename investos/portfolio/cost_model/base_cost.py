import copy
import datetime as dt

import numpy as np
import pandas as pd


class BaseCost:
    """Base cost model for InvestOS.
    Other cost models should subclass BaseCost.
    The only requirement of custom cost models is that they (re)implement :py:meth:`~investos.portfolio.cost_model.base_cost.BaseCost.value_expr`.
    """

    def __init__(self, **kwargs):
        self.gamma = 1  # Can change without setting directly as: gamma * BaseCost(). Note that gamma doesn't impact actual costs in backtester / simulated performance, just trades in optimization strategy.
        self.exclude_assets = kwargs.get("exclude_assets", ["cash"])

    def weight_expr(self, t, w_plus, z, value, asset_idx):
        w_plus = self._remove_excluded_columns_np(w_plus, asset_idx)
        z = self._remove_excluded_columns_np(z, asset_idx)

        cost, constraints = self._estimated_cost_for_optimization(t, w_plus, z, value)
        return self.gamma * cost, constraints

    def actual_cost(self, t: dt.datetime, h_plus: pd.Series, u: pd.Series) -> pd.Series:
        h_plus = self._remove_excluded_columns_pd(h_plus)
        u = self._remove_excluded_columns_pd(u)

        return self.get_actual_cost(t, h_plus, u)

    def __mul__(self, other):
        """Read the gamma parameter as a multiplication; so you can change self.gamma without setting it directly as: gamma * BaseCost()"""
        newobj = copy.copy(self)
        newobj.gamma *= other
        return newobj

    def __rmul__(self, other):
        """Read the gamma parameter as a multiplication; so you can change self.gamma without setting it directly as: gamma * BaseCost()"""
        return self.__mul__(other)

    def metadata_dict(self):
        metadata_dict = {}

        metadata_dict["gamma"] = self.gamma

        if getattr(self, "price_movement_sensitivity", None):
            metadata_dict["price_movement_sensitivity"] = self.limit

        return metadata_dict

    def _remove_excluded_columns_pd(self, arg):
        if isinstance(arg, pd.DataFrame):
            return arg.drop(columns=self.exclude_assets, errors="ignore")
        elif isinstance(arg, pd.Series):
            return arg.drop(self.exclude_assets, errors="ignore")
        else:
            return arg

    def _remove_excluded_columns_np(self, np_arr, holdings_cols):
        idx_excl_assets = holdings_cols.get_indexer(self.exclude_assets)
        # Create a boolean array of True values
        mask = np.ones(np_arr.shape, dtype=bool)
        # Set the values at the indices to exclude to False
        mask[idx_excl_assets] = False

        return np_arr[mask]
