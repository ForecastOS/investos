import copy
import datetime as dt

import pandas as pd

import investos.util as util


class BaseCost:
    """Base cost model for InvestOS.
    Other cost models should subclass BaseCost.
    The only requirement of custom cost models is that they (re)implement :py:meth:`~investos.portfolio.cost_model.base_cost.BaseCost.value_expr`.
    """

    def __init__(self, **kwargs):
        self.gamma = 1  # Can change without setting directly as: gamma * BaseCost(). Note that gamma doesn't impact actual costs in backtester / simulated performance, just trades in optimization strategy.
        self.exclude_assets = kwargs.get("exclude_assets", ["cash"])
        self.include_assets = kwargs.get("include_assets", [])

    def weight_expr(self, t, w_plus, z, value, asset_idx):
        w_plus = util.remove_excluded_columns_np(
            w_plus,
            asset_idx,
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )
        z = util.remove_excluded_columns_np(
            z,
            asset_idx,
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )

        cost, constraints = self._estimated_cost_for_optimization(t, w_plus, z, value)
        return self.gamma * cost, constraints

    def actual_cost(self, t: dt.datetime, h_plus: pd.Series, u: pd.Series) -> pd.Series:
        h_plus = util.remove_excluded_columns_pd(
            h_plus,
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )
        u = util.remove_excluded_columns_pd(
            u, exclude_assets=self.exclude_assets, include_assets=self.include_assets
        )

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
