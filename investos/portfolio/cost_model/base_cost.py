import copy
import datetime as dt

import pandas as pd

import investos.util as util


class BaseCost:
    """Base cost model for InvestOS.
    Other cost models should subclass BaseCost.
    The only requirement of custom cost models is that they implement `_estimated_cost_for_optimization` and `get_actual_cost`.
    """

    def __init__(self, **kwargs):
        self.gamma = 1  # Can change without setting directly as: gamma * BaseCost(). Note that gamma doesn't impact actual costs in backtester / simulated performance, just trades in optimization strategy.
        self.exclude_assets = kwargs.get("exclude_assets", ["cash"])
        self.include_assets = kwargs.get("include_assets", [])

    def cvxpy_expression(
        self,
        t,
        weights_portfolio_plus_trades,
        weights_trades,
        portfolio_value,
        asset_idx,
    ):
        weights_portfolio_plus_trades = util.remove_excluded_columns_np(
            weights_portfolio_plus_trades,
            asset_idx,
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )
        weights_trades = util.remove_excluded_columns_np(
            weights_trades,
            asset_idx,
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )

        cost, constraints = self._estimated_cost_for_optimization(
            t, weights_portfolio_plus_trades, weights_trades, portfolio_value
        )
        return self.gamma * cost, constraints

    def actual_cost(
        self,
        t: dt.datetime,
        dollars_holdings_plus_trades: pd.Series,
        dollars_trades: pd.Series,
    ) -> pd.Series:
        dollars_holdings_plus_trades = util.remove_excluded_columns_pd(
            dollars_holdings_plus_trades,
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )
        dollars_trades = util.remove_excluded_columns_pd(
            dollars_trades,
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )

        return self.get_actual_cost(t, dollars_holdings_plus_trades, dollars_trades)

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
