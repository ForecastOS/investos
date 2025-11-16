import datetime as dt

import cvxpy as cvx
import numpy as np
import pandas as pd

from investos.portfolio.cost_model import BaseCost
from investos.util import get_value_at_t, remove_excluded_columns_pd


class ShortHoldingCost(BaseCost):
    """Calculates cost for holding short positions, given customizable short_rate."""

    def __init__(self, short_rates, **kwargs):
        super().__init__(**kwargs)
        self.short_rates = remove_excluded_columns_pd(
            short_rates,
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )

    def _estimated_cost_for_optimization(
        self, t, weights_portfolio_plus_trades, weights_trades, portfolio_value
    ):
        """Estimated holding costs.

        Used by optimization strategy to determine trades.

        Not used to calculate simulated holding costs for backtest performance.
        """
        expression = cvx.multiply(
            self._get_short_rate(t), cvx.neg(weights_portfolio_plus_trades)
        )

        return cvx.sum(expression), []

    def get_actual_cost(
        self,
        t: dt.datetime,
        dollars_holdings_plus_trades: pd.Series,
        dollars_trades: pd.Series,
    ) -> pd.Series:
        """Method that calculates per-period (short position) holding costs given period `t` holdings and trades."""
        return sum(
            -np.minimum(0, dollars_holdings_plus_trades) * self._get_short_rate(t)
        )

    def _get_short_rate(self, t):
        return get_value_at_t(self.short_rates, t)
