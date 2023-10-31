import datetime as dt

import cvxpy as cvx
import numpy as np
import pandas as pd

from investos.portfolio.cost_model import BaseCost
from investos.util import remove_excluded_columns_pd, values_in_time


class ShortHoldingCost(BaseCost):
    """Calculates cost for holding short positions, given customizable short_rate."""

    def __init__(self, short_rates, **kwargs):
        super().__init__(**kwargs)
        self.short_rates = remove_excluded_columns_pd(
            short_rates,
            exclude_assets=self.exclude_assets,
            include_assets=self.include_assets,
        )

    def _estimated_cost_for_optimization(self, t, w_plus, z, value):
        """Estimated holding costs.

        Used by optimization strategy to determine trades.

        Not used to calculate simulated holding costs for backtest performance.
        """
        expression = cvx.multiply(self._get_short_rate(t), cvx.neg(w_plus))

        return cvx.sum(expression), []

    def get_actual_cost(
        self, t: dt.datetime, h_plus: pd.Series, u: pd.Series
    ) -> pd.Series:
        """Method that calculates per-period (short position) holding costs given period `t` holdings and trades.

        Parameters
        ----------
        t : datetime.datetime
            The datetime for associated trades `u` and holdings plus trades `h_plus`.
        h_plus : pandas.Series
            Holdings at beginning of period t, plus trades for period `t` (`u`). Same as `u` + `h` for `t`.
        u : pandas.Series
            Trades (as values) for period `t`.
        """
        return sum(-np.minimum(0, h_plus) * self._get_short_rate(t))

    def _get_short_rate(self, t):
        return values_in_time(self.short_rates, t)
