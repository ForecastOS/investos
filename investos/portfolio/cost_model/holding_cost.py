import pandas as pd
import numpy as np
import datetime as dt
import cvxpy as cvx

from investos.portfolio.cost_model import BaseCost
from investos.util import values_in_time

class HoldingCost(BaseCost):
    """Calculates cost for holding short positions, given customizable short_rate.
    """

    def __init__(self):
        self.optimizer = None # Set during Optimizer initialization

        super().__init__()


    def _estimated_cost_for_optimization(self, t, w_plus, z, value):
        """Estimated holding costs.

        Used by optimization strategy to determine trades. 

        Not used to calculate simulated holding costs for backtest performance.
        """
        w_plus = w_plus[:-1]

        self.expression = self._get_short_rate(t) * cvx.neg(w_plus) # Borrowing rate for period * negative values

        return cvx.sum(self.expression), []


    def actual_cost(self, t: dt.datetime, h_plus: pd.Series, u: pd.Series) -> pd.Series:
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
        short_rate = self._get_short_rate(t)
        self.last_cost = -np.minimum(0, h_plus.iloc[:-1]) * short_rate

        return sum(self.last_cost)


    def _get_short_rate(self, t):
        # Check if t is period 0
        idx_t = self.optimizer.forecast['return'].index.get_loc(t)
        if idx_t == 0:
            idx_t = 1 # If it is, use first period as proxy

        t_minus_1 = self.optimizer.forecast['return'].index[idx_t - 1]
        fraction_of_year = (t - t_minus_1) / dt.timedelta(365,0,0,0)
        
        return (
            (1 + values_in_time(
                self.optimizer.config['borrowing']['short_rate'], t
            )) ** (fraction_of_year)
            - 1
        )