import pandas as pd
import numpy as np
import datetime as dt

from investos.portfolio_optimization.cost_model import BaseCost
from investos.util import values_in_time

class TradingCost(BaseCost):
    """Calculates per period cost for trades `u`, based on spread, standard deviation, volume, and price.
    
    Attributes
    ----------
    sensitivity_coeff : float
        For scaling transaction cost; 1 assumes 1 day's volume moves price by 1 std_dev in vol
    """

    def __init__(self):
        self.optimizer = None # Set during Optimizer initialization
        
        # For scaling transaction cost; 1 assumes 1 day's volume moves price by 1 day's vol
        # --> Eventually support DF for this
        self.sensitivity_coeff = 1

    def value_expr(self, t: dt.datetime, h_plus: pd.Series, u: pd.Series) -> pd.Series:
        """Method that calculates `t` period cost for trades `u`.

        Trading cost (per asset) is assumed to be half of bid-ask spread, plus (per asset) standard deviation * % of period volume traded. 
        Assumption is trading 100% of period volume moves price by 1 (default value for `self.sensitivity_coeff`) standard deviation. 

        Parameters
        ----------
        t : datetime.datetime
            The datetime for associated trades `u` and holdings plus trades `h_plus`.
        h_plus : pandas.Series
            Holdings at beginning of period t, plus trades for period `t` (`u`). Same as `u` + `h` for `t`.
        u : pandas.Series
            Trades (as values) for period `t`.
        """

        u_no_cash = u.iloc[:-1]
        
        spread_cost = np.abs(u_no_cash) * values_in_time(self.optimizer.actual['half_spread'], t)
        std_dev = values_in_time(self.optimizer.actual['std_dev'], t)
        volume_dollars = (
            values_in_time(self.optimizer.actual['volume'], t) *
            values_in_time(self.optimizer.actual['price'], t)
        )
        percent_volume_traded = np.abs(u_no_cash) / volume_dollars

        self.tmp_trading_costs = (
            spread_cost + (
                self.sensitivity_coeff * 
                std_dev * 
                np.abs(u_no_cash) *
                percent_volume_traded
            )
        )

        return self.tmp_trading_costs.sum()