import pandas as pd
import numpy as np
import datetime as dt
import cvxpy as cvx

from investos.portfolio.cost_model import BaseCost
from investos.util import values_in_time

class TradingCost(BaseCost):
    """Calculates per period cost for trades `u`, based on spread, standard deviation, volume, and price.
    
    Attributes
    ----------
    sensitivity_coeff : float
        For scaling transaction cost; 1 assumes 1 period's volume moves price by 1 std_dev in vol
    """

    def __init__(self, price_movement_sensitivity=1):
        self.optimizer = None # Set during Optimizer initialization
        # For scaling realized transaction cost; 1 assumes 1 day's volume moves price by 1 day's vol
        # --> Eventually support DF for this
        self.sensitivity_coeff = price_movement_sensitivity

        super().__init__()

    
    def _estimated_cost_for_optimization(self, t, w_plus, z, value):
        """Estimated trading costs.

        Used by optimization strategy to determine trades. 
        """
        z = z[:-1]
        constraints = []

        # Calculate terms for estimated trading cost
        std_dev = values_in_time(self.optimizer.forecast['std_dev'], t)
        volume_dollars = (
            values_in_time(self.optimizer.forecast['volume'], t) *
            values_in_time(self.optimizer.forecast['price'], t)
        )
        percent_volume_traded_pre_trade_weight = np.abs(value) / volume_dollars # Multiplied (using cvx) by trade weight (z) below!

        price_movement_term = (
            self.sensitivity_coeff * 
            std_dev * 
            percent_volume_traded_pre_trade_weight
        )
        # END calculate terms for estimated trading cost

        # Create estimated cost expression
        try: # Spread (estimated) costs
            self.estimate_expression = cvx.multiply(
                values_in_time(self.optimizer.forecast['half_spread'], t), cvx.abs(z))
        except TypeError:
            self.estimate_expression = cvx.multiply(
                values_in_time(self.optimizer.forecast['half_spread'], t).values, cvx.abs(z))
        
        try: # Price movement due to volume (estimated) costs
            self.estimate_expression += cvx.multiply(price_movement_term, cvx.abs(z) ** 2)
        except TypeError:
            self.estimate_expression += cvx.multiply(price_movement_term.values, cvx.abs(z) ** 2)
        # END create estimated cost expression

        return cvx.sum(self.estimate_expression), constraints


    def actual_cost(self, t: dt.datetime, h_plus: pd.Series, u: pd.Series) -> pd.Series:
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

        trading_costs = (
            spread_cost + (
                self.sensitivity_coeff * 
                std_dev * 
                np.abs(u_no_cash) *
                percent_volume_traded
            )
        )

        self.optimizer.backtest.save_data('costs_trading', t, trading_costs)

        return trading_costs.sum()