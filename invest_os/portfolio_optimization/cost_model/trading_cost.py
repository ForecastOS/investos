import numpy as np

from invest_os.portfolio_optimization.cost_model import BaseCost
from invest_os.util import values_in_time

class TradingCost(BaseCost):


    def __init__(self):
        self.optimizer = None # Set during Optimizer initialization
        
        # For scaling transaction cost; 1 assumes 1 day's volume moves price by 1 day's vol
        # --> Eventually support DF for this
        self.sensitivity_coeff = 1

    def value_expr(self, t, h_plus, u):
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