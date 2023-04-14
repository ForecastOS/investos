import numpy as np
import datetime as dt

from invest_os.portfolio_optimization.cost_model import BaseCost
from invest_os.util import values_in_time

class HoldingCost(BaseCost):


    def __init__(self):
        self.optimizer = None # Set during Optimizer initialization


    def value_expr(self, t, h_plus, u):
        # TBU - make sure first value time period isn't 0
        idx_t = self.optimizer.forecast['return'].index.get_loc(t)
        if idx_t == 0:
            idx_t = 1 # Use first period as proxy

        t_minus_1 = self.optimizer.forecast['return'].index[idx_t - 1]
        fraction_of_year = (t - t_minus_1) / dt.timedelta(365,0,0,0)
        short_rate = (
            (1 + values_in_time(
                self.optimizer.config['borrowing']['short_rate'], t
            )) ** (fraction_of_year)
            - 1
        )

        # print(idx_t, t, t_minus_1, fraction_of_year, short_rate)

        self.last_cost = -np.minimum(0, h_plus.iloc[:-1]) * short_rate

        # self.last_cost -= h_plus.iloc[:-1] * values_in_time(self.dividends, t)

        return sum(self.last_cost)