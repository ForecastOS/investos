import pandas as pd
import datetime as dt

from invest_os.portfolio_optimization.strategy import BaseStrategy
from invest_os.portfolio_optimization.cost_model import TradingCost, HoldingCost
from invest_os.util import values_in_time

class RankLongShort(BaseStrategy):
    def __init__(self, n_periods_held=1, leverage=1, percent_short=0.25, percent_long=0.25, costs=[]):
        self.forecast_returns = None # Set by Optimizer in init
        self.optimizer = None # Set by Optimizer in init

        self.costs = costs
        if not self.costs:
            self.costs = [TradingCost(), HoldingCost()]

        self.constraints = [] # TBU

        self.percent_short = percent_short
        self.percent_long = percent_long
        self.n_periods_held = n_periods_held
        self.leverage_per_trade = leverage / n_periods_held


    def generate_trade_list(self, holdings, t=dt.datetime.today()):
        w = self._get_trade_weights_for_t(holdings, t)
        u = sum(holdings) * w * self.leverage_per_trade
        
        idx_t = self.forecast_returns.index.get_loc(t)
        if idx_t - self.n_periods_held >= 0:
            # Use holdings_unwind, t_unwind, w_unwind, u_unwind, u_unwind_scaled
            t_unwind = self.forecast_returns.index[idx_t - self.n_periods_held]
            holdings_unwind = self.optimizer.backtest.h.loc[t_unwind]
            w_unwind = self._get_trade_weights_for_t(holdings_unwind, t_unwind)
            u_unwind_pre = sum(holdings_unwind) * w_unwind * self.leverage_per_trade
            u_unwind_scaled = u_unwind_pre * self._cum_returns_to_scale_unwind(t_unwind, t)
            
            u -= u_unwind_scaled

        return u


    def _get_trade_weights_for_t(self, holdings, t):
        n_short = round(self.forecast_returns.shape[1] * self.percent_short)
        n_long = round(self.forecast_returns.shape[1] * self.percent_long)

        prediction = values_in_time(self.forecast_returns, t)
        prediction_sorted = prediction.sort_values()

        short_trades = prediction_sorted.index[:n_short]
        long_trades = prediction_sorted.index[-n_long:]
        
        w = pd.Series(0., index=prediction.index)
        w[short_trades] = -1.
        w[long_trades] = 1.

        w /= sum(abs(w))

        return w


    def _cum_returns_to_scale_unwind(self, t_unwind, t):        
        df = self.optimizer.actual['return'] + 1
        df = df[(df.index >= t_unwind) & (df.index < t)]

        return df.cumprod().iloc[-1]