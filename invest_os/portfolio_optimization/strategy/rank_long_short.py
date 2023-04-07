import pandas as pd
import datetime as dt

from invest_os.portfolio_optimization.strategy import BaseStrategy
from invest_os.util import values_in_time

class RankLongShort(BaseStrategy):
    def __init__(self, n_periods_held=1, leverage=1, percent_short=0.25, percent_long=0.25):
        self.forecast_returns = None # Set by Optimizer in init

        self.costs = []
        self.constraints = []

        self.n_periods_held = n_periods_held
        self.leverage = leverage
        self.leverage_per_trade = n_periods_held / leverage

        self.percent_short = percent_short
        self.percent_long = percent_long

        self.trade_vectors = []

    def generate_trade_list(self, holdings, t=dt.datetime.today()):
        n_short = round(self.forecast_returns.shape[1] * self.percent_short)
        n_long = round(self.forecast_returns.shape[1] * self.percent_long)

        prediction = values_in_time(self.forecast_returns, t)
        prediction_sorted = prediction.sort_values()

        short_trades = prediction_sorted.index[:n_short]
        long_trades = prediction_sorted.index[-n_long:]
        
        u = pd.Series(0., index=prediction.index)
        u[short_trades] = -1.
        u[long_trades] = 1.
        u /= sum(abs(u))
        u = sum(holdings) * u * self.leverage_per_trade

        self.trade_vectors.append(u)
        
        # Unwind trades held for n periods
        # TBU: need to adjust for returns!!! How is this done elsewhere?
        # Maybe can do based on t index? Cleaner way to do this
        if len(self.trade_vectors) > self.n_periods_held:
            u += self.trade_vectors.pop(0) * -1

        return u