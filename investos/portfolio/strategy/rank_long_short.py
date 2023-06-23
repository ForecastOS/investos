import pandas as pd
import datetime as dt

from investos.portfolio.strategy import BaseStrategy
from investos.portfolio.cost_model import TradingCost, HoldingCost, BaseCost
from investos.util import values_in_time

class RankLongShort(BaseStrategy):
    """Optimization strategy that builds trade list by going long assets with best return forecasts and short stocks with worst return forecasts.


    Attributes
    ----------
    costs : list[:py:class:`~investos.portfolio.cost_model.base_cost.BaseCost`]
        Cost models evaluated during optimization strategy.
    constraints : list[TBU]
        Constraints applied for optimization strategy. Defaults to empty list.
    percent_short : float
        Percent of assets in forecast returns to go short.
    percent_long : float
        Percent of assets in forecast returns to go long.
    leverage : float
        Absolute value of exposure / AUM. Used to calculate holdings.
    n_periods_held : integer
        Number of periods positions held. After n number of periods, positions unwound.
    """

    def __init__(self, n_periods_held: int = 1, leverage: float = 1, percent_short: float = 0.25, percent_long: float = 0.25, costs: list[BaseCost] = []):
        self.forecast_returns = None # Set by Controller in init
        self.optimizer = None # Set by Controller in init

        self.costs = costs
        if not self.costs:
            self.costs = [TradingCost(), HoldingCost()]

        self.constraints = []
        self.risk_model = None

        self.percent_short = percent_short
        self.percent_long = percent_long
        self.n_periods_held = n_periods_held
        self.leverage_per_trade = leverage / n_periods_held


    def generate_trade_list(self, holdings: pd.Series, t: dt.datetime) -> pd.Series:
        """Calculates and returns trade list (in units of currency passed in) by going long top :py:attr:`~investos.portfolio.strategy.rank_long_short.RankLongShort.percent_long` assets and short bottom :py:attr:`~investos.portfolio.strategy.rank_long_short.RankLongShort.percent_short` assets.

        Parameters
        ----------
        holdings : pandas.Series
            Holdings at beginning of period `t`.
        t : datetime.datetime
            The datetime for associated holdings `holdings`.
        """
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


    def _get_trade_weights_for_t(self, holdings: pd.Series, t: dt.datetime):
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


    def _cum_returns_to_scale_unwind(self, t_unwind: dt.datetime, t: dt.datetime):        
        df = self.optimizer.actual['return'] + 1
        df = df[(df.index >= t_unwind) & (df.index < t)]

        return df.cumprod().iloc[-1]