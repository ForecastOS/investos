import datetime as dt

import pandas as pd

from investos.portfolio.cost_model import BaseCost
from investos.portfolio.strategy import BaseStrategy
from investos.util import values_in_time


class RankLongShort(BaseStrategy):
    """Optimization strategy that builds trade list by going long assets with best return forecasts and short stocks with worst return forecasts.

    Attributes
    ----------
    costs : list[:py:class:`~investos.portfolio.cost_model.base_cost.BaseCost`]
        Cost models evaluated during optimization strategy.
    percent_short : float
        Percent of assets in forecast returns to go short.
    percent_long : float
        Percent of assets in forecast returns to go long.
    leverage : float
        Absolute value of exposure / AUM. Used to calculate holdings.
    n_periods_held : integer
        Number of periods positions held. After n number of periods, positions unwound.
    """

    def __init__(
        self,
        actual_returns: pd.DataFrame,
        metric_to_rank: pd.DataFrame,
        n_periods_held: int = 1,
        leverage: float = 1,
        ratio_long: float = 1,
        ratio_short: float = 1,
        percent_short: float = 0.25,
        percent_long: float = 0.25,
        costs: [BaseCost] = [],
        **kwargs,
    ):
        super().__init__(
            actual_returns=actual_returns,
            costs=costs,
            **kwargs,
        )
        self.metric_to_rank = metric_to_rank
        self.percent_short = percent_short
        self.percent_long = percent_long
        self.n_periods_held = n_periods_held
        self.leverage = leverage
        self.leverage_per_trade = leverage / n_periods_held
        self.ratio_long = ratio_long
        self.ratio_short = ratio_short

        self.metadata_properties = [
            "n_periods_held",
            "leverage",
            "ratio_long",
            "ratio_short",
            "percent_long",
            "percent_short",
        ]

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

        idx_t = self.metric_to_rank.index.get_loc(t)
        positions_saved = self.backtest_controller.results.h.shape[0]

        if positions_saved >= self.n_periods_held:
            # Use holdings_unwind, t_unwind, w_unwind, u_unwind, u_unwind_scaled
            t_unwind = self.metric_to_rank.index[idx_t - self.n_periods_held]
            holdings_unwind = self.backtest_controller.results.h.loc[t_unwind]
            w_unwind = self._get_trade_weights_for_t(holdings_unwind, t_unwind)
            u_unwind_pre = sum(holdings_unwind) * w_unwind * self.leverage_per_trade
            u_unwind_scaled = u_unwind_pre * self._cum_returns_to_scale_unwind(
                t_unwind, t
            )

            u -= u_unwind_scaled

        return u

    def _get_trade_weights_for_t(self, holdings: pd.Series, t: dt.datetime):
        n_short = round(self.metric_to_rank.shape[1] * self.percent_short)
        n_long = round(self.metric_to_rank.shape[1] * self.percent_long)

        prediction = values_in_time(self.metric_to_rank, t)
        prediction_sorted = prediction.sort_values()

        short_trades = prediction_sorted.index[:n_short]
        long_trades = prediction_sorted.index[-n_long:]

        w = pd.Series(0.0, index=prediction.index)
        w[short_trades] = (
            -1.0 * self.ratio_short * (self.percent_long / self.percent_short)
        )
        w[long_trades] = 1.0 * self.ratio_long

        w /= sum(abs(w))

        return w

    def _cum_returns_to_scale_unwind(self, t_unwind: dt.datetime, t: dt.datetime):
        df = self.actual_returns + 1
        df = df[(df.index >= t_unwind) & (df.index < t)]

        return df.cumprod().iloc[-1]
